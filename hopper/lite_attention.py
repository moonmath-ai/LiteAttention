"""
LiteAttention: A lightweight Flash Attention 3 wrapper with skip list optimization.

This module provides a clean interface for Flash Attention 3 with internal management
of read and write skip lists, hiding the complexity from users.

Skip List Data Structure:
=========================
The skip list is a key optimization that tracks which attention tiles can be skipped
during computation. It uses a compact representation to store ranges of tiles to compute.

Format:
-------
Shape: [2, batch, heads, qtiles,  ktiles + 2]
- Dimension 0 (size 2): Alternates between read_list and write_list based on phase
- Dimension 1: Batch dimension
- Dimension 2: Attention heads
- Dimension 3: Query tiles (rows of the attention matrix)
- Dimension 4: Key tiles + 1 (the +1 is for storing the list length)

The format depends on the `reverse_skip_list` flag:

**When reverse_skip_list == True (default):**
Each entry format: [length, end_n, start_n, ..., end_1, start_1, end_0, start_0, uninitialized...]

The relationship between start and end depends on the phase:
- When self._phase == 1: start_x < end_x
- When self._phase == 0: start_x > end_x

To compute the actual range of tiles:
```python
step = 1 if self._phase else -1
for tile_idx in range(start=start_x + step, end=end_x + step, step=step):
    # Compute this tile
```

Example (reverse_skip_list=True, phase=1):
skip_list[0, 0, 0, 0, :] = [4, 99, 50, 30, 0, ?, ?, ...]
- Range 1: start=50, end=99, step=1 → compute tiles 51, 52, ..., 99 (49 tiles)
- Range 0: start=0, end=30, step=1 → compute tiles 1, 2, ..., 30 (30 tiles)

**When reverse_skip_list == False:**
Each entry format: [length, start_0, end_0, ..., start_n, end_n, uninitialized...]

Always: start_x > end_x
The range is: range(start=start_x, end=end_x, step=-1)

Example (reverse_skip_list=False):
skip_list[0, 0, 0, 0, :] = [4, 99, 50, 30, 0, ?, ?, ...]
- Range 0: start=99, end=50 → compute tiles 99, 98, ..., 50 (50 tiles)
- Range 1: start=30, end=0 → compute tiles 30, 29, ..., 0 (31 tiles)

Must-Do List:
=============
An optional list that forces certain tile ranges to be computed regardless of threshold.
Useful for ensuring specific attention patterns are always included.

Format:
-------
Input: 1D list of sequence indices [seq_start_0, seq_end_0, seq_start_1, seq_end_1, ...]
where end indices are EXCLUSIVE (Python-style ranges: [start, end)).

The function automatically converts these to tile indices and prepares them for the kernel.

Conversion:
- Sequence indices are converted to tile indices by dividing by tile size
- Start indices (inclusive) use floor division: start // tile_size
- End indices (exclusive) use ceiling division: ceil(end / tile_size)
- End indices remain exclusive in the output

Example:
must_do_list = [0, 128, 500, 640]  # Compute sequence positions [0, 128) and [500, 640)
# This means positions 0-127 and 500-639
# If tile size is 128, this converts to [4, 0, 1, 3, 5] internally
# Where tile ranges are [0, 1) and [3, 5), meaning tile 0 and tiles 3-4
"""

import math
import os
import typing
import warnings
from dataclasses import dataclass

import torch

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from pathlib import Path
from typing import Optional, Tuple, Union

# Import the C++ extension to register operators with PyTorch
import lite_attention._C  # noqa: F401
import structlog
import torch.nn as nn
import torch.nn.functional as F

from ._internal.flash_attn_interface import flash_attn_func
from .calibrated_module import (
    CalibratedCalibConfig,
    CalibratedRunConfig,
    ConfigList,
    ConfigurableModule,
    ModuleRegistry,
)

_lite_attention_ops = torch.ops.lite_attention

log = structlog.get_logger()


@dataclass
class LiteAttentionRunConfig(CalibratedRunConfig):
    """Runtime configuration for LiteAttention threshold."""

    threshold: float

    @classmethod
    def default(cls) -> Self:
        return cls(threshold=-10.0)


@dataclass
class LiteAttentionDisabledConfig(LiteAttentionRunConfig):
    """Runtime config that disables skipping for this timestep (regular attention)."""

    threshold: float = 0.0


@dataclass
class LiteAttentionReplayConfig(CalibratedRunConfig):
    """Runtime config for replay mode: load pre-computed skip lists from a capture file.

    The capture file is a .pt file produced by ``enable_capture`` + ``save()``.

    Two sub-modes:

    1. **Skip-list replay** (``threshold=None``, default): the stored write-lists
       are shifted by one timestep and fed back as read-lists, bypassing
       threshold computation entirely.  Requires the capture to include
       ``skip_lists`` (i.e. captured with ``qk_block_map=True`` or
       ``attn_map=True``).

    2. **QK-map replay** (``threshold`` is set): skip lists are *computed* at
       load time from the captured ``qk_block_map`` and the given threshold.
       This lets you replay with a different threshold than the original
       capture without re-running the model.  Requires the capture to include
       ``qk_block_map``.

    Attributes:
        skip_list_file: Path to the .pt capture file.
        write_next: If True (default) the kernel still writes a write-list each
            step.  Set to False to signal that write is unnecessary.
            NOTE: the CUDA kernel always writes regardless; this flag is a
            placeholder for a future C/CUDA optimisation that would skip the
            write entirely.
        threshold: If set, compute skip lists from ``qk_block_map`` using this
            threshold instead of using the captured ``skip_lists`` directly.
            Values are in log2 scale (must be ≤ 0 in non-debug mode), same
            semantics as ``LiteAttentionRunConfig.threshold``.
    """

    skip_list_file: str = ""
    write_next: bool = True
    threshold: float | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        """Serialize to dict, omitting None values (not TOML-serializable)."""
        d = super().to_dict()
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def default(cls):
        raise NotImplementedError(
            "LiteAttentionReplayConfig requires explicit configuration"
        )


@dataclass
class LiteAttentionCalibConfig(CalibratedCalibConfig):
    """Calibration configuration for finding optimal threshold."""

    metric: typing.Literal["Cossim", "L1", "RMSE"] = "L1"
    target_error: float = 0.01


class LiteAttention(nn.Module, ConfigurableModule):
    """
    A lightweight attention class that encapsulates Flash Attention 3 with optimized skip lists.

    This class manages read and write skip lists internally, providing a clean interface for users.
    The skip list optimization allows the attention computation to skip tiles (blocks) of the
    attention matrix that have low contribution, significantly reducing computation time.

    How It Works:
    -------------
    1. The attention matrix Q@K^T is computed in tiles (blocks)
    2. Each tile's maximum score is compared against a threshold
    3. Tiles below the threshold are skipped in subsequent computations
    4. A "write list" is generated for the current forward pass
    5. This write list becomes the "read list" for the next forward pass
    6. The process alternates between two buffers for efficiency

    Args:
        enable_skipping (bool, optional): Whether to enable skip list optimizations.
            Defaults to True. When False, performs standard Flash Attention.
        threshold (float, optional): Log-space threshold for skipping tiles. Defaults to -10.0.
            Tiles with max(log-attention-score) < threshold will be skipped.
            Must be negative in non-debug mode. Lower values = more aggressive skipping.
        max_batch_size (int, optional): Maximum batch size to pre-allocate memory for.
            Defaults to 2. Actual batch size can be smaller but not larger.
        reverse_skip_list (bool, optional): Whether to use reversed skip list format.
            Defaults to True. Affects the ordering of ranges in skip lists.
        config (LiteAttentionRunConfig | LiteAttentionCalibConfig, optional): Configuration
            for threshold or calibration. Supports per-timestep configs via ConfigList.
            If LiteAttentionCalibConfig, runs calibration to find optimal threshold.

    Attributes:
        enable_skipping (bool): Current state of skip optimization
        threshold (float): Current threshold value
        read_list (torch.Tensor): Current read skip list (read-only property)
        write_list (torch.Tensor): Current write skip list (read-only property)

    Example:
        >>> # in the common case, the config is managed with the LiteAttentionRegistry
        >>> lite_attn = LiteAttention()

        >>> # In case you want to run with a specific threshold
        >>> lite_attn = LiteAttention(threshold=-5.0)
        >>> output = lite_attn(query, key, value)

        >>> # With must-do list to force certain sequence ranges
        >>> lite_attn = LiteAttention(enable_skipping=True, threshold=-8.0)
        >>> # Force computation of sequence positions [0, 128) and [500, 640) (exclusive end)
        >>> must_do = [0, 128, 500, 640]
        >>> output = lite_attn(query, key, value, must_do_list=must_do)

        >>> # Disable skipping for specific forward pass
        >>> lite_attn.enable_skip_optimization(False)
        >>> output = lite_attn(query, key, value)
    """

    run_config_type = LiteAttentionRunConfig

    def __init__(
        self,
        enable_skipping: bool = True,
        threshold: float | None = None,
        max_batch_size: int = 2,
        reverse_skip_list: bool = True,
        use_int8: bool = False,
        config: LiteAttentionRunConfig | LiteAttentionCalibConfig | None = None,
    ):
        nn.Module.__init__(self)
        if threshold is not None and config is not None:
            raise ValueError("Cannot specify both 'threshold' and 'config'")
        if threshold is not None:
            config = LiteAttentionRunConfig(threshold=threshold)
        ConfigurableModule.__init__(self, config)
        # Internal skip list management
        self._skip_list = None  # Shape: [2, max_batch_size, heads, qtiles, ktiles+2]
        self._phase = 0  # Alternates between 0 and 1 for double-buffering
        self.reverse_skip_list = reverse_skip_list  # Controls skip list format
        self.use_int8 = use_int8  # Whether using int8 quantization

        # Cache of last tensor properties (used to detect when reinitialization is needed)
        self._last_batch_size = None  # Actual batch size used (not max_batch_size)
        # Sequence lengths used to size the skip list.
        # For self-attention: (q_len, k_len) == (seq_len, seq_len)
        # For rectangular attention (e.g. KV-sharded sequence parallel): q_len != k_len
        self._last_seq_len = None  # Tuple[int, int] = (q_len, k_len)
        self._last_head_dim = None  # Head dimension
        self._last_v_colmajor = None  # Value tensor layout
        self._last_dtype = None  # Data type (fp16, bf16, fp32)
        self._last_device = None  # Device (cuda:0, cuda:1, etc.)
        self._last_num_heads = None  # Number of attention heads
        # Statistics
        self._last_percentage = 0.0  # Percentage of tiles computed in last pass
        self._last_use_int8 = use_int8  # Whether using int8 quantization in last pass

        # Public configuration
        self.enable_skipping = enable_skipping
        self.max_batch_size = max_batch_size

        # Replay mode state (set via _hydrate_replay_data on the registry)
        self._replay_skip_lists = (
            None  # list[Tensor] | None — one read_list per active step
        )
        self._replay_step_counter = 0  # counts active (non-disabled) replay steps

        # Debug capture state (set via _enable_capture, cleared via _disable_capture)
        self._capture_enabled = False  # pct for all heads/timesteps
        self._captured_pct = []  # [{timestep, pct_per_head, threshold}]

        # universal map capture:
        self._capture_map_heads = None  # list[int] | None — None means all
        self._capture_map_batches = None  # list[int] | None — None means all
        self._captured_maps = []  # [{timestep, skip_list, attn_map}]

        # Skip list capture (write_list snapshots, no QK recomputation):
        self._capture_skip_lists = False

        # qk max map capture:
        self._capture_qk_block_map = False

        # Detailed attn map capture:
        self._capture_attn_map = False
        self._capture_attn_map_timesteps = None  # set[int] | None — None means all
        self._capture_attn_map_res = 256

        # Stats capture:
        self._capture_stats = False
        self._stats_count = 0
        self._stats_mean = (
            None  # lazily initialized: [n_batches, n_heads, seq_q, seq_k] float32 CPU
        )
        self._stats_m2 = (
            None  # [n_batches, n_heads, seq_q, seq_k] float32 CPU (Welford's M2)
        )
        self._stats_max = None  # [n_batches, n_heads, seq_q, seq_k] float32 CPU
        self._stats_min = None  # [n_batches, n_heads, seq_q, seq_k] float32 CPU

    @staticmethod
    def ceil_div(x, y):
        """Ceiling division utility function."""
        return (x + y - 1) // y

    @staticmethod
    def calc_percentage_per_head(read_list: torch.Tensor) -> float:
        """
        Calculate the percentage of non-skipped (computed) attention tiles per head.

        This function analyzes the skip list to determine what fraction of attention tiles
        were actually computed vs. skipped. The skip list stores ranges of tiles to compute,
        and this function calculates the total number of tiles covered by those ranges.

        Args:
            read_list (torch.Tensor): Skip list of shape [batch, heads, qtiles,  ktiles + 2]
                Each entry: [length, start_0, end_0, start_1, end_1, ...]

        Returns:
            torch.Tensor: Percentage of computed tiles per query tile, per head, per batch.
                Shape: [batch, heads, qtiles]
                Values range from 0.0 (all skipped) to 1.0 (none skipped)

        Algorithm:
        ----------
        1. Remove the length field (first element) from each skip list entry
        2. Reshape pairs of (start, end) indices into explicit ranges
        3. Calculate the size of each range (end - start)
        4. Sum up all range sizes to get total computed tiles
        5. Divide by total number of k-tiles to get percentage
        """

        read_list = read_list.to(torch.int64)
        # Remove the first element (the length of the skip list)
        # [batch, heads, qtiles,  ktiles + 2] -> [batch, heads, qtiles, ktiles]
        reshaped_read_list = read_list[..., 1:]  # [batch, heads, qtiles, ktiles]

        # Pad last dimension to be even (required for pairing start/end indices)
        # [batch, heads, qtiles, ktiles] -> [batch, heads, qtiles, ktiles + (ktiles % 2)]
        if reshaped_read_list.shape[-1] % 2 != 0:
            # Pad with 0 if uneven (will not affect the calculation)
            padding_shape = list(reshaped_read_list.shape)
            padding_shape[-1] = 1
            padding = torch.zeros(
                padding_shape,
                dtype=reshaped_read_list.dtype,
                device=reshaped_read_list.device,
            )
            reshaped_read_list = torch.cat([reshaped_read_list, padding], dim=-1)

        # Reshape to pair up (start, end) indices explicitly
        # [batch, heads, qtiles, ktiles + (ktiles % 2)] -> [batch, heads, qtiles, num_ranges, 2]
        # where num_ranges = (ktiles + (ktiles % 2)) / 2
        reshaped_read_list = reshaped_read_list.view(
            reshaped_read_list.shape[0],
            reshaped_read_list.shape[1],
            reshaped_read_list.shape[2],
            -1,
            2,
        )

        # Calculate the size of each range: |end - start|
        # Works for both reversed (start > end) and normal (start < end) formats
        # range_sizes: [batch, heads, qtiles, num_ranges]
        range_sizes = (reshaped_read_list[..., 1] - reshaped_read_list[..., 0]).abs()

        # Cumulative sum gives us total tiles computed up to each range
        # not_skipped_per_head: [batch, heads, qtiles, num_ranges]
        not_skipped_per_head = range_sizes.cumsum(dim=-1)

        # Get the actual number of valid ranges from the length field
        # skip_list_sizes: [batch, heads, qtiles]
        # Length is always even, divide by 2 to get number of (start, end) pairs
        skip_list_sizes = (read_list[:, :, :, 0] - 1) // 2

        # Extract the cumulative sum at the last valid range position
        # real_not_skipped_per_head: [batch, heads, qtiles, num_ranges] -> [batch, heads, qtiles]
        real_not_skipped_per_head = torch.gather(
            not_skipped_per_head, dim=-1, index=skip_list_sizes.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate percentage: (tiles computed) / (total tiles)
        num_of_k_tiles = (
            read_list.shape[-1] - 2
        )  # fixme: this is wrong when we use max_len
        return real_not_skipped_per_head / num_of_k_tiles

    @staticmethod
    def calc_percentage(read_list: torch.Tensor) -> float:
        """
        Calculate the average percentage of non-skipped attention computations.

        Args:
            read_list (torch.Tensor): Skip list of shape [batch, heads, qtiles,  ktiles + 2]

        Returns:
            float: Average percentage across all query tiles, heads, and batches.
                Value ranges from 0.0 (all skipped) to 1.0 (none skipped)
        """
        return LiteAttention.calc_percentage_per_head(read_list).mean()

    @staticmethod
    def calc_error(quant_o, fa2_o):
        if quant_o.shape[-2] > 200000:
            quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
        x, xx = quant_o.float(), fa2_o.float()
        sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
        l1 = ((x - xx).abs().sum() / xx.abs().sum()).item()
        rmse = torch.sqrt(torch.mean((x - xx) ** 2)).item()
        return {"Cossim": 1.0 - sim, "L1": l1, "RMSE": rmse}

    @staticmethod
    def get_MN(head_dim, dtype, v_colmajor=False, is_skipable=True):
        """
        Get the tile sizes (block dimensions) for attention computation.

        These tile sizes determine how the attention matrix is divided into blocks
        for computation. Different head dimensions and data types require different
        tile sizes for optimal performance.

        This function directly calls the C++ `tile_size_fwd_sm90()` function from
        `tile_size.h` to ensure consistency between Python and CUDA kernel tile sizes.

        Args:
            head_dim (int): Dimension of each attention head
            dtype (torch.dtype): Data type of the tensors (fp16, bf16, fp32, int8)
            v_colmajor (bool, optional): Whether value tensor is column-major. Defaults to False.
            is_int8 (bool, optional): Whether using int8 quantization. Defaults to False.

        Returns:
            tuple[int, int]: (kBlockM, kBlockN) where:
                - kBlockM: Number of rows per tile (query dimension)
                - kBlockN: Number of columns per tile (key dimension)
        """
        is_int8 = dtype == torch.int8
        element_size = dtype.itemsize
        # Call C++ tile_size_fwd_sm90 function
        # Arguments: headdim, headdim_v, is_causal, is_local, element_size,
        #            v_colmajor, paged_kv_non_TMA, softcap, is_skipable, is_int8
        # Returns: [kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap]
        result = _lite_attention_ops.get_tile_size_fwd_sm90(
            head_dim,  # headdim
            head_dim,  # headdim_v (same as headdim for standard attention)
            False,  # is_causal (not relevant for skipable case)
            False,  # is_local
            element_size,  # element_size (2 for fp16/bf16, 4 for fp32)
            v_colmajor,  # v_colmajor
            False,  # paged_kv_non_TMA
            False,  # softcap
            is_skipable,  # is_skipable
            is_int8,  # is_int8
        )
        kBlockM, kBlockN = result[0], result[1]
        return kBlockM, kBlockN

    @staticmethod
    def init_skip_list(
        batch,
        seq_len,
        heads,
        head_dim,
        v_colmajor,
        dtype,
        device,
        must_skip_list: list = None,
        reverse_skip_list: bool = True,
    ) -> torch.Tensor:
        """
        Initialize skip list tensors with default "compute all tiles" configuration.

        The skip list is initialized to compute all tiles by default. As the forward pass
        executes, it will be updated based on which tiles exceed the threshold.

        Tile Dimensions:
        ---------------
        The attention matrix Q@K^T is divided into tiles (blocks) for computation:
        - qtiles: Number of tiles along the query dimension (rows of Q@K^T)
          Calculated as: ceil(seq_len / kBlockM) where kBlockM is the tile height
        - ktiles: Number of tiles along the key dimension (columns of Q@K^T)
          Calculated as: ceil(seq_len / kBlockN) where kBlockN is the tile width

        Args:
            batch (int): Batch size
            seq_len (int): Sequence length
            heads (int): Number of attention heads
            head_dim (int): Dimension of each head
            v_colmajor (bool): Whether value tensor is column-major layout
            dtype (torch.dtype): Data type of the tensors (fp16, bf16, fp32)
            device (torch.device): Device to allocate tensors on
            must_skip_list (list, optional): List of sequence ranges to always skip.
        Returns:
            torch.Tensor: Initialized skip list of shape [2, batch, heads, qtiles,  ktiles + 2]
                where qtiles and ktiles are the number of tiles along query and key dimensions.
                Dtype: torch.int16

        Initial Configuration:
        ---------------------
        The skip list is initialized with a single range covering all tiles:
        [2, ktiles-1, -1, ?, ?, ...]

        Where:
        - 2: Length (one range = 2 elements: start and end)
        - ktiles-1: End of range (highest tile index)
        - -1: Start of range (will wrap to 0 in kernel iteration)

        This corresponds to iterating: for i in range(ktiles-1, -1, -1)
        Which computes all tiles: ktiles-1, ktiles-2, ..., 1, 0 (inclusive)
        """

        # Get tile dimensions for this configuration
        # kBlockM: number of query rows per tile
        # kBlockN: number of key columns per tile
        kBlockM, kBlockN = LiteAttention.get_MN(head_dim, dtype, v_colmajor)

        # Support both square and rectangular attention.
        # - For standard self-attention, `seq_len` is an int.
        # - For rectangular attention, pass `seq_len=(q_len, k_len)`.
        if isinstance(seq_len, (tuple, list)):
            q_len, k_len = int(seq_len[0]), int(seq_len[1])
        else:
            q_len = k_len = int(seq_len)

        # Calculate number of tiles needed to cover the attention matrix
        # qtiles: number of tiles along query dimension (rows of Q@K^T)
        qtiles = LiteAttention.ceil_div(q_len, kBlockM)
        # ktiles: number of tiles along key dimension (columns of Q@K^T)
        ktiles = LiteAttention.ceil_div(k_len, kBlockN)

        # Allocate memory for skip list data structure
        # Shape explained:
        #   [0]: Size 2 for double-buffering (alternates between read_list and write_list)
        #   [1]: Batch dimension
        #   [2]: Head dimension
        #   [3]: Query tiles dimension
        #   [4]:  ktiles + 2 (the +1 stores the list length at index 0)
        skip_list = torch.empty(
            2, batch, heads, qtiles, ktiles + 2, dtype=torch.int16, device=device
        )

        if must_skip_list is not None:
            tile_indices = LiteAttention.convert_sequence_indices_to_tile_indices(
                "must_skip_list",
                must_skip_list,
                kBlockN,
                k_len,
            )

            # convert from skip-ranges to do-ranges for read list
            tile_indices.pop(0) if tile_indices[0] == 0 else tile_indices.insert(0, 0)
            tile_indices.pop() if tile_indices[-1] == ktiles else tile_indices.append(
                ktiles
            )

            tile_indices = [len(tile_indices)] + list(reversed(tile_indices))
            skip_list[0, :, :, :, : len(tile_indices)] = torch.tensor(
                tile_indices, dtype=torch.int16, device=device
            )
        else:
            # Initialize first buffer with "compute all tiles" configuration
            # [2, ktiles-1, -1] means: length=2, one range from ktiles-1 down to 0 (via -1)
            skip_list[0, :, :, :, 0:3] = torch.tensor(
                [2, ktiles - 1, -1], dtype=torch.int16, device=device
            )

            # Note: Second buffer (skip_list[1]) is left uninitialized and will be populated
            # during the first forward pass

        return skip_list

    def _init_skip_list(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        must_skip_list: list = None,
    ) -> torch.Tensor:
        """
        Initialize skip list tensors based on query and value tensor shapes.

        This is a wrapper around the static init_skip_list method that extracts
        all necessary parameters from the input tensors.

        Args:
            query (torch.Tensor): Query tensor of shape [batch, seq_len, heads, head_dim]
            value (torch.Tensor): Value tensor (used to determine memory layout)
            must_skip_list (list, optional): List of sequence ranges to always skip.
        Returns:
            torch.Tensor: Initialized skip list

        Note:
            The skip list is allocated with max_batch_size (not actual batch size) to
            avoid reallocation when batch size varies across forward passes.
        """
        batch = value.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]
        heads = value.shape[2]
        head_dim = value.shape[3]
        assert batch <= self.max_batch_size, (
            "batch size must be less than or equal to max_batch_size (modify max_batch_size in LiteAttention constructor)"
        )

        # Determine if value tensor is column-major (affects tile size selection)
        v_colmajor = value.shape[-3] == head_dim
        dtype = torch.int8 if self.use_int8 else query.dtype
        device = query.device

        # Allocate for max_batch_size to avoid reallocation on batch size changes.
        return LiteAttention.init_skip_list(
            self.max_batch_size,
            (q_len, k_len),
            heads,
            head_dim,
            v_colmajor,
            dtype,
            device,
            must_skip_list,
            self.reverse_skip_list,
        )

    def _get_read_write_lists(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        must_skip_list: list = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the current read and write skip lists for this attention forward pass.

        This method manages the double-buffering of skip lists, alternating between
        two buffers to enable read/write in a single pass. It also handles initialization
        and reinitialization when tensor properties change.

        Args:
            query (torch.Tensor): Query tensor [batch, seq_len, heads, head_dim]
            value (torch.Tensor): Value tensor (used for layout detection)
            must_skip_list (list, optional): List of sequence ranges to always skip.
        Returns:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - read_list: Skip list from previous pass (what to compute this pass)
                - write_list: Skip list to write to (for next pass)
                Returns (None, None) if skipping is disabled.

        Double-Buffering Mechanism:
        --------------------------
        - Phase 0: read from buffer[0], write to buffer[1]
        - Phase 1: read from buffer[1], write to buffer[0]
        - Phase alternates each forward pass

        Reinitialization Triggers:
        -------------------------
        Skip list is reinitialized if any of these change:
        - Sequence length
        - Number of heads
        - Head dimension
        - Data type
        - Device
        - Value tensor layout (row/column major)
        """

        # If skipping disabled, return None (standard Flash Attention)
        if not self.enable_skipping:
            return None, None

        # Backward-compat: older callers pass (query, value) only.
        # In that case `key` is actually the value tensor, and key/value lengths match.
        if value is None:
            value = key
            key = value

        # --- Replay mode: use pre-computed skip lists ---
        if self._replay_skip_lists is not None:
            replay_idx = min(
                self._replay_step_counter, len(self._replay_skip_lists) - 1
            )
            read_list = self._replay_skip_lists[replay_idx]

            # Expand batch dimension if needed (capture may have batch=1)
            batch_size = query.shape[0]
            if read_list.shape[0] < batch_size:
                read_list = read_list.expand(batch_size, -1, -1, -1).contiguous()
            read_list = read_list.to(query.device)

            # Kernel always needs a write buffer even if we discard it
            write_list = torch.empty_like(read_list)

            # Set phase so forward() passes the correct value to the kernel.
            # Normal flow: T=0 → phase flips 0→1 → kernel sees True
            #              T=1 → phase flips 1→0 → kernel sees False
            # forward() checks (self._phase == 1), so:
            self._phase = 1 if (replay_idx % 2 == 0) else 0

            self._replay_step_counter += 1

            # Track metadata (needed by _maybe_capture and other downstream code)
            head_dim = query.shape[-1]
            v_colmajor = value.shape[-3] == head_dim
            self._last_seq_len = (int(query.shape[1]), int(key.shape[1]))
            self._last_head_dim = head_dim
            self._last_v_colmajor = v_colmajor
            self._last_dtype = torch.int8 if self.use_int8 else query.dtype
            self._last_device = query.device
            self._last_num_heads = query.shape[2]
            self._last_batch_size = batch_size

            return read_list, write_list

        # attributes we check in the decision to REINITIALIZE the skip list
        current_seq_len = (int(query.shape[1]), int(key.shape[1]))
        head_dim = query.shape[-1]
        current_head_dim = head_dim
        current_num_heads = query.shape[2]
        v_colmajor = value.shape[-3] == head_dim
        dtype = torch.int8 if self.use_int8 else query.dtype
        device = query.device

        should_reinitialize = (
            self._skip_list is None
            or self._last_seq_len != current_seq_len
            or self._skip_list.device != query.device
            or self._last_head_dim != current_head_dim
            or self._last_v_colmajor != v_colmajor
            or self._last_dtype != dtype
            or self._last_device != device
            or self._last_num_heads != current_num_heads
        )

        if self.use_int8 != self._last_use_int8 and not should_reinitialize:
            should_reinitialize = LiteAttention.get_MN(
                head_dim, torch.int8, v_colmajor
            ) != LiteAttention.get_MN(head_dim, dtype, v_colmajor)
            self._last_use_int8 = self.use_int8

        # Initialize or reinitialize skip list if needed
        # we always enter this in the first call
        if should_reinitialize:
            # initialize the skip list (actually allocate the memory)
            self._skip_list = self._init_skip_list(query, key, value, must_skip_list)
            # ditermines which part of self._skip_list to use for read_list and write_list
            self._phase = 0

            # update the last attributes to the current values
            self._last_seq_len = current_seq_len
            self._last_head_dim = current_head_dim
            self._last_v_colmajor = v_colmajor
            self._last_dtype = dtype
            self._last_device = device
            self._last_num_heads = current_num_heads
            self._last_batch_size = query.shape[0]
            self._last_use_int8 = self.use_int8

            if os.getenv("LITE_ATTENTION_VERBOSE", "FALSE") != "FALSE":
                print(f"[Warning]: reinitialized skip list during the forward pass")

        # Alternate between the two skip list buffers
        if self._phase == 0:
            read_list = self._skip_list[0]
            write_list = self._skip_list[1]
            # switch so the current read_list and write_list roles would switch
            self._phase = 1
        else:
            read_list = self._skip_list[1]
            write_list = self._skip_list[0]
            # switch so the current read_list and write_list roles would switch
            self._phase = 0

        return read_list, write_list

    @staticmethod
    def _expand_must_do_list(
        must_do_list, list_shape, query, value, use_int8: bool = False
    ):
        """
        Convert user-provided must-do list from sequence indices to tile indices.

        The must-do list allows users to force computation of specific sequence ranges
        regardless of the threshold. This is useful for ensuring critical attention
        patterns are never skipped (e.g., attending to special tokens).

        This function converts sequence indices to tile indices and prepares the list
        for the kernel.

        Args:
            must_do_list (list): 1D list of sequence indices defining ranges to always compute.
                Format: [seq_start_0, seq_end_0, seq_start_1, seq_end_1, ...]
                where end indices are EXCLUSIVE (Python-style ranges).
                Example: [0, 128, 500, 640] means compute positions [0, 128) and [500, 640)
                         which is positions 0-127 and 500-639.
            list_shape (tuple): Shape of the skip list (not used in current implementation)
            query (torch.Tensor): Query tensor (used for device and dimension info)
            value (torch.Tensor): Value tensor (used to determine memory layout)

        Returns:
            torch.Tensor: Must-do list in internal format [length, tile_start_0, tile_end_0, ...]
                Shape: [length + 1] where length = len(must_do_list)
                Tile indices are also in exclusive format for end indices.

        Raises:
            ValueError: If must_do_list has odd number of elements
            ValueError: If any start or end index is negative
            ValueError: If any range is empty or invalid (start >= end)

        Conversion Algorithm:
        --------------------
        Sequence indices are converted to tile indices as follows:
        - Start indices (even positions): INCLUSIVE, use floor division
          tile_start = seq_start // kBlockN
        - End indices (odd positions): EXCLUSIVE, use ceiling division
          tile_end = ceil(seq_end / kBlockN)

        Example:
        -------
        If kBlockN=128 (tile size) and input is [0, 128, 500, 640]:
        - seq_start_0=0   → tile_start_0 = 0 // 128 = 0
        - seq_end_0=128   → tile_end_0 = ceil(128/128) = 1 (exclusive)
        - seq_start_1=500 → tile_start_1 = 500 // 128 = 3
        - seq_end_1=640   → tile_end_1 = ceil(640/128) = 5 (exclusive)
        Output: [4, 0, 1, 3, 5]

        This means tile ranges [0, 1) and [3, 5), i.e., tile 0 and tiles 3-4

        The kernel will merge this with the skip list to ensure these tile ranges
        are always computed.
        """

        # Extract tensor properties needed for tile size calculation
        head_dim = query.shape[-1]
        v_colmajor = value.shape[-3] == head_dim
        dtype = torch.int8 if use_int8 else query.dtype
        device = query.device

        # Get tile dimensions (kBlockM, kBlockN)
        _, k_tile_size = LiteAttention.get_MN(head_dim, dtype, v_colmajor)

        # Prepend the length and convert to tensor
        result = LiteAttention.convert_sequence_indices_to_tile_indices(
            "must_do_list", must_do_list, k_tile_size, value.shape[1]
        )
        return torch.tensor(
            [len(result)] + result, dtype=torch.int16, device=device
        ).contiguous()

    @staticmethod
    def convert_sequence_indices_to_tile_indices(
        list_name: str, sequence_indices: list, k_tile_size: int, seq_len: int
    ) -> list:
        if len(sequence_indices) % 2 != 0:
            raise ValueError(
                f"{list_name} must have an even number of elements (pairs of start/end indices). "
                f"Got {len(sequence_indices)} elements: {sequence_indices}"
            )

        converted_list = []
        for i, seq_idx in enumerate(sequence_indices):
            # Validate index is non-negative
            if seq_idx < 0:
                range_idx = i // 2
                idx_type = "start" if i % 2 == 0 else "end"
                raise ValueError(
                    f"{list_name} range {range_idx}: {idx_type} index must be non-negative. "
                    f"Got {idx_type}={seq_idx}"
                )

            if i % 2 == 0:  # Start index (even position in list, INCLUSIVE)
                # Floor division: start of tile containing this position
                tile_idx = seq_idx // k_tile_size
            else:  # End index (odd position in list, EXCLUSIVE)
                # Validate range is non-empty (start < end)
                start_seq = sequence_indices[i - 1]
                end_seq = seq_idx
                if start_seq >= end_seq:
                    range_idx = (i - 1) // 2
                    raise ValueError(
                        f"{list_name} range {range_idx}: end must be greater than start (exclusive range). "
                        f"Got [{start_seq}, {end_seq}) which is empty or invalid."
                    )

                # validate end index is less than ktile_num
                if end_seq > seq_len:
                    raise ValueError(f"{end_seq} is greater than {seq_len}.")

                # Ceiling division: tile after the last position
                tile_idx = LiteAttention.ceil_div(seq_idx, k_tile_size)
            converted_list.append(tile_idx)

        if len(converted_list) == 0:
            return []

        # merge intersecting ranges:
        merged = []
        s, e = converted_list[:2]
        for a, b in zip(converted_list[2::2], converted_list[3::2]):
            if a <= e:
                e = b
            else:
                merged += [s, e]
                s, e = a, b

        return merged + [s, e]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        return_softmax_lse: bool = False,
        must_do_list: list = None,
        must_skip_list: list = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Flash Attention 3 computation with optional skip list optimization.

        This is the main forward pass method. It handles:
        1. Skip list management (read/write alternation)
        2. Must-do list processing (force specific tiles)
        3. Calling the underlying Flash Attention 3 kernel
        4. Statistics calculation (if verbose mode enabled)

        Args:
            query (torch.Tensor): Query tensor of shape [batch, seq_len, heads, head_dim]
            key (torch.Tensor): Key tensor of shape [batch, seq_len, heads, head_dim]
            value (torch.Tensor): Value tensor of shape [batch, seq_len, heads, head_dim]
            scale (float, optional): Attention scale factor.
                If None, uses 1/sqrt(head_dim). Defaults to None.
            return_softmax_lse (bool, optional): Whether to return log-sum-exp values.
                Defaults to False.
            must_do_list (list, optional): List of sequence ranges to always compute.
                Format: [seq_start_0, seq_end_0, seq_start_1, seq_end_1, ...]
                where end indices are EXCLUSIVE (Python-style ranges: [start, end)).
                Example: [0, 128, 500, 640] forces positions [0, 128) and [500, 640) to be computed.
                Indices are automatically converted to tile indices internally.
                Defaults to None (no forced computation).
            must_skip_list (list, optional): List of sequence ranges to always skip.
                Format: [seq_start_0, seq_end_0, seq_start_1, seq_end_1, ...]
                where end indices are EXCLUSIVE (Python-style ranges: [start, end)).
                Example: [0, 128, 500, 640] skips positions [0, 128) and [500, 640) to be skipped.
                Indices are automatically converted to tile indices internally.
                Defaults to None (no forced skipping).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                If return_softmax_lse=False:
                    - torch.Tensor: Attention output [batch, seq_len, heads, head_dim]
                If return_softmax_lse=True:
                    - Tuple of (output, lse) where lse is the log-sum-exp values

        Notes:
        -----
        - The skip list is updated during each forward pass based on tile scores
        - Set LITE_ATTENTION_VERBOSE=1 environment variable to see skip statistics
        - The method automatically manages skip list initialization and reinitialization

        Example:
        -------
        >>> lite_attn = LiteAttention(threshold=-8.0)
        >>> output = lite_attn(q, k, v)
        >>> # With must-do list to force specific sequence ranges
        >>> # Force computation for positions [0, 128) and [500, 640) (exclusive end)
        >>> output = lite_attn(q, k, v, must_do_list=[0, 128, 500, 640])
        """
        cfg = self.config if self.enable_skipping else None
        disabled = isinstance(cfg, LiteAttentionDisabledConfig)

        # if we are disabled, we temporarily disable skipping
        enable_skipping = self.enable_skipping
        if disabled:
            self.enable_skipping = False

        # Get read and write lists (internal mask management)
        read_list, write_list = self._get_read_write_lists(
            query, key, value, must_skip_list
        )

        if self.enable_skipping and must_do_list is not None:
            # handle must-do list - expand the 1d list to a list per head per batch per qi
            must_do_list_expanded = self._expand_must_do_list(
                must_do_list, write_list.shape, query, value, self.use_int8
            )
        else:
            must_do_list_expanded = None

        self.enable_skipping = enable_skipping

        # For disabled timesteps, _get_read_write_lists returned early without
        # setting metadata that _save_capture needs (head_dim, dtype, seq_len, etc.).
        if disabled and self._last_dtype is None:
            v_colmajor = value.shape[-3] == query.shape[-1]
            self._last_seq_len = (int(query.shape[1]), int(key.shape[1]))
            self._last_head_dim = query.shape[-1]
            self._last_v_colmajor = v_colmajor
            self._last_dtype = torch.int8 if self.use_int8 else query.dtype
            self._last_device = query.device
            self._last_num_heads = query.shape[2]
            self._last_batch_size = query.shape[0]

        # softmax_scale: for INT8 use q_scale (1.44269504089 * scale or / sqrt(head_dim)); else use scale as-is
        head_dim = query.shape[-1]
        softmax_scale = (
            (
                (1.44269504089 / math.sqrt(head_dim))
                if scale is None
                else (1.44269504089 * scale)
            )
            if self.use_int8
            else scale
        )

        if not self.enable_skipping or isinstance(cfg, LiteAttentionDisabledConfig):
            threshold = 0.0  # unused
        elif isinstance(cfg, LiteAttentionReplayConfig):
            # read_list already determines what to compute.  The kernel still
            # performs threshold comparison and writes to write_list.
            # Pass the replay threshold (if set) so the write_list is populated
            # with a meaningful result; otherwise use 0.0 (no-op).
            # TODO: if write_next=False, a C/CUDA optimisation could skip the
            # threshold comparison and write_list population entirely.
            threshold = cfg.threshold if cfg.threshold is not None else 0.0
        elif isinstance(cfg, LiteAttentionCalibConfig):
            temp_list = read_list.clone()

            def calibration_step(curr_th):
                output_old_th = flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=softmax_scale,
                    attn_read_list=read_list,
                    attn_must_do_list=must_do_list_expanded,
                    attn_write_list=write_list,
                    thr=curr_th,
                    return_softmax_lse=return_softmax_lse,
                    reverse_skip_list=self.reverse_skip_list,
                    phase=(self._phase == 1) if self.reverse_skip_list else False,
                    use_int8=self.use_int8,
                )
                # we switch read <-> write manually; we remember to flip phase
                self._phase = 1 - self._phase
                output_new_th = flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=softmax_scale,
                    attn_read_list=write_list,  # this injects the new threshold calculated before.
                    attn_must_do_list=must_do_list_expanded,
                    attn_write_list=temp_list,  # we will drop this result
                    thr=curr_th,
                    return_softmax_lse=return_softmax_lse,
                    reverse_skip_list=self.reverse_skip_list,
                    phase=(self._phase == 1) if self.reverse_skip_list else False,
                    use_int8=self.use_int8,
                )
                # and we must flip back
                self._phase = 1 - self._phase
                # calc error
                curr_error = self.calc_error(output_new_th, output_old_th)[cfg.metric]
                return curr_error

            def find_threshold(low, high):
                curr_error = calibration_step(high)
                error_diff = curr_error - cfg.target_error
                if error_diff <= 0:
                    log.warning(
                        "can't find a threshold with the requested target error. using the high limit (below noise target)",
                        threshold=high,
                        error=curr_error,
                        target=cfg.target_error,
                    )
                    return high

                curr_error = calibration_step(low)
                error_diff = curr_error - cfg.target_error
                if error_diff >= 0:
                    log.warning(
                        "can't find a threshold with the requested target error. using the low limit (above noise target)",
                        threshold=low,
                        error=curr_error,
                        target=cfg.target_error,
                    )
                    return low

                # binary search between high (error > target) and low (error <= target)
                for _ in range(30):
                    curr_th = (low + high) / 2
                    curr_error = calibration_step(curr_th)
                    error_diff = curr_error - cfg.target_error
                    if abs(error_diff / cfg.target_error) < 0.1:
                        return curr_th
                    elif error_diff > 0:
                        high = curr_th
                    else:
                        low = curr_th
                log.warning(
                    "binary search did not converge, using midpoint",
                    threshold=curr_th,
                    error=curr_error,
                    target=cfg.target_error,
                )
                return curr_th

            threshold = find_threshold(low=-20.0, high=0.0)
        elif isinstance(cfg, LiteAttentionRunConfig):
            threshold = cfg.threshold
        else:
            raise ValueError(f"Unknown config type: {type(cfg)}")

        output = flash_attn_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=softmax_scale,
            attn_read_list=read_list,
            attn_must_do_list=must_do_list_expanded,
            attn_write_list=write_list,
            thr=threshold,
            return_softmax_lse=return_softmax_lse,
            reverse_skip_list=self.reverse_skip_list,
            # self._phase == 1 because we changed it in _get_read_write_lists!
            phase=(self._phase == 1) if self.reverse_skip_list else False,
            use_int8=self.use_int8,
        )

        # Record calibration results and advance timestep
        if self.enable_skipping:
            self.add_calibration_results(
                LiteAttentionDisabledConfig()
                if disabled
                else LiteAttentionRunConfig(threshold=threshold)
            )

        # Capture debug data if enabled (after add_calibration_results so timestep is set)
        if self.enable_skipping:
            self._maybe_capture(write_list, threshold, query, key, scale)

        # Old way to calculate and store statistics (if enabled)
        if (
            self.enable_skipping
            and not disabled
            and os.getenv("LITE_ATTENTION_VERBOSE", "FALSE") != "FALSE"
        ):
            real_batch_size = query.shape[0]
            self._last_percentage = self.calc_percentage(read_list[:real_batch_size])
            log.info(
                "LiteAttention forward pass statistics",
                skip_percentage=1.0 - self._last_percentage,
                threshold=threshold,
            )

        return output

    def reset_skip_state(self):
        """
        Reset the internal skip list state to force reinitialization.

        This method clears all cached state, forcing the skip list to be reinitialized
        on the next forward pass. Useful when:
        - Manually changing sequence lengths between forward passes
        - Switching to a different model/configuration
        - Debugging skip list behavior
        - Starting a new sequence (e.g., in autoregressive generation)

        After calling this method, the next forward pass will:
        1. Allocate new skip list buffers
        2. Initialize with "compute all tiles" configuration
        3. Reset phase to 0

        Note:
        ----
        In most cases, you don't need to call this manually. The skip list will
        automatically reinitialize when tensor properties change (seq_len, dtype, etc.)
        """
        self._skip_list = None
        self._phase = 0
        self._last_seq_len = None
        self._last_head_dim = None
        self._last_v_colmajor = None
        self._last_dtype = None
        self._last_device = None
        self.verbose_reinitialization = False
        self._last_percentage = 0.0
        self._last_num_heads = None
        self._replay_step_counter = 0
        if self._captured_pct or self._captured_maps:
            warnings.warn(
                "reset_skip_state() called with unsaved capture data; data will be lost.",
                stacklevel=2,
            )
            self._captured_pct = []
            self._captured_maps = []
        self.restart_config()

    @property
    def threshold(self):
        if isinstance(self.config, LiteAttentionRunConfig):
            return self.config.threshold
        else:
            raise RuntimeError("Can't access threshold for a calibreation config")

    @threshold.setter
    def threshold(self, value):
        return self.set_threshold(value)

    def set_threshold(self, threshold: float):
        """
        Update the threshold value for skip list optimization.

        The threshold determines how aggressively tiles are skipped. Tiles with
        max(log-attention-score) below this threshold will be skipped.

        Args:
            threshold (float): Threshold value in log-space. Must be negative
                unless LITE_ATTENTION_DEBUG environment variable is set.
                Lower values = more aggressive skipping = faster but less accurate.
                Typical values: -5.0 to -15.0

        Raises:
            ValueError: If threshold >= 0 and not in debug mode

        Examples:
        --------
        >>> lite_attn = LiteAttention(threshold=-10.0)
        >>> lite_attn.set_threshold(-5.0)  # More aggressive skipping
        >>> lite_attn.set_threshold(-15.0)  # Less aggressive skipping

        Note:
        ----
        Changing the threshold does not reset the skip list state. The new threshold
        will be applied starting from the next forward pass.
        """
        warnings.warn(
            "usage of `LiteAttention.threshold = value` and `LiteAttention.set_threshold` is deprecated. Please use a module registry"
        )
        if threshold >= 0 and os.getenv("LITE_ATTENTION_DEBUG", "FALSE") == "FALSE":
            raise ValueError(
                "threshold must be negative when debug mode is not enabled"
            )

        self._instance_config = LiteAttentionRunConfig(threshold=threshold)

    def enable_skip_optimization(self, enable: bool = True):
        """
        Enable or disable skip list optimization.

        When disabled, the attention computation falls back to standard Flash Attention 3
        without any tile skipping. This is useful for:
        - Comparing performance with/without optimization
        - Debugging accuracy issues
        - Specific layers that need full attention

        Args:
            enable (bool, optional): Whether to enable skip optimization. Defaults to True.

        Note:
        ----
        The skip list state is preserved when toggling this flag, so you can
        switch between optimized and non-optimized modes without reinitializing.

        Example:
        -------
        >>> lite_attn = LiteAttention(enable_skipping=True)
        >>> output1 = lite_attn(q, k, v)  # With skipping
        >>> lite_attn.enable_skip_optimization(False)
        >>> output2 = lite_attn(q, k, v)  # Without skipping
        """
        self.enable_skipping = enable
        # Note: Skip state is preserved to allow toggling without reinitialization

    def _enable_capture(
        self,
        skip_lists: bool = False,
        qk_block_map: bool = False,
        attn_map: bool = False,
        stats: bool = False,
        attn_map_timesteps: Optional[set] = None,
        attn_map_res: int = 256,
        heads: Optional[list] = None,
        batches: Optional[list] = None,
    ):
        """Enable debug capture on this module.

        Always: capture pct_per_head for all heads/timesteps/batches.
        Optional: skip_lists — write_list snapshots (cheap, no QK recomputation).
        Optional: capture detailed attn maps + skip_lists for selected subset.
        Optional: qk_block_map — row-max-normalized pre-softmax QK scores maxpooled to tile granularity (≤ 0, comparable to threshold).
        Optional: running stats (mean/std/max/min) at full resolution across all passes.
        """
        self._capture_enabled = True
        self._captured_pct = []

        self._capture_map_heads = heads
        self._capture_map_batches = batches
        self._captured_maps = []

        self._capture_skip_lists = skip_lists
        self._capture_qk_block_map = qk_block_map

        self._capture_attn_map = attn_map
        self._capture_attn_map_timesteps = attn_map_timesteps
        self._capture_attn_map_res = attn_map_res

        self._capture_stats = stats
        self._stats_count = 0
        self._stats_mean = None
        self._stats_m2 = None
        self._stats_max = None
        self._stats_min = None

    def _disable_capture(self):
        """Disable debug capture and clear accumulated data."""
        self._capture_enabled = False
        self._captured_pct = []

        self._capture_skip_lists = False
        self._capture_qk_block_map = False

        self._capture_attn_map = False
        self._capture_attn_map_timesteps = None
        self._capture_map_heads = None
        self._capture_map_batches = None
        self._capture_attn_map_res = 256
        self._captured_maps = []

        self._capture_stats = False
        self._stats_count = 0
        self._stats_mean = None
        self._stats_m2 = None
        self._stats_max = None
        self._stats_min = None

    @torch.no_grad()
    def _maybe_capture(self, write_list, threshold, query, key, scale):
        """Capture debug data for the current forward pass if capture is enabled.

        Always (when capture enabled): compute pct_per_head for ALL heads and
        ALL batch items. Cheap — no attention map materialization.
        For disabled timesteps (write_list is None), emits pct=1.0.

        Detailed maps (when enabled and timestep/head matches): compute
        downsampled attention maps and save skip_lists for selected subset.
        For disabled timesteps, attn maps are still computed from Q@K^T and
        a synthetic "all tiles computed" skip list is stored.

        Stats (when enabled): accumulate running mean/std/max/min of attention
        maps at full resolution via Welford's online algorithm across ALL
        forward passes. Independent of detailed map capture.

        Args:
            write_list: The write skip list [batch, heads, qtiles, ktiles+2],
                or None for disabled timesteps.
            threshold: The threshold used for this forward pass.
            query: Query tensor [batch, seq_len_q, heads, head_dim].
            key: Key tensor [batch, seq_len_k, heads, head_dim].
            scale: Softmax scale factor (before int8 adjustment).
        """
        if not self._capture_enabled:
            return

        current_timestep = self._config_index - 1
        batch_size, seq_len, num_heads, head_dim = query.shape
        lite_attention_disabled = write_list is None
        scale = scale or 1.0 / math.sqrt(head_dim)

        # --- pct for ALL heads, ALL batch items ---
        if lite_attention_disabled:
            pct_per_head = torch.ones(batch_size, num_heads)
        else:
            pct = self.calc_percentage_per_head(write_list[:batch_size])
            # pct shape: [batch_size, num_heads, qtiles] -> mean over qtiles
            pct_per_head = pct.mean(dim=-1).float().cpu()

        self._captured_pct.append(
            {
                "timestep": current_timestep,
                "pct_per_head": pct_per_head,  # [real_batch, all_heads]
                "threshold": float(threshold),
            }
        )

        # --- Determine what needs attention computation ---
        capture_stats = self._capture_stats
        capture_qk_block_map = self._capture_qk_block_map

        capture_batch_idxs = (
            [bi for bi in self._capture_map_batches if bi < batch_size]
            if self._capture_map_batches is not None
            else range(batch_size)
        )
        capture_head_idxs = (
            [hi for hi in self._capture_map_heads if hi < num_heads]
            if self._capture_map_heads is not None
            else range(num_heads)
        )

        capture_attn_map = self._capture_attn_map
        capture_attn_map &= (
            self._capture_attn_map_timesteps is None
            or current_timestep in self._capture_attn_map_timesteps
        )

        capture_skip_lists = (
            capture_qk_block_map or capture_attn_map or self._capture_skip_lists
        )

        if not (
            capture_attn_map
            or capture_stats
            or capture_qk_block_map
            or capture_skip_lists
        ):
            return

        # --- Block size for qk_block_map ---
        dtype = torch.int8 if self.use_int8 else query.dtype
        v_colmajor = (
            self._last_v_colmajor if self._last_v_colmajor is not None else False
        )
        kBlockM, kBlockN = self.get_MN(head_dim, dtype, v_colmajor)
        qtiles = self.ceil_div(query.shape[1], kBlockM)
        ktiles = self.ceil_div(key.shape[1], kBlockN)

        # --- Skip list capture (actual batch_size, not max_batch_size) ---
        # Only save valid batch slots so replay expansion (broadcast) works
        # correctly when replaying with a larger batch.
        if capture_skip_lists:
            if lite_attention_disabled:
                captured_skip = torch.zeros(
                    batch_size,
                    num_heads,
                    qtiles,
                    ktiles + 2,
                    dtype=torch.int16,
                )
                captured_skip[:, :, :, 0] = 2
                captured_skip[:, :, :, 1] = ktiles - 1
                captured_skip[:, :, :, 2] = -1
            else:
                captured_skip = (
                    write_list[:batch_size].clone().to(dtype=torch.int16, device="cpu")
                )

        # --- Compute attention maps per head ---
        attn_maps = []
        qk_block_maps = []
        for bi_idx, bi in enumerate(capture_batch_idxs):
            head_qk_block_maps = []
            head_attn_maps = []
            for hi_idx, hi in enumerate(capture_head_idxs):
                q_h = query[bi, :, hi, :].unsqueeze(0).unsqueeze(0)
                k_h = key[bi, :, hi, :].unsqueeze(0).unsqueeze(0)
                qk = (q_h.float() @ k_h.float().transpose(-2, -1)) * scale
                # Block-level max of row-max-normalized pre-softmax QK scores.
                # Subtracting the row max first so values represent how far
                # each tile's best score is below the row peak — directly
                # comparable to the kernel's skip threshold.
                if capture_qk_block_map:
                    row_max = qk.max(dim=-1, keepdim=True).values
                    qk_normalized = qk - row_max
                    pad_q = (kBlockM - qk_normalized.shape[-2] % kBlockM) % kBlockM
                    pad_k = (kBlockN - qk_normalized.shape[-1] % kBlockN) % kBlockN
                    qk_padded = F.pad(
                        qk_normalized, (0, pad_k, 0, pad_q), value=float("-inf")
                    )
                    qk_down = F.max_pool2d(
                        qk_padded,
                        kernel_size=(kBlockM, kBlockN),
                        stride=(kBlockM, kBlockN),
                    )
                    head_qk_block_maps.append(qk_down[0, 0].half().cpu())

                attn = (
                    torch.softmax(qk, dim=-1)
                    if capture_attn_map or capture_stats
                    else None
                )

                del qk

                # Detailed map pooling
                if capture_attn_map:
                    res = self._capture_attn_map_res
                    attn_down = F.adaptive_max_pool2d(
                        attn,
                        output_size=(
                            min(res, attn.shape[-2]),
                            min(res, attn.shape[-1]),
                        ),
                    )
                    head_attn_maps.append(attn_down[0, 0].half().cpu())

                # Stats accumulation at full resolution
                if capture_stats:
                    val = attn[0, 0].float().cpu()

                    # Lazy init on first call
                    if self._stats_mean is None:
                        n_batches = len(capture_batch_idxs)
                        n_heads = len(capture_head_idxs)
                        h, w = val.shape
                        self._stats_mean = torch.zeros(
                            n_batches, n_heads, h, w, dtype=torch.float32
                        )
                        self._stats_m2 = torch.zeros(
                            n_batches, n_heads, h, w, dtype=torch.float32
                        )
                        self._stats_max = torch.full(
                            (n_batches, n_heads, h, w),
                            float("-inf"),
                            dtype=torch.float32,
                        )
                        self._stats_min = torch.full(
                            (n_batches, n_heads, h, w),
                            float("inf"),
                            dtype=torch.float32,
                        )

                    # Welford's online algorithm
                    if bi_idx == 0 and hi_idx == 0:
                        self._stats_count += 1
                    count = self._stats_count
                    mean_bh = self._stats_mean[bi_idx, hi_idx]
                    delta = val - mean_bh
                    mean_bh.add_(delta / count)
                    delta2 = val - mean_bh
                    self._stats_m2[bi_idx, hi_idx].add_(delta * delta2)
                    torch.maximum(
                        self._stats_max[bi_idx, hi_idx],
                        val,
                        out=self._stats_max[bi_idx, hi_idx],
                    )
                    torch.minimum(
                        self._stats_min[bi_idx, hi_idx],
                        val,
                        out=self._stats_min[bi_idx, hi_idx],
                    )

                del attn

            if head_qk_block_maps:
                qk_block_maps.append(torch.stack(head_qk_block_maps))
            if head_attn_maps:
                attn_maps.append(torch.stack(head_attn_maps))

        captured_map_entry = {}
        if attn_maps:
            captured_map_entry["attn_map"] = torch.stack(attn_maps)
        if qk_block_maps:
            captured_map_entry["qk_block_map"] = torch.stack(qk_block_maps)
        if capture_skip_lists:
            captured_map_entry["timestep"] = current_timestep
            captured_map_entry["skip_list"] = captured_skip
        if captured_map_entry:
            self._captured_maps.append(captured_map_entry)

    def visualize_skips(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        heads_list: torch.Tensor,
        scale: float,
        save_path: str,
        max_res: int = 520,
        name_prefix: str = "",
        do_softmax: bool = True,
        dims: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
    ):
        """
        Visualize which tiles are being computed vs skipped in the attention matrix.

        Creates visualization images showing the attention pattern with overlays indicating
        which tiles are skipped (based on the current skip list). Useful for debugging
        and understanding the skip list behavior.

        Args:
            query (torch.Tensor): Query tensor of shape [batch, seq_len, heads, head_dim]
            key (torch.Tensor): Key tensor of shape [batch, seq_len, heads, head_dim]
            heads_list (torch.Tensor): 1D tensor of head indices to visualize
                Example: torch.tensor([0, 2, 5]) visualizes heads 0, 2, and 5
            scale (float): Attention scale factor (typically 1/sqrt(head_dim))
            save_path (str): Directory to save visualization images
            max_res (int, optional): Resolution of output images. Defaults to 520.
            name_prefix (str, optional): Prefix for saved file names. Defaults to "".
            do_softmax (bool, optional): Whether to apply softmax before visualization.
                Defaults to True.

        Output:
        ------
        Creates a directory structure: {save_path}/batch_{b}/head_{h}/
        Saves PNG files with attention heatmaps overlaid with white rectangles
        indicating computed tiles (non-skipped regions).

        Example:
        -------
        >>> lite_attn = LiteAttention(enable_skipping=True, threshold=-8.0)
        >>> output = lite_attn(q, k, v)  # Run forward pass to populate skip list
        >>> # Visualize heads 0 and 1
        >>> lite_attn.visualize_skips(
        ...     q, k, torch.tensor([0, 1]), scale=0.125, save_path="./vis/"
        ... )

        Note:
        ----
        This method reads from the current skip list, so it should be called after
        at least one forward pass has been executed.
        """
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # os.makedirs(save_path, exist_ok=True)
        # Create subdirectories for each batch and attention head
        batch = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        skip_list = self._skip_list[self._phase]

        # find out if the skip list is reversed or not
        r1, r2 = skip_list[0, 0, 0, 1:3]
        if r1 > r2:
            step = 1
        else:
            step = 0

        for b in range(batch):
            for h in heads_list:
                batch_head_dir = os.path.join(save_path, f"batch_{b}", f"head_{h}")
                os.makedirs(batch_head_dir, exist_ok=True)

        # kBlockM, kBlockN = LiteAttention.get_MN(key.shape[-1], key.dtype)
        kBlockM, kBlockN = LiteAttention.get_MN(
            key.shape[-1], torch.int8 if self.use_int8 else key.dtype
        )

        # Add grid overlay
        height, width = max_res, max_res
        ratio_height = height / seq_len_q
        ratio_width = width / seq_len_k

        grid_height = kBlockM * ratio_height
        grid_width = kBlockN * ratio_width

        # Calculate grid line positions
        y_positions = [
            b * grid_height
            for b in range(int(height / grid_height) + 1)
            if b * grid_height <= height
        ]
        x_positions = [
            b * grid_width
            for b in range(int(width / grid_width) + 1)
            if b * grid_width <= width
        ]

        for b in range(batch):
            for h in heads_list:
                # Calculate QK for this specific head
                q_head = query[
                    b : b + 1, :, h : h + 1, :
                ]  # (1, seq_len_q, 1, head_dim)
                k_head = key[b : b + 1, :, h : h + 1, :]  # (1, seq_len_k, 1, head_dim)

                # Reshape: (1, seq_len, 1, head_dim) -> (1, 1, seq_len, head_dim)
                q_reshaped = q_head.transpose(1, 2)  # (1, 1, seq_len_q, head_dim)
                k_reshaped = k_head.transpose(1, 2)  # (1, 1, seq_len_k, head_dim)

                # Compute attention
                QK = (
                    q_reshaped @ k_reshaped.transpose(-2, -1)
                ) * scale  # (1, 1, seq_len_q, seq_len_k)
                if do_softmax:
                    attn_softmaxed = torch.softmax(QK, dim=-1)
                else:
                    attn_softmaxed = QK

                if dims is not None:
                    prev_shape = attn_softmaxed.shape
                    attn_softmaxed = (
                        attn_softmaxed.view(*dims[0])
                        .permute(*dims[1])
                        .contiguous()
                        .view(prev_shape)
                    )

                attn_down = F.adaptive_max_pool2d(
                    attn_softmaxed,  # (1, 1, seq_len_q, seq_len_k)
                    output_size=(max_res, max_res),
                )  # -> (1, 1, max_res, max_res)

                attn_map = attn_down[0, 0]  # (max_res, max_res)

                current_skip_list = skip_list[b, h][None, None, ...]
                percentage = self.calc_percentage(current_skip_list)

                plt.figure(figsize=(6, 6))
                attn_cpu = attn_map.detach().float().cpu()
                plt.imshow(attn_cpu, cmap="viridis", interpolation="nearest")
                plt.title(
                    f"Batch {b} | Head {h} | Percentage {percentage * 100:.2f}% | Do Softmax: {do_softmax}"
                )

                # Add horizontal grid lines
                for y in y_positions:
                    plt.axhline(y=y - 0.5, color="black", linewidth=0.2, alpha=0.7)

                # Add vertical grid lines
                for x in x_positions:
                    plt.axvline(x=x - 0.5, color="black", linewidth=0.2, alpha=0.7)

                if dims is None:
                    # plot the skip list
                    for i, row_skip_list in enumerate(current_skip_list[0, 0]):
                        # print(row_skip_list.shape)
                        l_row = row_skip_list[0]
                        # end0, start1, end1, start2, ...
                        # width_ranges = (row_skip_list[1 : l_row + 1] + 1) * grid_width
                        width_ranges = (
                            row_skip_list[1 : l_row + 1] + step
                        ) * grid_width
                        # height
                        row_height = i * grid_height

                        width_ranges = width_ranges.view(-1, 2).cpu()
                        # for end, start in width_ranges:
                        for r1, r2 in width_ranges:
                            start = min(r1, r2)
                            end = max(r1, r2)
                            # rect = plt.Rectangle((start, row_height), end + grid_width - start, grid_height, facecolor='white', edgecolor='none', linewidth=0.4, alpha=0.3)
                            rect = plt.Rectangle(
                                (start, row_height),
                                end - start,
                                grid_height,
                                facecolor="white",
                                edgecolor="none",
                                linewidth=0.4,
                                alpha=0.3,
                            )
                            plt.gca().add_patch(rect)

                plt.axis("off")
                plt.tight_layout()

                batch_head_dir = os.path.join(save_path, f"batch_{b}", f"head_{h}")
                filename = f"{name_prefix}.png" if name_prefix else "visualization.png"
                file_path = os.path.join(batch_head_dir, filename)
                plt.savefig(file_path, dpi=150)
                plt.close()

    @property
    def read_list(self) -> Optional[torch.Tensor]:
        """
        Get the current read skip list (what was computed in the last forward pass).

        The read list contains the tile ranges that were computed in the most recent
        forward pass. This can be used to analyze which tiles were skipped.

        Returns:
            Optional[torch.Tensor]: Skip list tensor of shape [batch, heads, qtiles, ktiles+2]
                Returns None if skip list hasn't been initialized yet.

        Note:
        ----
        Only includes data for the actual batch size used (not max_batch_size).
        The skip list format depends on the reverse_skip_list flag.
        """
        if self._skip_list is None:
            return None
        return self._skip_list[self._phase, : self._last_batch_size]

    @property
    def write_list(self) -> Optional[torch.Tensor]:
        """
        Get the current write skip list (where results will be written this pass).

        The write list is being populated during the current forward pass and will
        become the read list for the next forward pass.

        Returns:
            Optional[torch.Tensor]: Skip list tensor of shape [batch, heads, qtiles, ktiles+2]
                Returns None if skip list hasn't been initialized yet.

        Note:
        ----
        Only includes data for the actual batch size used (not max_batch_size).
        The skip list format depends on the reverse_skip_list flag.
        """
        if self._skip_list is None:
            return None
        return self._skip_list[1 - self._phase, : self._last_batch_size]


class SeqParallelLiteAttention:
    """
    Sequence-parallel version of LiteAttention for distributed attention computation.

    This class manages multiple LiteAttention instances, one for each node in a
    sequence-parallel setup. Each node processes a different portion of the sequence,
    and this class handles routing to the appropriate instance.

    Args:
        num_nodes (int): Number of nodes in the sequence-parallel setup
        enable_skipping (bool, optional): Whether to enable skip list optimizations.
            Defaults to True.
        threshold (float, optional): Log-space threshold for skipping tiles.
            Defaults to -10.0.
        max_batch_size (int, optional): Maximum batch size. Defaults to 2.
        config (LiteAttentionRunConfig | LiteAttentionCalibConfig, optional): Configuration
            for threshold or calibration. Applied to all LiteAttention instances.

    Attributes:
        num_nodes (int): Number of nodes
        lite_attention (list[LiteAttention]): List of LiteAttention instances,
            one per node

    Example:
    -------
    >>> # Setup for 4-way sequence parallelism
    >>> seq_parallel_attn = SeqParallelLiteAttention(num_nodes=4)
    >>> # Node 0 processes its portion
    >>> output_0 = seq_parallel_attn(q_0, k_0, v_0, split_idx=0)
    >>> # Node 1 processes its portion
    >>> output_1 = seq_parallel_attn(q_1, k_1, v_1, split_idx=1)
    """

    def __init__(
        self,
        num_nodes: int,
        enable_skipping: bool = True,
        threshold: float | None = None,
        max_batch_size: int = 2,
        use_int8: bool = False,
        config: LiteAttentionRunConfig | LiteAttentionCalibConfig | None = None,
    ):
        self.num_nodes = num_nodes
        # Create separate LiteAttention instance for each node
        self.lite_attention = [
            LiteAttention(
                enable_skipping=enable_skipping,
                threshold=threshold,
                max_batch_size=max_batch_size,
                use_int8=use_int8,
            )
            for _ in range(num_nodes)
        ]

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        split_idx: int,
        scale: Optional[float] = None,
        return_softmax_lse: bool = False,
        must_do_list: list = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform attention computation for a specific node in the sequence-parallel setup.

        Routes the computation to the appropriate LiteAttention instance based on
        the split_idx parameter.

        Args:
            query (torch.Tensor): Query tensor [batch, seq_len, heads, head_dim]
            key (torch.Tensor): Key tensor [batch, seq_len, heads, head_dim]
            value (torch.Tensor): Value tensor [batch, seq_len, heads, head_dim]
            split_idx (int): Index of the node to use (0 to num_nodes-1)
            scale (float, optional): Attention scale factor. Defaults to None.
            return_softmax_lse (bool, optional): Whether to return log-sum-exp.
                Defaults to False.
            must_do_list (list, optional): Sequence ranges to always compute (automatically
                converted to tile indices). Format: [seq_start_0, seq_end_0, ...] where end
                indices are EXCLUSIVE. Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Attention output

        Raises:
            AssertionError: If split_idx >= num_nodes
        """
        assert split_idx < self.num_nodes, "split_idx must be less than num_nodes"
        lite_attention = self.lite_attention[split_idx]
        return lite_attention(
            query, key, value, scale, return_softmax_lse, must_do_list
        )

    def reset_skip_state(self):
        """
        Reset skip list state for all nodes.

        Calls reset_skip_state() on each LiteAttention instance.
        """
        for lite_attention in self.lite_attention:
            lite_attention.reset_skip_state()

    def set_threshold(self, threshold: float):
        """
        Set threshold for all nodes.

        Args:
            threshold (float): Threshold value to apply to all nodes
        """
        for lite_attention in self.lite_attention:
            lite_attention.set_threshold(threshold)

    def enable_skip_optimization(self, enable: bool = True):
        """
        Enable or disable skip optimization for all nodes.

        Args:
            enable (bool, optional): Whether to enable optimization. Defaults to True.
        """
        for lite_attention in self.lite_attention:
            lite_attention.enable_skip_optimization(enable)


class LiteAttentionRegistry(ModuleRegistry):
    """
    LiteAttention-specific registry with convenience classmethods for
    creating configured registries from a model.
    """

    @classmethod
    def from_model(
        cls,
        model,
        mode: str | None = None,
        threshold: float | None = None,
        filename: str | Path | None = None,
        calib_config: dict | None = None,
        force: bool = False,
        disabled_steps: int = 0,
    ) -> Self:
        """
        Create a registry from a model and configure all its LiteAttention modules.

        Args:
            model: `nn.Module` that contains LiteAttention modules.
            mode: Configuration mode - 'const', 'load', 'calib', 'replay',
                or 'disable'.
            threshold: Threshold value for mode='const', or for mode='replay'
                to compute skip lists from qk_block_map (None = use captured
                skip lists directly).
            filename: Path to config file for mode='load' (input),
                mode='calib' (output via save_if_calib), or mode='replay'
                (.pt capture file or .toml config). Cast to Path internally.
            calib_config: Dict of calibration params for mode='calib',
                passed as kwargs to LiteAttentionCalibConfig
                (e.g. {"target_error": 0.001, "metric": "L1"}).
            force: If True, override instance-level configs on modules.
                If False (default), warn when a module has an instance config
                that will take precedence over the registry config.

        """
        if filename is not None:
            filename = Path(filename)
        if mode is None:
            warnings.warn(
                "No 'mode' supplied for the registry. Using a 'const' mode",
                stacklevel=2,
            )
            mode = "const"

        registry = cls(model.named_modules())
        registry._mode = mode
        registry._filename = filename

        for name, module in registry.named_modules.items():
            if module._instance_config is not None:
                if force or mode == "calib":
                    module._instance_config = None
                else:
                    log.warning(
                        "Module has instance config that will override registry config. "
                        "Use force=True to override.",
                        module_name=name,
                    )

        if mode == "disable":
            registry.set_bulk_config(LiteAttentionDisabledConfig())
        elif mode == "const":
            if threshold is None:
                warnings.warn(
                    "no 'threshold' specified for mode 'const'. Using default value",
                    stacklevel=2,
                )
                cfg = LiteAttentionRunConfig.default()
            else:
                cfg = LiteAttentionRunConfig(threshold=threshold)
            registry.set_bulk_config(cfg)
        elif mode == "load":
            if filename is None:
                raise ValueError("filename is required for mode='load'")
            registry.load_config(
                filename,
                config_types=[
                    LiteAttentionRunConfig,
                    LiteAttentionDisabledConfig,
                    LiteAttentionCalibConfig,
                    LiteAttentionReplayConfig,
                ],
            )
        elif mode == "replay":
            if filename is None:
                raise ValueError("filename is required for mode='replay'")
            if filename.suffix == ".toml":
                registry.load_config(
                    filename,
                    config_types=[
                        LiteAttentionReplayConfig,
                        LiteAttentionRunConfig,
                        LiteAttentionDisabledConfig,
                    ],
                )
            else:
                # .pt capture file — replay all modules from it
                registry.set_bulk_config(
                    LiteAttentionReplayConfig(
                        skip_list_file=str(filename), threshold=threshold
                    )
                )
        elif mode == "calib":
            if filename is None:
                raise ValueError("filename is required for mode='calib'")
            if calib_config is None:
                warnings.warn(
                    "no 'calib_config' specified for mode='calib'. Using default values",
                    stacklevel=2,
                )
                calib_config = {}
            registry.set_bulk_config(LiteAttentionCalibConfig(**calib_config))
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Must be 'const', 'load', 'calib', 'replay', or 'disable'."
            )

        if disabled_steps > 0 and mode != "disable":
            disabled_prefix = [LiteAttentionDisabledConfig()] * disabled_steps
            for module in registry.named_modules.values():
                cfg = module._registry_config
                if isinstance(cfg, ConfigList):
                    remainder = list(cfg)[disabled_steps:]
                    if not remainder:
                        remainder = [cfg[-1]]
                    module._registry_config = ConfigList(disabled_prefix + remainder)
                else:
                    module._registry_config = ConfigList(disabled_prefix + [cfg])

        # Hydrate replay data: load .pt files, convert write→read lists, attach
        if mode == "replay":
            registry._hydrate_replay_data()

        return registry

    def _hydrate_replay_data(self) -> None:
        """Load .pt capture files and attach replay skip lists to modules.

        For each module with a ``LiteAttentionReplayConfig``, this method:

        1. Loads the referenced .pt capture file (caching across modules).
        2. Validates that all heads were captured (replay requires full head coverage).
        3. Produces read-lists from the capture data:

           - **threshold=None** (skip-list replay): shifts captured write-lists
             by one timestep.
           - **threshold is set** (QK-map replay): computes skip lists from
             ``qk_block_map`` and the given threshold.

           In both cases ``read_list[0]`` = "compute all" initial buffer.
        4. Stores the result as ``module._replay_skip_lists``.
        """
        pt_cache: dict[str, dict] = {}

        for name, module in self.named_modules.items():
            # Find replay configs for this module
            cfg = module._registry_config
            replay_cfg: LiteAttentionReplayConfig | None = None
            if isinstance(cfg, LiteAttentionReplayConfig):
                replay_cfg = cfg
            elif isinstance(cfg, ConfigList):
                for c in cfg:
                    if isinstance(c, LiteAttentionReplayConfig):
                        replay_cfg = c
                        break

            if replay_cfg is None:
                module._replay_skip_lists = None
                continue

            skip_list_file = replay_cfg.skip_list_file
            if not skip_list_file:
                raise ValueError(
                    f"Module '{name}': LiteAttentionReplayConfig has empty skip_list_file"
                )

            # Load .pt file (with caching)
            if skip_list_file not in pt_cache:
                pt_cache[skip_list_file] = torch.load(
                    skip_list_file, map_location="cpu", weights_only=False
                )
            capture_data = pt_cache[skip_list_file]

            if name not in capture_data["modules"]:
                raise ValueError(
                    f"Module '{name}' not found in capture file '{skip_list_file}'"
                )
            mod_data = capture_data["modules"][name]

            # Validate all heads were captured
            total_heads = mod_data["pct_per_head"].shape[-1]
            map_heads = mod_data.get("map_heads", list(range(total_heads)))
            if len(map_heads) != total_heads:
                raise ValueError(
                    f"Module '{name}': replay requires all {total_heads} heads "
                    f"to be captured, but only {len(map_heads)} were "
                    f"(map_heads={map_heads}). Re-capture with heads=None."
                )

            # Validate batch coverage
            total_batches = mod_data["pct_per_head"].shape[1]
            map_batches = mod_data.get("map_batches", list(range(total_batches)))
            if len(map_batches) != total_batches:
                log.warning(
                    "Replay: not all batches were captured; will expand at runtime",
                    module=name,
                    captured_batches=len(map_batches),
                    total_batches=total_batches,
                )

            # Count leading disabled steps in the replay config to determine
            # which captured entries correspond to active (non-disabled) timesteps.
            # The capture file includes synthetic "compute all" entries for
            # disabled timesteps; we skip those.
            n_disabled = 0
            if isinstance(cfg, ConfigList):
                for c in cfg:
                    if isinstance(c, LiteAttentionDisabledConfig):
                        n_disabled += 1
                    else:
                        break

            use_qk_map = replay_cfg.threshold is not None

            if use_qk_map:
                # --- QK-map replay: compute skip lists from qk_block_map ---
                if "qk_block_map" not in mod_data:
                    raise ValueError(
                        f"Module '{name}': capture file has no qk_block_map. "
                        "Re-capture with qk_block_map=True."
                    )
                qk_block_map = mod_data[
                    "qk_block_map"
                ]  # [T, B_sel, H_sel, qtiles, ktiles]
                replay_list = self._qk_map_to_replay_skip_lists(
                    qk_block_map, replay_cfg.threshold, n_disabled
                )
            else:
                # --- Skip-list replay: use captured write-lists directly ---
                if "skip_lists" not in mod_data:
                    raise ValueError(
                        f"Module '{name}': capture file has no skip_lists. "
                        "Re-capture with qk_block_map=True or attn_map=True."
                    )
                skip_lists = mod_data[
                    "skip_lists"
                ]  # [T, B_sel, H_sel, qtiles, ktiles+2]
                B_sel, H_sel, qtiles, ktiles_plus_2 = skip_lists.shape[1:]
                ktiles = ktiles_plus_2 - 2
                compute_all = self._make_compute_all(B_sel, H_sel, qtiles, ktiles)
                active_write_lists = [
                    skip_lists[t] for t in range(n_disabled, skip_lists.shape[0])
                ]
                replay_list = [compute_all] + active_write_lists

            module._replay_skip_lists = replay_list
            module._replay_step_counter = 0

    @staticmethod
    def _make_compute_all(B: int, H: int, qtiles: int, ktiles: int) -> torch.Tensor:
        """Build a "compute all tiles" skip list buffer."""
        buf = torch.zeros(B, H, qtiles, ktiles + 2, dtype=torch.int16)
        buf[:, :, :, 0] = 2
        buf[:, :, :, 1] = ktiles - 1
        buf[:, :, :, 2] = -1
        return buf

    @staticmethod
    def _qk_map_to_replay_skip_lists(
        qk_block_map: torch.Tensor,
        threshold: float,
        n_disabled: int = 0,
    ) -> list[torch.Tensor]:
        """Convert ``qk_block_map`` + threshold to replay skip lists.

        For each active timestep, tiles whose block-level QK score (row-max-
        normalized, ≤ 0) is **≥ threshold** are marked as "compute".
        Contiguous ranges of compute-tiles are encoded in the skip list format
        expected by the CUDA kernel (reversed range pairs, phase-aware).

        Args:
            qk_block_map: Shape ``[T, B, H, qtiles, ktiles]``, float16/32,
                values ≤ 0.
            threshold: Tiles with ``qk_block_map >= threshold`` are computed.
            n_disabled: Number of leading disabled timesteps in the capture
                (their entries are skipped).

        Returns:
            List of ``[B, H, qtiles, ktiles+2]`` int16 tensors, starting with
            a "compute all" initial buffer at index 0.
        """
        T_total, B, H, qtiles, ktiles = qk_block_map.shape
        compute_all = LiteAttentionRegistry._make_compute_all(B, H, qtiles, ktiles)
        result: list[torch.Tensor] = [compute_all]

        # Boolean mask: which tiles to compute
        compute_mask = qk_block_map >= threshold  # [T, B, H, qtiles, ktiles]

        for t in range(n_disabled, T_total):
            # replay_idx = position in result list (1-based)
            replay_idx = t - n_disabled + 1
            phase_true = replay_idx % 2 == 0

            skip_list = torch.zeros(B, H, qtiles, ktiles + 2, dtype=torch.int16)

            for b in range(B):
                for h in range(H):
                    for q in range(qtiles):
                        row = compute_mask[t, b, h, q]  # [ktiles] bool

                        # Find contiguous True ranges
                        ranges: list[tuple[int, int]] = []
                        i = 0
                        while i < ktiles:
                            if row[i]:
                                start = i
                                while i < ktiles and row[i]:
                                    i += 1
                                ranges.append((start, i - 1))  # inclusive end
                            else:
                                i += 1

                        if not ranges:
                            # length=0 → skip everything
                            continue

                        skip_list[b, h, q, 0] = len(ranges) * 2
                        idx = 1
                        # Reversed order: last range first
                        for s, e in reversed(ranges):
                            if phase_true:
                                # phase=True: start_x < end_x
                                # kernel iterates range(start_x+1, end_x+1)
                                skip_list[b, h, q, idx] = e  # end_x
                                skip_list[b, h, q, idx + 1] = s - 1  # start_x
                            else:
                                # phase=False: start_x > end_x
                                # kernel iterates range(start_x-1, end_x-1, -1)
                                skip_list[b, h, q, idx] = s  # end_x
                                skip_list[b, h, q, idx + 1] = e + 1  # start_x
                            idx += 2

            result.append(skip_list)

        return result

    def enable_capture(
        self,
        save_path: str | Path,
        skip_lists: bool = False,
        qk_block_map: bool = False,
        attn_map: bool = False,
        stats: bool = False,
        attn_map_modules: Union[list, typing.Callable, None] = None,
        attn_map_timesteps: Optional[list] = None,
        attn_map_res: int = 256,
        heads: Optional[list] = None,
        batches: Optional[list] = None,
    ) -> None:
        """Enable debug capture on all modules.

        All modules always capture pct_per_head for every timestep and head.
        skip_lists captures write_list snapshots cheaply (no QK recomputation).
        Attn maps + skip_lists are also captured for the filtered subset when attn_map or qk_block_map is enabled.
        qk_block_map captures row-max-normalized pre-softmax QK scores maxpooled to tile granularity (≤ 0, directly comparable to threshold).
        Stats (mean/std/max/min) are accumulated at full resolution on the same
        subset of modules, across ALL forward passes.

        Args:
            save_path: Path for the .pt capture file (written by save()).
            skip_lists: Enable cheap skip list capture (write_list snapshots, no QK recomputation).
            attn_map: Enable detailed attention map capture.
            attn_map_modules: Which modules capture attn maps and/or stats. Can be:
                - list[str]: exact module names
                - Callable[[str], bool]: predicate on module name
                - None: all modules
            attn_map_timesteps: Timestep indices for map capture, or None for all.
            heads: Head indices for map/stats capture, or None for all.
            batches: Batch indices for map/stats capture, or None for all.
            attn_map_res: Resolution for downsampled attention maps.
            qk_block_map: Enable row-max-normalized block-level QK scores (≤ 0, comparable to threshold).
            stats: Enable running stats accumulation at full resolution.
        """
        self._capture_path = Path(save_path)

        attn_map_timesteps = (
            set(attn_map_timesteps) if attn_map_timesteps is not None else None
        )

        for name, module in self.named_modules.items():
            # Determine if this module is in the selected subset
            if attn_map_modules is None:
                selected = True
            elif callable(attn_map_modules):
                selected = attn_map_modules(name)
            else:
                selected = name in attn_map_modules

            module._enable_capture(
                skip_lists=skip_lists,
                qk_block_map=qk_block_map and selected,
                attn_map=attn_map and selected,
                stats=stats and selected,
                attn_map_timesteps=attn_map_timesteps,
                attn_map_res=attn_map_res,
                heads=heads,
                batches=batches,
            )

    def save(self) -> None:
        """Unified save: calibration config and/or debug capture data.

        - If in calib mode: saves calibration config to self._filename.
        - If capture is enabled: collects captured data from all modules,
          stacks into the capture file format, and writes to self._capture_path.
          Capture remains enabled and data is preserved (not cleared).
        """
        # Save calibration if applicable
        if getattr(self, "_mode", None) == "calib":
            if self._filename is None:
                raise ValueError(
                    "Cannot save calibration results: no filename specified"
                )
            self.config_output.save(self._filename)

        # Save capture data if applicable
        capture_path = getattr(self, "_capture_path", None)
        if capture_path is not None:
            self._save_capture(capture_path)

    def _save_capture(self, path: Path) -> None:
        """Collect captured data from all modules and write to a .pt file."""
        file_data = {"modules": {}}
        has_data = False

        for name, module in self.named_modules.items():
            if not module._captured_pct:
                continue
            if module._last_dtype is None:
                continue
            has_data = True

            kBlockM, kBlockN = LiteAttention.get_MN(
                module._last_head_dim,
                torch.int8 if module.use_int8 else module._last_dtype,
            )

            # --- pct for all timesteps/heads ---
            timesteps = [d["timestep"] for d in module._captured_pct]
            thresholds = [d["threshold"] for d in module._captured_pct]
            pct_per_head = torch.stack(
                [d["pct_per_head"] for d in module._captured_pct]
            )  # [T_all, B, H_all]

            mod_data = {
                "timesteps": timesteps,
                "thresholds": thresholds,
                "pct_per_head": pct_per_head,
                "seq_len_q": module._last_seq_len[0],
                "seq_len_k": module._last_seq_len[1],
                "head_dim": module._last_head_dim,
                "use_int8": module.use_int8,
                "kBlockM": kBlockM,
                "kBlockN": kBlockN,
            }

            # --- Detailed attn maps + skip_lists + qk_block_map for filtered subset ---
            if module._captured_maps:
                map_timesteps = [d["timestep"] for d in module._captured_maps]
                mod_data["map_timesteps"] = map_timesteps

                has_maps = (
                    "attn_map" in module._captured_maps[0]
                    or "qk_block_map" in module._captured_maps[0]
                )
                if has_maps:
                    # map_heads/map_batches only apply to attn_map/qk_block_map (filtered)
                    if module._capture_map_heads is not None:
                        mod_data["map_heads"] = module._capture_map_heads
                    else:
                        mod_data["map_heads"] = list(range(pct_per_head.shape[-1]))

                    if module._capture_map_batches is not None:
                        mod_data["map_batches"] = module._capture_map_batches
                    else:
                        mod_data["map_batches"] = list(range(pct_per_head.shape[1]))

                if "skip_list" in module._captured_maps[0]:
                    mod_data["skip_lists"] = torch.stack(
                        [d["skip_list"] for d in module._captured_maps]
                    )
                if "attn_map" in module._captured_maps[0]:
                    mod_data["attn_maps"] = torch.stack(
                        [d["attn_map"] for d in module._captured_maps]
                    )
                    mod_data["attn_map_res"] = module._capture_attn_map_res

                if "qk_block_map" in module._captured_maps[0]:
                    mod_data["qk_block_map"] = torch.stack(
                        [d["qk_block_map"] for d in module._captured_maps]
                    )

            # --- Running stats (independent of detailed maps) ---
            if module._stats_count > 0 and module._stats_mean is not None:
                count = module._stats_count
                mod_data["stats_mean"] = (
                    module._stats_mean.float()
                )  # [B_sel, H_sel, h, w]
                mod_data["stats_std"] = (module._stats_m2 / count).sqrt().float()
                mod_data["stats_max"] = module._stats_max.float()
                mod_data["stats_min"] = module._stats_min.float()
                mod_data["stats_count"] = module._stats_count
                if module._capture_map_batches is not None:
                    mod_data["stats_batch_indices"] = module._capture_map_batches
                else:
                    mod_data["stats_batch_indices"] = list(range(pct_per_head.shape[1]))
                if module._capture_map_heads is not None:
                    mod_data["stats_heads"] = module._capture_map_heads
                else:
                    mod_data["stats_heads"] = list(range(pct_per_head.shape[-1]))

            file_data["modules"][name] = mod_data

        if has_data:
            torch.save(file_data, path)
            from lite_attention.debug_capture import mean_pct

            print(f"Debug capture saved to {path}")
            print(f"  Mean tiles computed: {mean_pct(file_data):.1%}")

    def save_if_calib(self) -> None:
        """Save calibration results to file if in calibration mode.

        .. deprecated:: Use save() instead, which handles both calibration and capture.
        """
        if self._mode != "calib":
            return
        if self._filename is None:
            raise ValueError("Cannot save calibration results: no filename specified")
        self.config_output.save(self._filename)
