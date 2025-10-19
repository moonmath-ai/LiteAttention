"""
LiteAttention: A lightweight Flash Attention 3 wrapper with skip list optimization.

This module provides a clean interface for Flash Attention 3 with internal management
of read and write skip lists, hiding the complexity from users.
"""

import torch
import os
from typing import Optional, Tuple, Union

from ._internal.flash_attn_interface import flash_attn_func


class LiteAttention:
    """
    A lightweight attention class that encapsulates Flash Attention 3 with optimized skip lists.
    This class manages read and write masks internally and provides a clean interface for users.
    
    Args:
        enable_skipping (bool, optional): Whether to enable skip list optimizations. Defaults to False.
        threshold (float, optional): Threshold value for skip list optimization. Defaults to -10.0. If positive, it will be converted to negative.
        max_batch_size (int, optional): Maximum batch size. Defaults to 4.
        
    Example:
        >>> # Basic usage without skip optimization
        >>> lite_attn = LiteAttention()
        >>> output = lite_attn(query, key, value)
        
        >>> # With skip optimization enabled  
        >>> lite_attn = LiteAttention(enable_skipping=True, threshold=5.0)
        >>> output = lite_attn(query, key, value)
        >>> print(f"Skipped {lite_attn.get_skip_percentage():.2%} computations")
    """
    
    def __init__(self, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 4):
        # Internal skip list management
        self._skip_list = None
        self._phase = 0

        self._last_seq_len = None
        self._last_head_dim = None
        self._last_v_colmajor = None
        self._last_dtype = None
        self._last_device = None
        self._last_num_heads = None

        self._last_percentage = 0.0
        
        # Public configuration
        self.enable_skipping = enable_skipping
        self.set_threshold(threshold)

        self.max_batch_size = max_batch_size

    @staticmethod
    def ceil_div(x, y):
        """Ceiling division utility function."""
        return (x + y - 1) // y

    @staticmethod
    def calc_percentage(read_list: torch.Tensor) -> float:
        """Calculate the percentage of non-skipped attention computations."""
        read_list = read_list.to(torch.int64)
        skip_lengths = read_list[:, :, :, 0] // 2
        
        # Calculate range sizes
        read_list_range_sized = read_list[:, :, :, 2:] - read_list[:, :, :, 1:-1]
        if read_list_range_sized.shape[-1] % 2 != 0:
            # Pad with 0 if uneven
            padding_shape = list(read_list_range_sized.shape)
            padding_shape[-1] = 1
            padding = torch.zeros(padding_shape, dtype=read_list_range_sized.dtype, device=read_list_range_sized.device)
            read_list_range_sized = torch.cat([read_list_range_sized, padding], dim=-1)
        
        read_list_range_sized = read_list_range_sized.view(
            read_list_range_sized.shape[0], read_list_range_sized.shape[1], 
            read_list_range_sized.shape[2], -1, 2
        )
        read_list_range_sized = read_list_range_sized[..., 0].cumsum(dim=-1)
        
        total_possible = read_list.shape[0] * read_list.shape[1] * read_list.shape[2] * (read_list.shape[3] - 1)
        total_not_skipped = torch.gather(read_list_range_sized, dim=-1, index=skip_lengths.unsqueeze(-1)).squeeze(-1).sum()
        
        return total_not_skipped / total_possible if total_possible > 0 else 1.0

    @staticmethod
    def get_MN(head_dim, element_size, v_colmajor=False):
        """Get the tile sizes of tiles for the key and value tensors."""
        if element_size == 2:
            if head_dim <= 64:
                return 192, 192
            elif head_dim <= 96:
                return 192, 144
            elif head_dim <= 128:
                return 128, 176
            elif head_dim <= 192:
                return 128, 112
            else:
                return 128, 80
        else:
            if head_dim <= 64:
                return 192, 160
            elif head_dim <= 96:
                return 192, 128
            elif head_dim <= 128:
                return 128, (192 if v_colmajor else 224)
            elif head_dim <= 192:
                return 128, 160
            else:
                return 128, 128

    @staticmethod
    def init_skip_list(batch, seq_len, heads, head_dim, v_colmajor, dtype, device) -> torch.Tensor:
        """Initialize skip list tensors based on query shape."""

        # the number of bytes needed to represent dtype (size(dtype) if it where C code)
        element_size = dtype.itemsize
        kTileM, kTileN = LiteAttention.get_MN(head_dim, element_size, v_colmajor)

        qtiles = LiteAttention.ceil_div(seq_len, kTileM)
        ktiles = LiteAttention.ceil_div(seq_len, kTileN)
        
        skip_list = torch.zeros(2, batch, heads, qtiles, ktiles + 1, dtype=torch.int32, device=device)
        skip_list[:, :, :, :, 2] = ktiles
        skip_list[:, :, :, :, 0] = 2  # First element is the length of skip list
        
        return skip_list

    def _init_skip_list(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Initialize skip list tensors based on query shape."""
        batch, seq_len, heads, head_dim = query.shape
        assert batch <= self.max_batch_size, "batch size must be less than or equal to max_batch_size (modify max_batch_size in LiteAttention constructor)"
        v_colmajor = value.shape[-3] == head_dim
        dtype = query.dtype
        device = query.device
        return LiteAttention.init_skip_list(self.max_batch_size, seq_len, heads, head_dim, v_colmajor, dtype, device)
    
    
    def _get_read_write_lists(self, query: torch.Tensor, value: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the current read and write lists for this attention step."""
        if not self.enable_skipping:
            return None, None
            
        current_seq_len = query.shape[1]
        head_dim = query.shape[-1]
        current_head_dim = head_dim
        current_num_heads = query.shape[2]
        v_colmajor = value.shape[-3] == head_dim
        dtype = query.dtype
        device = query.device
        
        # Initialize or reinitialize skip list if needed
        if (self._skip_list is None or 
            self._last_seq_len != current_seq_len or 
            self._skip_list.device != query.device or
            self._last_head_dim != current_head_dim or
            self._last_v_colmajor != v_colmajor or
            self._last_dtype != dtype or
            self._last_device != device or
            self._last_num_heads != current_num_heads
            ):

            self._skip_list = self._init_skip_list(query, value)
            self._phase = 0

            self._last_seq_len = current_seq_len
            self._last_head_dim = current_head_dim
            self._last_v_colmajor = v_colmajor
            self._last_dtype = dtype
            self._last_device = device
            self._last_num_heads = current_num_heads

            if os.getenv("LITE_ATTENTION_VERBOSE", "FALSE") != "FALSE":
                print(f"[Warning]: reinitialized skip list during the forward pass")
        
        # Alternate between the two skip list buffers
        if self._phase == 0:
            read_list = self._skip_list[0]
            write_list = self._skip_list[1]
            self._phase = 1
        else:
            read_list = self._skip_list[1]
            write_list = self._skip_list[0]
            self._phase = 0
            
        return read_list, write_list
    
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                 scale: Optional[float] = None, return_softmax_lse: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform flash attention 3 with optional skip list optimization.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch, seq_len, heads, head_dim)
            key (torch.Tensor): Key tensor of shape (batch, seq_len, heads, head_dim)  
            value (torch.Tensor): Value tensor of shape (batch, seq_len, heads, head_dim)
            scale (float, optional): Attention scale factor. If None, uses 1/sqrt(head_dim)
            
        Returns:
            torch.Tensor: Attention output of shape (batch, seq_len, heads * head_dim)
        """
        # Get read and write lists (internal mask management)
        read_list, write_list = self._get_read_write_lists(query, value)
        
        # Perform flash attention 3 with skip lists
        output = flash_attn_func(
            q=query,
            k=key, 
            v=value,
            softmax_scale=scale,
            attn_read_list=read_list,
            attn_write_list=write_list,
            thr=self.threshold,
            return_softmax_lse=return_softmax_lse
        )

        # Calculate and store statistics if enabled
        if self.enable_skipping and os.getenv("LITE_ATTENTION_VERBOSE", "FALSE") != "FALSE":
            real_batch_size = query.shape[0]
            self._last_percentage = self.calc_percentage(read_list[:real_batch_size])
            print(f"[Info]: Percentage of tiles skipped: {1.0 - self._last_percentage:.2%}")
        
        return output
    
    def reset_skip_state(self):
        """Reset the internal skip list state. Useful when changing sequence lengths."""
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
    
    def set_threshold(self, threshold: float):
        """Update the threshold value for skip list optimization.
        Threshold must be negative when debug mode is not enabled.
        """
        if threshold >= 0 and os.getenv("LITE_ATTENTION_DEBUG", "FALSE") == "FALSE":
            raise ValueError("threshold must be negative when debug mode is not enabled")

        self.threshold = threshold
    
    def enable_skip_optimization(self, enable: bool = True):
        """Enable or disable skip list optimization."""
        self.enable_skipping = enable
        # TODO @dor: commented out as a reminder to reconsider in the future if resetting the skip state is needed
        # if not enable:
        #     self.reset_skip_state()

class SeqParallelLiteAttention:
    def __init__(self, num_nodes: int, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 4):

        self.num_nodes = num_nodes
        self.lite_attention = [LiteAttention(enable_skipping, threshold, max_batch_size) for _ in range(num_nodes)]
        self.set_threshold(threshold)

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, split_idx: int,
                 scale: Optional[float] = None, return_softmax_lse: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert split_idx < self.num_nodes, "split_idx must be less than num_nodes"
        lite_attention = self.lite_attention[split_idx]
        return lite_attention(query, key, value, scale, return_softmax_lse)

    def reset_skip_state(self):
        for lite_attention in self.lite_attention:
            lite_attention.reset_skip_state()

    def set_threshold(self, threshold: float):
        for lite_attention in self.lite_attention:
            lite_attention.set_threshold(threshold)
    
    def enable_skip_optimization(self, enable: bool = True):
        for lite_attention in self.lite_attention:
            lite_attention.enable_skip_optimization(enable)