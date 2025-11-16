"""
LiteAttention: A lightweight Flash Attention 3 wrapper with skip list optimization.

This module provides a clean interface for Flash Attention 3 with internal management
of read and write skip lists, hiding the complexity from users.
"""

import torch
import torch.nn.functional as F
import os
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt

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
    
    def __init__(self, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 2, reverse_skip_list: bool = True):
        # Internal skip list management
        self._skip_list = None
        self._phase = 0
        self.reverse_skip_list = reverse_skip_list
        self._last_batch_size = None
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
    def calc_percentage_per_head(read_list: torch.Tensor) -> float:
        """
        Calculate the percentage of non-skipped attention computations per head.
        read_list: [batch, heads, qtiles, ktiles + 1]
        """

        read_list = read_list.to(torch.int64)
        # remove the first element (the length of the skip list)
        reshaped_read_list = read_list[..., 1:] # [batch, heads, qtiles, ktiles]

        # pad last dimension to be even
        # [batch, heads, qtiles, ktiles] -> [batch, heads, qtiles, ktiles + (ktiles % 2)]
        if reshaped_read_list.shape[-1] % 2 != 0:
            # Pad with 0 if uneven
            padding_shape = list(reshaped_read_list.shape)
            padding_shape[-1] = 1
            padding = torch.zeros(padding_shape, dtype=reshaped_read_list.dtype, device=reshaped_read_list.device)
            reshaped_read_list = torch.cat([reshaped_read_list, padding], dim=-1)
        
        # reshaped_read_list: [batch, heads, qtiles, ktiles + (ktiles % 2)] -> [batch, heads, qtiles, -1, 2]
        reshaped_read_list = reshaped_read_list.view(
            reshaped_read_list.shape[0],
            reshaped_read_list.shape[1],
            reshaped_read_list.shape[2],
            -1, 2)
        # range_sizes: [batch, heads, qtiles, -1]
        range_sizes = (reshaped_read_list[..., 1] - reshaped_read_list[..., 0]).abs()
        # not_skipped_per_head: [batch, heads, qtiles, -1]
        not_skipped_per_head = range_sizes.cumsum(dim=-1)
        # the index for the end of the ranges in not_skipped_per_head
        # skip_list_sizes: [batch, heads, qtiles]
        skip_list_sizes = (read_list[:, :, :, 0] - 1) // 2
        # real_not_skipped_per_head: [batch, heads, qtiles, -1] -> [batch, heads, qtiles]
        real_not_skipped_per_head = torch.gather(not_skipped_per_head, dim=-1, index=skip_list_sizes.unsqueeze(-1)).squeeze(-1)
        # take the mean for every q tile
        num_of_k_tiles = read_list.shape[-1] - 1
        return real_not_skipped_per_head / num_of_k_tiles

    @staticmethod
    def calc_percentage(read_list: torch.Tensor) -> float:
        """
        Calculate the percentage of non-skipped attention computations.
        read_list: [batch, heads, qtiles, ktiles + 1]
        """
        return LiteAttention.calc_percentage_per_head(read_list).mean()

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
        """
        Initialize skip list tensors based on query shape.
        Skip List Format:
        ================
        skip_list.shape == [2, batch, heads, qtiles, ktiles + 1]
        example_skip_list = skip_list[0,0,0,0,:]
        the data inside example_skip_list is of the form:
        [length, start_0, end_0, start_1, end_1, ..., start_{(length/2)-1}, end_{(length/2)-1}, uninitialized value, ...., uninitialized value]
        start_0 >= end_0 >= start_1 >= end_1 >= ... >= start_{(length/2)-1} >= end_{(length/2)-1}
        start_0 <= end_0 <= start_1 <= end_1 <= ... <= start_{(length/2)-1} <= end_{(length/2)-1}
        **the length is always an even number!
        """

        # the number of bytes needed to represent dtype (size(dtype) if it where C code)
        element_size = dtype.itemsize
        # kBlockM: number of rows in each tile, kBlockN: number of columns in each tile
        kBlockM, kBlockN = LiteAttention.get_MN(head_dim, element_size, v_colmajor)

        # qtiles: number of tiles in each row of Q@K.T
        qtiles = LiteAttention.ceil_div(seq_len, kBlockM)
        # ktiles: number of tiles in each column of Q@K.T
        ktiles = LiteAttention.ceil_div(seq_len, kBlockN)
        
        # skip_list = torch.zeros(2, batch, heads, qtiles, ktiles + 1, dtype=torch.int32, device=device)
        # skip_list[:, :, :, :, 1] = ktiles - 1
        # skip_list[:, :, :, :, 0] = 2  # First element is the length of skip list

        # memory allocation for the skip list data structre
        # Shape explained:
        # skip_list.shape[0] == 2:
        #   some times skip_list[0] would be the read_list and skip_list[1] the write_list and some times the oposite
        # skip_list.shape[4] == ktiles + 1:
        #   the +1 is because the first element (skip_list[..., 0]) is always the length of the skip list

        # dtype = torch.int32
        dtype = torch.int16
        skip_list = torch.empty(2, batch, heads, qtiles, ktiles + 1, dtype=dtype, device=device)
        skip_list[0, :, :, :, 0:3] = torch.tensor([2, ktiles - 1, -1], dtype=dtype, device=device)
        # skip_list[0, :, :, :, 0:3] = torch.tensor([2, ktiles - 1, 0], dtype=torch.int32, device=device)
        """
        why the order is reversed? (ktiles - 1 and then 0)
        we iterate in reverse order inside of the kernel like in the following code:
        for i in range(start = ktiles - 1, end = -1, step = -1):
            pass
        ktiles - 1
        ktiles - 2
        ...
        1
        0
        **important: the end index is inclusive!
        """

        return skip_list

    def _init_skip_list(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Initialize skip list tensors based on query shape."""
        # (query @ key.T).shape: [batch, heads, seq_len, seq_len]
        batch, seq_len, heads, head_dim = query.shape
        assert batch <= self.max_batch_size, "batch size must be less than or equal to max_batch_size (modify max_batch_size in LiteAttention constructor)"
        # attributes that help with finding the TILE size
        v_colmajor = value.shape[-3] == head_dim
        dtype = query.dtype
        device = query.device
        return LiteAttention.init_skip_list(self.max_batch_size, seq_len, heads, head_dim, v_colmajor, dtype, device)
    
    
    def _get_read_write_lists(self, query: torch.Tensor, value: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the current read and write lists for this attention step."""

        # if disable skipping, return None
        if not self.enable_skipping:
            return None, None
            
        # attributes we check in the decision to REINITIALIZE the skip list
        current_seq_len = query.shape[1]
        head_dim = query.shape[-1]
        current_head_dim = head_dim
        current_num_heads = query.shape[2]
        v_colmajor = value.shape[-3] == head_dim
        dtype = query.dtype
        device = query.device
        
        # Initialize or reinitialize skip list if needed
        # we always enter this in the first call
        if (self._skip_list is None or 
            self._last_seq_len != current_seq_len or 
            self._skip_list.device != query.device or
            self._last_head_dim != current_head_dim or
            self._last_v_colmajor != v_colmajor or
            self._last_dtype != dtype or
            self._last_device != device or
            self._last_num_heads != current_num_heads
            ):

            # initialize the skip list (actually allocate the memory)
            self._skip_list = self._init_skip_list(query, value)
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
    def _expand_must_do_list(must_do_list, list_shape, query, value):
        """
        This function expands the 1d list to a list per head per batch per qi.
        """

        return torch.tensor([len(must_do_list)] + must_do_list, dtype=torch.int16, device=query.device)

        # head_dim = query.shape[-1]
        # v_colmajor = value.shape[-3] == head_dim
        # dtype = query.dtype
        # device = query.device

        # element_size = dtype.itemsize
        # q_tile_size, k_tile_size = LiteAttention.get_MN(head_dim, element_size, v_colmajor)

        # must_do_list = [len(must_do_list)] + must_do_list # append the list length at the start

        # # from sequence indices to block indices:
        # for i in range(1,must_do_list[0]+1):
        #     if i % 2 == 1:
        #         must_do_list[i] = (must_do_list[i] + k_tile_size - 1) // k_tile_size  # round up start indices
        #     else:
        #         must_do_list[i] = must_do_list[i] // k_tile_size  # round down end indices

        # # print("must_do_list", must_do_list)

        # values = torch.tensor(must_do_list, dtype=torch.int16, device=device)
        # values = torch.cat([values, torch.zeros(list_shape[3] - values.size(0), dtype=values.dtype, device=values.device)])
        # expanded = values.repeat(*list_shape[:3], 1).contiguous()
        # return expanded
    
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                 scale: Optional[float] = None, return_softmax_lse: bool = False, must_do_list: list = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

        # handle must-do list - expand the 1d list to a list per head per batch per qi
        if (must_do_list is not None) and self.enable_skipping:
            must_do_list_expanded = self._expand_must_do_list(must_do_list, write_list.shape, query, value)
        else:
            must_do_list_expanded = self._expand_must_do_list([0,0], write_list.shape, query, value)  # [0,0] is for an empty must-do list

        # print("must_do_list_expanded", must_do_list_expanded.shape)
        
        # Perform flash attention 3 with skip lists
        output = flash_attn_func(
            q=query,
            k=key, 
            v=value,
            softmax_scale=scale,
            attn_read_list=read_list,
            attn_must_do_list=must_do_list_expanded,
            attn_write_list=write_list,
            thr=self.threshold,
            return_softmax_lse=return_softmax_lse,
            reverse_skip_list=self.reverse_skip_list,
            # self._phase == 1 because we changed it in _get_read_write_lists!
            phase=(self._phase == 1) if self.reverse_skip_list else False
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
    
    def visualize_skips(self, query: torch.Tensor, key: torch.Tensor, heads_list: torch.Tensor, scale: float, save_path: str, max_res: int = 520, name_prefix: str = "", do_softmax: bool = True):
        '''
        heads_list: [N], heads_list[i] == head_idx
        query: (batch, seq_len, heads, head_dim)
        key: (batch, seq_len, heads, head_dim)
        name_prefix: optional prefix for the saved file names
        '''
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

        kBlockM, kBlockN = LiteAttention.get_MN(key.shape[-1], key.dtype.itemsize)
        # Add grid overlay
        height, width = max_res, max_res
        ratio_height = height / seq_len_q
        ratio_width = width / seq_len_k

        grid_height = kBlockM * ratio_height
        grid_width = kBlockN * ratio_width

        # Calculate grid line positions
        y_positions = [b * grid_height for b in range(int(height / grid_height) + 1) if b * grid_height <= height]
        x_positions = [b * grid_width for b in range(int(width / grid_width) + 1) if b * grid_width <= width]

        for b in range(batch):
            for h in heads_list:
                # Calculate QK for this specific head
                q_head = query[b:b+1, :, h:h+1, :]  # (1, seq_len_q, 1, head_dim)
                k_head = key[b:b+1, :, h:h+1, :]    # (1, seq_len_k, 1, head_dim)
                
                # Reshape: (1, seq_len, 1, head_dim) -> (1, 1, seq_len, head_dim)
                q_reshaped = q_head.transpose(1, 2)  # (1, 1, seq_len_q, head_dim)
                k_reshaped = k_head.transpose(1, 2)  # (1, 1, seq_len_k, head_dim)
                
                # Compute attention
                QK = (q_reshaped @ k_reshaped.transpose(-2, -1)) * scale  # (1, 1, seq_len_q, seq_len_k)
                if do_softmax:
                    attn_softmaxed = torch.softmax(QK, dim=-1)
                else:
                    attn_softmaxed = QK
                attn_down = F.adaptive_max_pool2d(
                    attn_softmaxed,  # (1, 1, seq_len_q, seq_len_k)
                    output_size=(max_res, max_res)
                )  # -> (1, 1, max_res, max_res)
                
                attn_map = attn_down[0, 0]  # (max_res, max_res)
                
                current_skip_list = skip_list[b, h][None, None, ...]
                perecentage = self.calc_percentage(current_skip_list)

                plt.figure(figsize=(6, 6))
                attn_cpu = attn_map.detach().float().cpu()
                plt.imshow(attn_cpu, cmap='viridis', interpolation='nearest')
                plt.title(f"Batch {b} | Head {h} | Percentage {perecentage * 100:.2f}% | Do Softmax: {do_softmax}")
                
                # Add horizontal grid lines
                for y in y_positions:
                    plt.axhline(y=y-0.5, color='black', linewidth=0.2, alpha=0.7)
                
                # Add vertical grid lines  
                for x in x_positions:
                    plt.axvline(x=x-0.5, color='black', linewidth=0.2, alpha=0.7)


                for i, row_skip_list in enumerate(current_skip_list[0, 0]):
                    # print(row_skip_list.shape)
                    l_row = row_skip_list[0]
                    # end0, start1, end1, start2, ...
                    # width_ranges = (row_skip_list[1 : l_row + 1] + 1) * grid_width
                    width_ranges = (row_skip_list[1 : l_row + 1] + step) * grid_width
                    # height
                    row_height = i * grid_height

                    width_ranges = width_ranges.view(-1, 2).cpu()
                    # for end, start in width_ranges:
                    for r1, r2 in width_ranges:
                        start = min(r1, r2)
                        end = max(r1, r2)
                        # rect = plt.Rectangle((start, row_height), end + grid_width - start, grid_height, facecolor='white', edgecolor='none', linewidth=0.4, alpha=0.3)
                        rect = plt.Rectangle((start, row_height), end - start, grid_height, facecolor='white', edgecolor='none', linewidth=0.4, alpha=0.3)
                        plt.gca().add_patch(rect)
                
                plt.axis("off")
                plt.tight_layout()

                batch_head_dir = os.path.join(save_path, f"batch_{b}", f"head_{h}")
                filename = f"{name_prefix}.png" if name_prefix else "visualization.png"
                file_path = os.path.join(batch_head_dir, filename)
                plt.savefig(file_path, dpi=150)
                plt.close()

    @property
    def read_list(self) -> torch.Tensor:
        if self._skip_list is None:
            return None
        return self._skip_list[self._phase, :self._last_batch_size]
    
    @property
    def write_list(self) -> torch.Tensor:
        if self._skip_list is None:
            return None
        return self._skip_list[1 - self._phase, :self._last_batch_size]

class SeqParallelLiteAttention:
    def __init__(self, num_nodes: int, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 2):

        self.num_nodes = num_nodes
        self.lite_attention = [LiteAttention(enable_skipping, threshold, max_batch_size) for _ in range(num_nodes)]
        self.set_threshold(threshold)

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, split_idx: int,
                 scale: Optional[float] = None, return_softmax_lse: bool = False, must_do_list: list = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert split_idx < self.num_nodes, "split_idx must be less than num_nodes"
        lite_attention = self.lite_attention[split_idx]
        return lite_attention(query, key, value, scale, return_softmax_lse, must_do_list)

    def reset_skip_state(self):
        for lite_attention in self.lite_attention:
            lite_attention.reset_skip_state()

    def set_threshold(self, threshold: float):
        for lite_attention in self.lite_attention:
            lite_attention.set_threshold(threshold)
    
    def enable_skip_optimization(self, enable: bool = True):
        for lite_attention in self.lite_attention:
            lite_attention.enable_skip_optimization(enable)