# LiteAttention

LiteAttention is a lightweight Flash Attention 3 wrapper with skip list optimization.

## Requirements
- H100 / H200 GPU
- CUDA >= 12.8
- CUDA toolkit
- PyTorch 2.2 and above
- `packaging` Python package (`pip install packaging`)
- `ninja` Python package (`pip install ninja`) *
- Linux

\* Make sure that `ninja` is installed and that it works correctly (e.g. `ninja
--version` then `echo $?` should return exit code 0). If not (sometimes `ninja
--version` then `echo $?` returns a nonzero exit code), uninstall then reinstall
`ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`,
compiling can take a very long time (2h) since it does not use multiple CPU
cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.

## Installation

Clone this repo and build from source

```sh
cd hopper
python setup.py install
```

If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might
run too many parallel compilation jobs that could exhaust the amount of RAM. To
limit the number of parallel compilation jobs, you can set the environment
variable `MAX_JOBS`:
```sh
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## How to use LiteAttention

### Basic Usage (Single GPU)

```python
def LiteAttention(enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 4)
```

```python
from lite_attention import LiteAttention


# In your model, set the attention class to be LiteAttention with an optional threshold
self.attn = LiteAttention(threshold=-6.0)
.
.
.
hidden_states_a_raw = self.attn(query, key, value, scale)

# If you don't know the threshold at the point of initialization, you can set it later via the set_threshold function
self.attn = LiteAttention()
.
.
.
self.attn.set_threshold(threshold=calculated_threshold)

# Additionally, we provide the capability to reset the skip state if needed 
self.attn.reset_skip_state()

# or to toggle the skipping optimization; turning it off falls back to regular FA3
self.attn.enable_skip_optimization(enable=False)
```

> [!IMPORTANT]
> Each `LiteAttention` instance maintains internal skip state that should not be shared across different attention layers in your model. Create a separate instance for each attention layer:
> ```python
> # Correct: Separate instances for different layers
> self.attn_layer1 = LiteAttention(threshold=-6.0)
> self.attn_layer2 = LiteAttention(threshold=-6.0)
> 
> # Incorrect: Don't reuse the same instance across different layers
> self.shared_attn = LiteAttention(threshold=-6.0)  # Don't share!
> ```
> However, **do reuse** the same instance across multiple forward passes (different calls to your model over time).

For parts of the sequence that should not be skipped use the must-do feature. Pass the must_do_list parameter:

```python
self.attn(query, key, value, scale, must_do_list = must_do_list)
```

The must_do_list defines ranges that must not be skipped and the format is as follows:

    must_do_list = [start_0, end_0, start_1, end_1, ...]
    start_i - start index of a range we must no skip. (inclusive)
    end_i - end index of a range we must not skip. (exclusive)
    IMPORTANT: start_i > end_i > start_(i+1) > end_(i+1) > ... because we iterate in reverse order inside of the kernel.

For example, if we have a sequence of length 100, the must_do_list could look like this:

```python
must_do_list = [80, 60, 45, 40, 12, 2]
```

### Multi-GPU Usage (Sequence Parallelism)

When using multi-GPU with sequence parallelism, use `SeqParallelLiteAttention`:

```python
def SeqParallelLiteAttention(num_nodes: int, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 4)
```

```python
from lite_attention import SeqParallelLiteAttention

# In your model, set the attention class to be SeqParallelLiteAttention with the number of nodes
self.attn = SeqParallelLiteAttention(num_nodes=8, threshold=-6.0)
.
.
.
# Pass split_idx to indicate which split (of K and V) we are processing
hidden_states_a_raw = self.attn(query, key, value, split_idx, scale)
```

> [!IMPORTANT]
> When using `SeqParallelLiteAttention`, you **must** provide the `split_idx` parameter in the forward call. This parameter indicates which split of K and V you are currently processing (0 to num_nodes-1), **not** the current GPU index. Each node processes a different split of the K and V tensors in sequence parallel attention.

### Returning Softmax LSE

Both `LiteAttention` and `SeqParallelLiteAttention` support returning the softmax log-sum-exp (LSE) values for combining results from multiple partial attention computations.

Example use case: When you have both text and video tokens, you can break down full self-attention into partial computations:
- **t2t, t2v, v2t**: text-to-text, text-to-video, video-to-text - **no skip optimization**
- **v2v**: video-to-video - **with skip optimization**

```python
# Example: Breaking down full self-attention with text and video tokens
self.attn = LiteAttention(enable_skipping=True, threshold=-6.0)

# Split queries, keys, values into text and video parts
query_text, query_video = query[:, :text_len, :, :], query[:, text_len:, :, :]
key_text, key_video = key[:, :text_len, :, :], key[:, text_len:, :, :]
value_text, value_video = value[:, :text_len, :, :], value[:, text_len:, :, :]

# Disable skip optimization when calculating t2t, t2v, v2t
self.attn.enable_skip_optimization(enable=False)
output_t2t, lse_t2t = self.attn(query_text, key_text, value_text, scale, return_softmax_lse=True)
output_t2v, lse_t2v = self.attn(query_text, key_video, value_video, scale, return_softmax_lse=True)
output_v2t, lse_v2t = self.attn(query_video, key_text, value_text, scale, return_softmax_lse=True)

# Enable skip optimization only for video-to-video
self.attn.enable_skip_optimization(enable=True)
output_v2v, lse_v2v = self.attn(query_video, key_video, value_video, scale, return_softmax_lse=True)

# Combine the partial results using their LSE values to get the final output
```

> [!IMPORTANT]
> LiteAttention should only be used in DiT models

> [!IMPORTANT]
> The skip optimization should **only be enabled for video-to-video self-attention**. For other attention types (e.g., cross-attention or text-to-video attention), you should disable the skip optimization:
> ```python
> # For video-to-video self-attention - keep skipping enabled
> self.attn_self = LiteAttention(enable_skipping=True, threshold=-6.0)
> 
> # For cross-attention or text-to-video attention - disable skipping
> self.attn_cross = LiteAttention(enable_skipping=False)
> ```

## Example of Wan2.1 14B

Import the lite attention module into the [model.py](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py) file

```python
# Import lite_attention for optimized attention
try:
    from lite_attention import LiteAttention
    LITE_ATTENTION_AVAILABLE = True
except ImportError:
    LITE_ATTENTION_AVAILABLE = False
```

Then update the [WanSelfAttention class'](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py#L105) __init__ function to initialize the lite_attention class

```python
class WanSelfAttention(nn.Module):
    def __init__(...):
      .
      .
      .
      # Initialize LiteAttention if available
      if LITE_ATTENTION_AVAILABLE:
          print("Using LiteAttention")
          self.lite_attention = LiteAttention(enable_skipping=True, threshold=-10.0)
      else:
          self.lite_attention = None
```

Lastly, update the forward function to call the lite_attention instance:

```python
    def forward(self, x, seq_lens, grid_sizes, freqs):
      .
      .
      .
      # Apply RoPE to q and k
      q_rope = rope_apply(q, grid_sizes, freqs)
      k_rope = rope_apply(k, grid_sizes, freqs)

      # Use LiteAttention if available, otherwise fall back to flash_attention
      if self.lite_attention is not None:
          # LiteAttention expects (batch, seq_len, heads, head_dim) format
          # and returns (batch, seq_len, heads * head_dim) format
          # Convert to bfloat16 for memory efficiency; FA3 does not support float32
          q_rope = q_rope.bfloat16()
          k_rope = k_rope.bfloat16()
          v = v.bfloat16()
          x = self.lite_attention(q_rope, k_rope, v)
          # Convert result back to float32 to maintain consistency with model expectations
          x = x.float()
      else:
          x = flash_attention(
              q=q_rope,
              k=k_rope,
              v=v,
              k_lens=seq_lens,
              window_size=self.window_size)
```

### Wan2.1 Times

| Threshold           | Time    | Video                |
|:-------------------:|:-------:|:-----------------------------:|
| Baseline (no skip)  | 23m51s  | ![baseline](assets/wan_outputs/baseline.gif)|
| -10                 | 14m19s  | ![thershold -10](assets/wan_outputs/minus10.gif)|
| -3                  | 11m46s  | ![thershold -3](assets/wan_outputs/minus3.gif)|
| 0                   | 8m31s    | ![thershold zero](assets/wan_outputs/zero.gif)|


## Debugging

You can see additional debug logs by setting the `LITE_ATTENTION_VERBOSE` to anything other than "FALSE"

If you want to be able to test thresholds greater than 0, you need to set the `LITE_ATTENTION_DEBUG` environment variable to anything other than "FALSE"

## License

LiteAttention is build on top of FA3 which has a BSD 3-Clause license. As such the original code maintains that license and any new code for LiteAttention is distributed under an MIT license.

See [LICENSE-BSD](LICENSE-BSD) and [LICENSE-MIT](LICENSE-MIT) for further details.
