# LiteAttention

### [Project Page](https://moonmath-ai.github.io/LiteAttention/) | [arXiv](https://arxiv.org/abs/2511.11062)  | [HuggingFace](https://huggingface.co/papers/2511.11062) | [MoonMath.ai](https://moonmath.ai)

We present *LiteAttention*, a temporal sparse attention mechanism that exploits the slow evolution of attention patterns across diffusion timesteps. By identifying non-essential tiles early and propagating skip decisions forward, LiteAttention eliminates redundant attention computations without repeated profiling overheads. LiteAttention achieves **up to 54% attention sparsity** on production video diffusion models **with no degradation in generation quality**.


## 📖 Overview

**LiteAttention** is a **temporal sparse attention mechanism** for video diffusion models that exploits the **temporal coherence of sparsity patterns** across denoising timesteps. Unlike traditional sparse attention methods, LiteAttention achieves the adaptivity of dynamic methods with the efficiency of static ones. Here are our core contributions:

- **Evolutionary Computation Skips**: Identify non-essential tiles once during early denoising and propagate skip decisions forward through the entire trajectory.
- **Full-Stage Elimination**: Skip the entire attention iteration (QK product, softmax, PV product) for marked tiles, not just partial stages.
- **Error Calibration**: Assign different error bounds to different timesteps, with stricter bounds for earlier timesteps that have greater influence on the final output.
- **Zero Training Required**: Production-ready, requires no model retraining or architectural modifications.

## 🔍 How It Works

LiteAttention introduces **evolutionary computation skips** that leverage temporal coherence in diffusion attention:

**QK-Skip Algorithm**: Unlike dynamic methods that repeatedly recompute sparsity at every step (incurring 10-20% overhead), LiteAttention maintains a Skip-Mask that is updated at each timestep. As the diffusion process progresses, the number of tiles marked for skipping gradually increases. Once a tile is marked as skippable, the entire attention iteration is bypassed for subsequent timesteps.

This approach combines:
- **Content adaptivity** of dynamic sparsity (patterns derived from actual attention statistics)
- **Efficiency** of static sparsity (no per-step re-evaluation overhead)
- **Completeness** of full computation elimination

## 📊 Performance

LiteAttention achieves remarkable video quality with significant speedups compared to other sparse attention methods. We evaluate using VBench metrics on production video diffusion models.

### Summary Results

| Model | AQ ↑ | BC ↑ | DD ↑ | IQ ↑ | SC ↑ | TF ↑ | TS ↑ | Sparsity ↑ | Runtime ↓ |
|-------|------|------|------|------|------|------|------|-----------|-----------|
| **Wan2.1-14B** | **0.677** | **0.975** | **0.500** | 66.76 | 0.963 | 0.962 | **0.142** | 42% | **902 sec** |
| **Wan2.2-14B** | **0.698** | **0.977** | **0.500** | 71.44 | **0.969** | **0.953** | **0.135** | 32% | **893 sec** |

*VBench Metrics: AQ (Aesthetic Quality), BC (Background Consistency), DD (Dynamic Degree), IQ (Imaging Quality), SC (Subject Consistency), TF (Temporal Flickering), TS (Temporal Style)*

### Speedup Analysis

LiteAttention achieves significant speedups over FlashAttention3 baseline:

- **Wan2.1-14B**: 1707 sec → 902 sec = **1.89× speedup** (47% time reduction)
- **Wan2.2-14B**: 1473 sec → 893 sec = **1.65× speedup** (39% time reduction)

LiteAttention achieves the **best runtime** on both models while maintaining **superior quality metrics** compared to SparseVideoGen and RadialAttention.

<details>
<summary>Click to see detailed benchmark comparisons</summary>

### Wan2.1-14B Detailed Comparison

| Method | AQ ↑ | BC ↑ | DD ↑ | IQ ↑ | SC ↑ | TF ↑ | TS ↑ | Sparsity ↑ | Runtime ↓ |
|--------|------|------|------|------|------|------|------|-----------|-----------|
| FlashAttention3 | 0.676 | 0.977 | 0.417 | **68.74** | 0.965 | 0.962 | 0.137 | 0% | 1707 sec |
| SparseVideoGen | *0.665* | *0.971* | **0.500** | *68.58* | 0.962 | 0.959 | *0.066* | *66%* | *1019 sec* |
| RadialAttention | 0.660 | 0.970 | *0.417* | 64.73 | **0.964** | **0.972** | 0.061 | **74%** | 1192 sec |
| **LiteAttention** | **0.677** | **0.975** | **0.500** | *66.76* | *0.963* | *0.962* | **0.142** | 42% | **902 sec** |

### Wan2.2-14B Detailed Comparison

| Method | AQ ↑ | BC ↑ | DD ↑ | IQ ↑ | SC ↑ | TF ↑ | TS ↑ | Sparsity ↑ | Runtime ↓ |
|--------|------|------|------|------|------|------|------|-----------|-----------|
| FlashAttention3 | 0.693 | 0.977 | **0.583** | **72.73** | 0.970 | 0.953 | 0.133 | 0% | 1473 sec |
| SparseVideoGen | *0.689* | 0.962 | *0.417* | *72.24* | 0.961 | *0.952* | *0.061* | **66%** | *1022 sec* |
| RadialAttention | 0.682 | *0.974* | **0.500** | **72.73** | *0.967* | 0.947 | *0.061* | **66%** | 1207 sec |
| **LiteAttention** | **0.698** | **0.977** | **0.500** | 71.44 | **0.969** | **0.953** | **0.135** | *32%* | **893 sec** |

*Best results in **bold**, second-best in *italic**

</details>

### Ablation Study: Sparsity vs Runtime

Our ablation studies demonstrate that runtime improvement scales with attention sparsity:

| Sparsity | Self-Attention Runtime | Runtime Improvement |
|----------|------------------------|---------------------|
| 0% | 695 sec | 0% (baseline) |
| 21% | 573 sec | 18% |
| 42% | 418 sec | 40% |
| 57% | 308 sec | 56% |
| 77% | 163 sec | 77% |

The near-linear scaling between sparsity and runtime improvement demonstrates the efficiency of our QK-Skip algorithm.

## 🎥 Visual Results

### 🔹 Wan2.1-14B Generation Times

| Threshold           | Time    | Video                |
|:-------------------:|:-------:|:-----------------------------:|
| Baseline (no skip)  | 23m51s  | ![baseline](assets/wan_outputs/baseline.gif)|
| -10                 | 14m19s  | ![threshold -10](assets/wan_outputs/minus10.gif)|
| -3                  | 11m46s  | ![threshold -3](assets/wan_outputs/minus3.gif)|
| 0                   | 8m31s    | ![threshold zero](assets/wan_outputs/zero.gif)|

## 🔧 Installation

### Requirements
- H100 / H200 GPU
- CUDA >= 12.8
- CUDA toolkit
- C++ 20
- PyTorch 2.2 and above
- `packaging` Python package (`pip install packaging`)
- `ninja` Python package (`pip install ninja`) *
- Linux

\* Make sure that `ninja` is installed and that it works correctly (e.g. `ninja --version` then `echo $?` should return exit code 0). If not (sometimes `ninja --version` then `echo $?` returns a nonzero exit code), uninstall then reinstall `ninja` (`pip uninstall -y ninja && pip install ninja`). Without `ninja`, compiling can take a very long time (2h) since it does not use multiple CPU cores. With `ninja` compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.

### Build from Source

Clone this repo and build from source:

```sh
git clone https://github.com/moonmath-ai/LiteAttention.git
cd LiteAttention/hopper
pip install .
```

If your machine has less than 96GB of RAM and lots of CPU cores, `ninja` might run too many parallel compilation jobs that could exhaust the amount of RAM. To limit the number of parallel compilation jobs, you can set the environment variable `MAX_JOBS`:

```sh
MAX_JOBS=4 pip install .
```

## 🚀 Usage

### Basic Usage (Single GPU)

```python
def LiteAttention(
    enable_skipping: bool = True, 
    threshold: float = -10.0, 
    max_batch_size: int = 2, 
    reverse_skip_list: bool = True, 
    use_int8: bool = False
)
```

**Parameters:**
- `enable_skipping` (bool): Whether to enable skip list optimizations. Defaults to `True`. When `False`, performs standard Flash Attention.
- `threshold` (float): Log-space threshold for skipping tiles. Defaults to `-10.0`. Tiles with `max(log-attention-score) < threshold` will be skipped. Must be negative in non-debug mode. Lower values generally mean more aggressive skipping.
- `max_batch_size` (int): Maximum batch size to pre-allocate memory for. Defaults to `2`. The actual batch size used during inference can be smaller than this value, but not larger.
- `reverse_skip_list` (bool): Whether to use the reversed skip list format (internal optimization). Defaults to `True`.
- `use_int8` (bool): Whether to use Int8 quantization for Q and K. Defaults to `False`. Enables per-block quantization for Q and channel-smoothed per-block quantization for K.

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
    IMPORTANT: start_i < end_i < start_(i+1) < end_(i+1) < ... (regular ascending order).

For example, if we have a sequence of length 100, the must_do_list could look like this:

```python
must_do_list = [2, 12, 40, 45, 60, 80]
```

The must_skip_list defines ranges that can always be skipped according to the same convention as the must_do_list. For example if:
```python
must_skip_list = [40, 80]
```
then all the tokens between 40 and 80 can always be skipped.

### Multi-GPU Usage (Sequence Parallelism)

When using multi-GPU with sequence parallelism, use `SeqParallelLiteAttention`:

```python
def SeqParallelLiteAttention(num_nodes: int, enable_skipping: bool = True, threshold: float = -10.0, max_batch_size: int = 2, use_int8: bool = False)
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

### Quantization Support

LiteAttention supports Int8 quantization for Q and K to further reduce memory usage and increase performance. The quantization scheme is as follows:
- **Q**: Per-block quantization
- **K**: Per-block quantization with channel-wise mean smoothing

To enable quantization, simply set `use_int8=True` when initializing. This works for both `LiteAttention` and `SeqParallelLiteAttention`.

```python
# Enable quantization
self.attn = LiteAttention(enable_skipping=True, threshold=-6.0, use_int8=True)
# or for sequence parallelism
self.attn = SeqParallelLiteAttention(num_nodes=8, threshold=-6.0, use_int8=True)
```

### Visualization

LiteAttention provides a built-in method to visualize the attention patterns and skipped tiles. This is useful for debugging and understanding the effectiveness of the skip mask.

```python
# Run a forward pass first to populate the skip list
output = self.attn(query, key, value, scale)

# Visualize specific heads (e.g., heads 0 and 2)
# Results will be saved to the specified directory (creates batch_{b}/head_{h} subfolders)
self.attn.visualize_skips(
    query=query, 
    key=key, 
    heads_list=torch.tensor([0, 2]), 
    scale=scale, 
    save_path="./attention_viz"
)
```

## 📝 Integration Example: Wan2.1-14B

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

## 🐛 Debugging

You can see additional debug logs by setting the `LITE_ATTENTION_VERBOSE` environment variable to anything other than "FALSE"

If you want to be able to test thresholds greater than 0, you need to set the `LITE_ATTENTION_DEBUG` environment variable to anything other than "FALSE"

## 📚 Citation

If you find LiteAttention useful or relevant to your research, please cite our paper:

```bibtex
@misc{shmilovich2025liteattentiontemporalsparseattention,
      title={LiteAttention: A Temporal Sparse Attention for Diffusion Transformers}, 
      author={Dor Shmilovich and Tony Wu and Aviad Dahan and Yuval Domb},
      year={2025},
      eprint={2511.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.11062}, 
}
```

## 🙏 Acknowledgements

LiteAttention is built on top of [FlashAttention3](https://github.com/Dao-AILab/flash-attention) by Tri Dao and contributors. We thank the FlashAttention team for their foundational work on efficient attention mechanisms.

We also thank the teams behind [SparseVideoGen](https://github.com/svg-project/Sparse-VideoGen), [RadialAttention](https://github.com/mit-han-lab/radial-attention), [SageAttention](https://github.com/thu-ml/SageAttention), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [LTX-Video](https://github.com/Lightricks/LTX-Video) for their insights and benchmarking support.

## License

LiteAttention is build on top of FA3 which has a BSD 3-Clause license. As such the original code maintains that license and any new code for LiteAttention is distributed under an MIT license.

See [LICENSE-BSD](LICENSE-BSD) and [LICENSE-MIT](LICENSE-MIT) for further details.
