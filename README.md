# LiteAttention: Transforming Video Diffusion with Temporal Sparse Attention

![LiteAttention Hero Image](assets/wan_outputs/baseline.gif) *Sample output demonstrating video quality retention.*

### [Project Page](https://moonmath-ai.github.io/LiteAttention/) | [arXiv](https://arxiv.org/abs/2511.11062) | [HuggingFace](https://huggingface.co/papers/2511.11062) | [MoonMath.ai](https://moonmath.ai)

Video diffusion models generate stunningly realistic content, but their computational demands—specifically within self-attention layers—are staggering. To address this, we present **LiteAttention**, a temporal sparse attention mechanism directly addressing the redundancy in attention computations across diffusion timesteps. 

By identifying non-essential tiles early in the generation process and propagating these "skip decisions" forward, LiteAttention eliminates redundant computations without repeated profiling overheads. The result? **Up to 54% attention sparsity** on production-grade models like Wan2.1 and Wan2.2, with **zero degradation in visual quality**. This translates to a nearly **1.9x speedup** in wall-clock time.

---

## 🌟 What's New: Version History & Features

LiteAttention is actively developed to provide the fastest, most flexible sparse attention for diffusion models. Here is the recent evolution of the codebase:

### **v0.4.0 (Latest): Calibration Registry & Dynamic Thresholding**
*   **Calibration Registry:** An experimental feature to automatically search and find per-layer, per-timestep skip thresholds that meet a target error budget. This balances speed and quality better than fixed thresholds.
*   **Dynamic Thresholding:** Ability to save (`mode="calib"`) and load (`mode="load"`) these optimal threshold configurations from a TOML file.
*   **Fixes:** Resolved sequence parallelism correctness issues for rectangular QK skip lists and fixed default modes for `torch.compile` support.

### **v0.3.x: INT8 Quantization & Bi-directional Consumer**
*   **INT8 Quantization:** Added support for INT8 quantization (`use_int8=True`) for Q (per-block) and K (per-block with channel-wise mean smoothing), significantly reducing memory usage and boosting performance.
*   **Full Producer-Consumer Pipeline:** Introduced q-pad and bi-directionality for enhanced execution efficiency and sequence handling.

### **v0.2.0: Programmable Block Processing (`must-do` & `must-skip`)**
*   **Fine-Grained Sequence Control:** Added `must_do_list` and `must_skip_list` parameters. You can now explicitly define token ranges (e.g., prompt tokens vs padding) that *must* always be computed or that can *always* be skipped, bypassing the threshold logic entirely.

### **v0.1.0: Initial Release & Core Architecture**
*   **Evolutionary Computation Skips (QK-Skip):** The core algorithm that maintains a Skip-Mask, identifying non-essential tiles and completely bypassing the attention iteration (QK product, softmax, PV product) in later timesteps.
*   **Sequence Parallelism:** Introduced `SeqParallelLiteAttention` for multi-GPU scale-out.
*   **Softmax LSE:** Added the ability to return the softmax log-sum-exp (`return_softmax_lse=True`) for combining partial attention computations (e.g., separating text-to-video vs video-to-video attention).

---

## 🔍 How It Works: The QK-Skip Algorithm

Traditional dynamic sparse attention methods evaluate sparsity criteria at *every single timestep*. This incurs a massive 10-20% runtime overhead just to figure out what to compute.

LiteAttention introduces **evolutionary computation skips** by leveraging the *temporal coherence* of diffusion attention. 
1.  **Early Profiling:** During the initial diffusion timesteps, we compute the full attention matrix and track the maximum log-attention score for each tile.
2.  **The Skip-Mask:** If a tile's score falls below a set `threshold`, it's marked as skippable.
3.  **Forward Propagation:** Once a tile is marked as skippable, the *entire* attention computation for that tile is bypassed for all subsequent timesteps. 

This gives us the **content adaptivity** of dynamic sparsity without the overhead, acting like an efficient, static pre-computation.

---

## 📊 Performance Benchmark

LiteAttention achieves state-of-the-art speeds while maintaining top-tier visual consistency metrics (evaluated via VBench).

| Model | AQ ↑ | BC ↑ | DD ↑ | IQ ↑ | SC ↑ | TF ↑ | TS ↑ | Sparsity ↑ | Runtime ↓ (Speedup) |
|-------|------|------|------|------|------|------|------|-----------|--------------------|
| **Wan2.1-14B Base** | 0.676 | 0.977 | 0.417 | 68.74 | 0.965 | 0.962 | 0.137 | 0% | 1707 sec (1.00x) |
| **Wan2.1-14B + LiteAttn** | **0.677** | **0.975** | **0.500** | *66.76* | 0.963 | 0.962 | **0.142** | **42%** | **902 sec (1.89x)** |
| **Wan2.2-14B Base** | 0.693 | 0.977 | 0.583 | 72.73 | 0.970 | 0.953 | 0.133 | 0% | 1473 sec (1.00x) |
| **Wan2.2-14B + LiteAttn** | **0.698** | **0.977** | 0.500 | 71.44 | 0.969 | 0.953 | **0.135** | **32%** | **893 sec (1.65x)** |

*VBench Metrics: AQ (Aesthetic Quality), BC (Background Consistency), DD (Dynamic Degree), IQ (Imaging Quality), SC (Subject Consistency), TF (Temporal Flickering), TS (Temporal Style)*

<details>
<summary>Click to view Ablation Study: Sparsity vs Runtime</summary>

| Sparsity | Self-Attention Runtime | Runtime Improvement |
|----------|------------------------|---------------------|
| 0% | 695 sec | 0% (baseline) |
| 21% | 573 sec | 18% |
| 42% | 418 sec | 40% |
| 57% | 308 sec | 56% |
| 77% | 163 sec | 77% |

*The near-linear scaling demonstrates the efficiency of the QK-Skip algorithm.*
</details>

---

## 🎥 Visual Results: Wan2.1-14B Configurations

| Threshold | Generation Time | Preview |
|:---:|:---:|:---:|
| Baseline (no skip) | 23m 51s | ![baseline](assets/wan_outputs/baseline.gif)|
| Threshold -10 | 14m 19s | ![threshold -10](assets/wan_outputs/minus10.gif)|
| Threshold -3 | 11m 46s | ![threshold -3](assets/wan_outputs/minus3.gif)|
| Threshold 0 | 8m 31s | ![threshold zero](assets/wan_outputs/zero.gif)|

---

## 🔧 Installation

**Requirements:** H100/H200 GPU, CUDA >= 12.8, C++ 20, PyTorch 2.2+, Linux.

LiteAttention is built on top of FlashAttention3 and requires ninja for fast compilation.
```bash
# Ensure ninja is working properly
pip uninstall -y ninja && pip install ninja

# Install dependencies
pip install torch packaging einops structlog tomli-w

# Clone and build
git clone https://github.com/moonmath-ai/LiteAttention.git
cd LiteAttention/hopper

# Note: If RAM limited, prepend `MAX_JOBS=4`
pip install --no-build-isolation .
```

---

## 🔌 Integration Guide

LiteAttention is designed as a drop-in replacement for standard flash attention modules in DiT (Diffusion Transformer) models. 

### 1. Basic Substitution
Replace your standard attention call with a `LiteAttention` instance. **Crucially, instantiate a separate `LiteAttention` object for each layer** so they maintain independent skip states.

```python
from lite_attention import LiteAttention

class MyDiTBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Enable skipping and INT8 quantization!
        self.lite_attention = LiteAttention(enable_skipping=True, use_int8=True)

    def forward(self, q, k, v, must_do_list=None):
        # ...
        # Standard input format: (batch, seq_len, heads, head_dim)
        x = self.lite_attention(q, k, v, must_do_list=must_do_list)
        return x
```

> ⚠️ **Important:** Skip optimization should *only* be enabled for **video-to-video self-attention**. For cross-attention or text-to-video partial computations, disable skipping using `self.attn.enable_skip_optimization(enable=False)`.

### 2. Multi-GPU Sequence Parallelism
For scaled inference across nodes, utilize `SeqParallelLiteAttention`. You must pass the `split_idx` indicating the K/V split being processed by the current node.
```python
from lite_attention import SeqParallelLiteAttention
self.attn = SeqParallelLiteAttention(num_nodes=8)
x = self.attn(query, key, value, split_idx, scale)
```

### 3. Using the Calibration Registry (v0.4.0+)
To unlock optimal generation/speed ratios, employ the new Registry to automatically calibrate thresholds for your specific model.

```python
from lite_attention import LiteAttentionRegistry

model = build_my_model(...) # Initializes modules utilizing LiteAttention()

# Wrap the model. Modes: "calib", "load", "const"
registry = LiteAttentionRegistry.from_model(
    model,
    mode="calib", 
    filename="optimized_thresholds.toml", 
    calib_config={"target_error": 0.05, "metric": "L1"},
)

# Run Inference
video = model.generate(prompt, ...)

# Save the calibrated thresholds (triggers only if mode="calib")
registry.save_if_calib() 
```

To run normally using a fixed static threshold, just initialize with `mode="const"` and `threshold=-10.0`.

---

## 📚 Citation & Acknowledgements

If you utilize LiteAttention in your research or deployment, please consider citing:

```bibtex
@misc{shmilovich2025liteattentiontemporalsparseattention,
      title={LiteAttention: A Temporal Sparse Attention for Diffusion Transformers}, 
      author={Dor Shmilovich and Tony Wu and Aviad Dahan and Yuval Domb},
      year={2025},
      eprint={2511.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Built upon the incredible foundation of [FlashAttention3](https://github.com/Dao-AILab/flash-attention) by Tri Dao. We also thank the teams behind SparseVideoGen, RadialAttention, SageAttention, Wan2.1, and LTX-Video. 

**License:** LiteAttention inherits the BSD 3-Clause license from FA3 for original code; new LiteAttention additions are distributed under the MIT license. See [LICENSE-BSD](LICENSE-BSD) and [LICENSE-MIT](LICENSE-MIT).
