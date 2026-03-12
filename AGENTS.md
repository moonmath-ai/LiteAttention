## What This Is

LiteAttention is a CUDA extension built on top of Flash Attention 3 that adds temporal sparse attention for video diffusion models.
It skips redundant attention tiles across diffusion timesteps.
The source code lives in `hopper/` but installs as the `lite_attention` Python package (mapped via `pyproject.toml`'s `[tool.setuptools.package-dir]`).

Key concepts:
- **Skip lists**: Track which attention tiles can be skipped. Double-buffered (read/write alternate each forward pass).
- **Threshold-based skipping**: Tiles are skipped when their max score is too far below the running max (compared in log2 scale). Threshold must be negative in non-debug mode.
- **Must-do/must-skip lists**: Force computation or skipping of specific sequence ranges (e.g., text tokens vs video tokens).

## Build Commands

Requires an H100/H200 GPU, CUDA >= 12.3, and a C++20 compiler.

```bash
git submodule update --init
MAX_JOBS=$(nproc) NVCC_THREADS=4 \
LITE_ATTENTION_DISABLE_BACKWARD=TRUE \
LITE_ATTENTION_DISABLE_FP16=TRUE \
LITE_ATTENTION_DISABLE_FP8=TRUE \
LITE_ATTENTION_DISABLE_SM80=TRUE \
LITE_ATTENTION_DISABLE_SOFTCAP=TRUE \
LITE_ATTENTION_DISABLE_CLUSTER=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
LITE_ATTENTION_DISABLE_PACKGQA=TRUE \
LITE_ATTENTION_DISABLE_PAGEDKV=TRUE \
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --extra dev
```

Full build is very slow — the disable flags above skip unused kernel variants. Build isolation is disabled (`no-build-isolation-package` in pyproject.toml) so the extension links against the venv's PyTorch.

Stale `.so` files in `hopper/` can shadow the installed package — clean before rebuilding: `rm -rf build hopper/*.so`

See `BUILDING.md` for all optional flags, alternative methods (pip, setup.py, two-step uv), and consuming-project setup.

## Running Tests

```bash
uv sync --extra dev --extra vis        # install test + vis deps (first time / after changes)
uv run pytest                          # all tests
uv run pytest hopper/tests/test_debug_capture.py  # single file
uv run pytest -k test_flash_attn_output            # single test by name
```

Tests require a GPU. pytest config is in `pyproject.toml` (`testpaths = ["hopper/tests"]`).

## Remote Development

This project requires a GPU to build and test. Develop locally, rsync to a remote GPU machine:

```bash
rsync -avz --delete --exclude='.venv' --exclude='build/' --exclude='*.so' \
  --exclude='__pycache__' --exclude='.git' --exclude='csrc/cutlass' --exclude='csrc/composable_kernel' \
  ~/code/LiteAttention/ <remote>:~/code/LiteAttention/
```

On the remote, build and test:
```bash
cd ~/code/LiteAttention
MAX_JOBS=$(nproc) NVCC_THREADS=4 \
LITE_ATTENTION_DISABLE_BACKWARD=TRUE \
LITE_ATTENTION_DISABLE_FP16=TRUE \
LITE_ATTENTION_DISABLE_FP8=TRUE \
LITE_ATTENTION_DISABLE_SM80=TRUE \
LITE_ATTENTION_DISABLE_SOFTCAP=TRUE \
LITE_ATTENTION_DISABLE_CLUSTER=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
LITE_ATTENTION_DISABLE_PACKGQA=TRUE \
LITE_ATTENTION_DISABLE_PAGEDKV=TRUE \
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --extra dev --extra vis
uv run pytest
```

## Architecture

### Package layout (`hopper/` → `lite_attention`)

- `lite_attention.py` — Main module. `LiteAttention` (single GPU) and `SeqParallelLiteAttention` (multi-GPU) are `nn.Module` subclasses that wrap flash attention with skip list optimization.
- `calibrated_module/` — Configuration framework. `ConfigurableModule` mixin + `ModuleRegistry` enable per-layer, per-timestep threshold configuration with TOML serialization. `LiteAttentionRegistry` discovers all `LiteAttention` modules in a model and configures them.
- `_internal/flash_attn_interface.py` — Python bindings to the `lite_attention._C` CUDA extension.
- `_internal/cpp/` — CUDA kernels. `flash_api.cpp` registers PyTorch operators. Kernel files are instantiated per head-dim/dtype/feature combination.
- `instantiations/` — Generated `.cu` files (cartesian product of head dims, dtypes, split/paged/softcap variants).
- `tests/` — `test_lite_attention.py` (skip list, quantization, must-do list), `test_flash_attn.py` (upstream flash attention correctness).

### CUDA build system (`setup.py`)

`setup.py` monkey-patches PyTorch's ninja file writer to route `_sm80.cu`, `_sm90.cu`, and `_sm100.cu` files to their respective GPU architecture flags. `SRC_DIR = "hopper"` is the base path for all source file references. Feature flags (`LITE_ATTENTION_DISABLE_*` env vars) control which kernel variants are compiled.

### Upstream relationship

`csrc/cutlass` (NVIDIA CUTLASS, git submodule) provides the CUDA template library. `flash_attn/` contains upstream Flash Attention code. LiteAttention adds the skip list optimization, INT8 quantization support, and the calibration framework on top.

## Debugging

- `LITE_ATTENTION_VERBOSE=TRUE` — enable debug logging
- `LITE_ATTENTION_DEBUG=TRUE` — allow positive thresholds for testing
- `visualize_skips()` method on `LiteAttention` instances creates attention pattern visualizations showing computed vs skipped tiles
