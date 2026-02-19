# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

LiteAttention is a CUDA extension built on top of Flash Attention 3 that adds temporal sparse attention for video diffusion models. It skips redundant attention tiles across diffusion timesteps. The source code lives in `hopper/` but installs as the `lite_attention` Python package (mapped via `pyproject.toml`'s `[tool.setuptools.package-dir]`).

## Build Commands

Requires an H100/H200 GPU, CUDA >= 12.3, and a C++20 compiler.

```bash
git submodule update --init
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --extra dev
```

This creates a venv, installs all deps, and compiles the CUDA extension. Build isolation is disabled (`no-build-isolation-package` in pyproject.toml) so the extension links against the venv's PyTorch.

See `BUILDING.md` for alternative methods (pip, setup.py, two-step uv) and optional build flags to disable unused features and speed up compilation.

## Running Tests

```bash
uv run pytest                          # all tests
uv run pytest hopper/tests/test_lite_attention.py  # single file
uv run pytest -k test_flash_attn_output  # single test by name
```

Tests require a GPU. pytest config is in `pyproject.toml` (`testpaths = ["hopper/tests"]`).

## Architecture

### Package layout (`hopper/` → `lite_attention`)

- `lite_attention.py` — Main module. `LiteAttention` (single GPU) and `SeqParallelLiteAttention` (multi-GPU) are `nn.Module` subclasses that wrap flash attention with skip list optimization.
- `calibrated_module/` — Configuration framework. `ConfigurableModule` mixin + `ModuleRegistry` enable per-layer, per-timestep threshold configuration with TOML serialization. `LiteAttentionRegistry` discovers all `LiteAttention` modules in a model and configures them.
- `_internal/flash_attn_interface.py` — Python bindings to the `lite_attention._C` CUDA extension.
- `_internal/cpp/` — CUDA kernels. `flash_api.cpp` registers PyTorch operators. Kernel files are instantiated per head-dim/dtype/feature combination.
- `instantiations/` — Generated `.cu` files (cartesian product of head dims, dtypes, split/paged/softcap variants).
- `tests/` — `test_lite_attention.py` (skip list, quantization, must-do list), `test_flash_attn.py` (upstream flash attention correctness).

### CUDA build system (`setup.py`)

`setup.py` monkey-patches PyTorch's ninja file writer to route `_sm80.cu`, `_sm90.cu`, and `_sm100.cu` files to their respective GPU architecture flags. `SRC_DIR = "hopper"` is the base path for all source file references. Feature flags (`FLASH_ATTENTION_DISABLE_*` env vars) control which kernel variants are compiled.

### Upstream relationship

`csrc/cutlass` (NVIDIA CUTLASS, git submodule) provides the CUDA template library. `flash_attn/` contains upstream Flash Attention code. LiteAttention adds the skip list optimization, INT8 quantization support, and the calibration framework on top.
