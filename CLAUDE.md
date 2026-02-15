# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteAttention is a temporal sparse attention mechanism for video diffusion models built on top of FlashAttention3. It exploits temporal coherence of sparsity patterns across denoising timesteps to achieve significant speedups (up to 1.89x) with minimal quality degradation.

Key concepts:
- **Skip Lists**: Data structures tracking which attention tiles can be skipped during computation
- **Double Buffering**: Alternates between read/write buffers each forward pass
- **Threshold-based Skipping**: Tiles are skipped when their max score is too far below the running max (compared in log2 scale against the threshold)
- **Must-Do/Must-Skip Lists**: Force computation or skipping of specific sequence ranges

## Build & Installation

### Requirements
- H100 / H200 GPU
- CUDA >= 12.8 with CUDA toolkit
- C++ 20
- PyTorch 2.2+
- Linux only
- Python dependencies: `structlog`, `tomli-w` (for calibration config serialization)

### Build LiteAttention (the main package)
```bash
# With uv (recommended):
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --group dev

# Or with pip:
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ pip install .
# With limited parallelism if low on RAM:
MAX_JOBS=4 CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ pip install .
```

Note: `CUDA_HOME` must point to a CUDA 12.8 toolkit matching the PyTorch cu128 build.
The PyTorch index is configured in `pyproject.toml` (`tool.uv.sources` / `tool.uv.index`).

### Using LiteAttention as a dependency in another project

CUDA extensions must be built against the target venv's torch for ABI
compatibility. Use `no-build-isolation-package` so the build runs in the
project environment, and include build deps (`setuptools`, `packaging`, `ninja`)
as project dependencies so they are installed before lite-attention is built.

```toml
# pyproject.toml of the consuming project
[project]
dependencies = [
    "torch>=2.2",
    "setuptools>=64",
    "packaging",
    "ninja",
    "lite-attention",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv]
no-build-isolation-package = ["lite-attention"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
lite-attention = { path = "../LiteAttention" }  # or a git URL
```

```bash
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync
```

Key points:
- `no-build-isolation-package`: builds lite-attention in the target venv (same torch at build and runtime)
- `setuptools`, `packaging`, `ninja` in project deps: uv installs them first, before building lite-attention
- `CUDA_HOME` must match the PyTorch CUDA version (cu128 -> cuda-12.8)

## Running Tests

### LiteAttention integration tests (root)
```bash
pytest test_lite_attention.py                          # all tests
pytest test_lite_attention.py::test_skip_all            # single test
pytest test_lite_attention.py -k "d128 and bf16"        # filter by head_dim / dtype
```

`test_must_do_list.py` need to be converted to pytest. meanwhile it is run with python

### FlashAttention kernel tests (hopper/tests/)
```bash
cd hopper/tests
pytest test_flash_attn.py
pytest test_flash_attn.py::test_lite_attn_output
pytest test_flash_attn.py -k "seqlen_q=1024"
```

## Remote Development (Nebius GPU machines)

The code require a GPU. Use rsync from the local machine instead to a remote server (ask which server).

```bash
# Sync code to remote (include .git for submodules)
rsync -az --exclude .venv --exclude __pycache__ --exclude build \
  ~/code/LiteAttention/ <remote>:~/code/LiteAttention/

# On remote: build and install everything with uv sync
cd ~/code/LiteAttention
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --group dev

# Run tests
.venv/bin/pytest test_lite_attention.py
```

## Code Architecture

### Core Components

**`hopper/` - LiteAttention Package (main focus)**
- `lite_attention.py` - Main `LiteAttention`, `SeqParallelLiteAttention`, `LiteAttentionRegistry`, and config classes
- `calibrated_module.py` - Generic calibration framework (`ConfigurableModule`, `ModuleRegistry`, config base classes)
- `_internal/flash_attn_interface.py` - Python bindings to CUDA kernels
- `_internal/cpp/` - CUDA/C++ kernel implementations
- `instantiations/` - Generated kernel instantiations for different configurations

**`flash_attn/` - FlashAttention2 Base**
- `flash_attn_interface.py` - FlashAttention2 Python interface
- `models/` - Pre-built model implementations (GPT, LLaMA, BERT, etc.)
- `modules/` - Neural network modules (MHA, MLP, Block)
- `ops/` - Fused operations (layer norm, fused dense)

**`csrc/` - CUDA Source**
- `flash_attn/` - FlashAttention2 CUDA kernels (SM80)
- `cutlass/` - CUTLASS library (submodule)

### Key Classes

**`LiteAttention`** (`hopper/lite_attention.py`)
- Main attention class with skip list optimization, inherits from `nn.Module` and `ConfigurableModule`
- Manages double-buffered skip lists internally
- Key methods: `forward()`, `reset_skip_state()`, `set_threshold()`, `enable_skip_optimization()`
- Tile sizes determined by `get_MN()` - must stay synchronized with C++ `tile_size.h`
- Threshold is managed via the config system (`LiteAttentionRunConfig`), though `set_threshold()` and `threshold=` constructor arg still work

**`SeqParallelLiteAttention`** (`hopper/lite_attention.py`)
- Wrapper for multi-GPU sequence parallelism
- Manages separate `LiteAttention` instances per node

**`LiteAttentionRegistry`** (`hopper/lite_attention.py`)
- Subclass of `ModuleRegistry` with `from_model()` classmethod
- Creates a registry from a model and configures all its `LiteAttention` modules
- Three modes: `"const"` (fixed threshold), `"load"` (from TOML file), `"calib"` (binary-search calibration)
- Call `save_if_calib()` after inference to persist calibration results

### Calibration Framework (`hopper/calibrated_module.py`)

A generic configuration and calibration system for PyTorch modules:

- **`ConfigurableModule`** - Mixin that adds per-timestep config resolution to any `nn.Module`. Config resolution order: instance config > registry config > default config
- **`ModuleRegistry`** - Central registry managing configs across all `ConfigurableModule` instances in a model. Supports bulk/per-module config, TOML load/save
- **`CalibratedRunConfig`** / **`CalibratedCalibConfig`** - Base dataclasses for runtime vs calibration configs
- **`ConfigList`** - List of configs (one per timestep) with `collect()`/`explode()` for TOML serialization
- **`CalibratedConfigDict`** - Dict mapping module names to configs, with `.load()` and `.save()` for TOML files

LiteAttention-specific configs:
- **`LiteAttentionRunConfig`** - holds `threshold: float` (default: -10.0)
- **`LiteAttentionCalibConfig`** - holds `metric` ("Cossim"/"L1"/"RMSE") and `target_error` for calibration

Typical calibration workflow:
```python
registry = LiteAttentionRegistry.from_model(
    model, mode="calib", filename="calibrated.toml",
    calib_config={"target_error": 0.01, "metric": "L1"},
)
# run inference...
registry.save_if_calib()

# later, load calibrated thresholds:
registry = LiteAttentionRegistry.from_model(
    model, mode="load", filename="calibrated.toml",
)
```

## Important Implementation Details

### Tile Size Synchronization
The Python `LiteAttention.get_MN()` method must mirror the C++ `tile_size_fwd_sm90()` in `tile_size.h`. If kernel tile sizes change, update both locations.

### Skip List Format
Shape: `[2, batch, heads, qtiles, ktiles + 1]`
- Dimension 0: Double buffer (read/write alternation)
- Last dimension: `ktiles + 1` where +1 stores list length
- Format depends on `reverse_skip_list` flag (see docstrings in `lite_attention.py`)

### Environment Variables
- `LITE_ATTENTION_VERBOSE` - Enable debug logging
- `LITE_ATTENTION_DEBUG` - Allow positive thresholds for testing
- `FLASH_ATTENTION_DISABLE_*` - Feature flags for build configuration

## Code Style

- Line length: 100 characters
- Python version: 3.9+ (target-version in pyproject.toml)
- Formatting: black, ruff

## Debugging

Set `LITE_ATTENTION_VERBOSE` to anything other than "FALSE" for debug logs. Use `visualize_skips()` method to create attention pattern visualizations showing which tiles are computed vs skipped.
