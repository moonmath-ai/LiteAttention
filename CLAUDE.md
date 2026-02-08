# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteAttention is a temporal sparse attention mechanism for video diffusion models built on top of FlashAttention3. It exploits temporal coherence of sparsity patterns across denoising timesteps to achieve significant speedups (up to 1.89x) with minimal quality degradation.

Key concepts:
- **Skip Lists**: Data structures tracking which attention tiles can be skipped during computation
- **Double Buffering**: Alternates between read/write buffers each forward pass
- **Threshold-based Skipping**: Tiles with max(log-attention-score) below threshold are skipped
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
cd hopper
pip install .
# Or with limited parallelism if low on RAM:
MAX_JOBS=4 pip install .
```

### Build FlashAttention2 (base library, optional)
```bash
pip install .  # from repo root
```

## Running Tests

Tests for LiteAttention are in `hopper/tests/`:
```bash
cd hopper/tests
pytest test_flash_attn.py  # Main attention tests
pytest test_flash_attn.py::test_lite_attn_output  # Run specific test
pytest test_flash_attn.py -k "seqlen_q=1024"  # Filter by parameter
```

Tests use pytest with many parameterized configurations (dtype, sequence lengths, head dimensions, etc.).

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
