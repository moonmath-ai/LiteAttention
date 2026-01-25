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
- `lite_attention.py` - Main `LiteAttention` and `SeqParallelLiteAttention` classes
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
- Main attention class with skip list optimization
- Manages double-buffered skip lists internally
- Key methods: `__call__()`, `reset_skip_state()`, `set_threshold()`, `enable_skip_optimization()`
- Tile sizes determined by `get_MN()` - must stay synchronized with C++ `tile_size.h`

**`SeqParallelLiteAttention`** (`hopper/lite_attention.py`)
- Wrapper for multi-GPU sequence parallelism
- Manages separate `LiteAttention` instances per node

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
