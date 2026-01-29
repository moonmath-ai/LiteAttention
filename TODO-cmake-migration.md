# CMake Migration - Open Tasks

## Summary

Migrated LiteAttention build from 682-line setup.py to scikit-build-core + CMake.

**Current status:** Build works, 4m46s (was 3m20s with old setup.py, 29m before optimizations)

**Key changes made:**
- Created `pyproject.toml` and `CMakeLists.txt` at repo root
- Renamed `hopper/` to `lite_attention/`
- Removed 531 instantiation files from git (generated at build time)
- Deleted `hopper/setup.py`
- Added ccache support
- Reduced kernels from 18 to 10 (removed batch files, disabled-by-default variants)
- Added `--threads` flag for nvcc intra-file parallelism

---

## Tasks

### 1. Investigate 1.5 minute gap (3m20s → 4m46s)

Old setup.py built in ~3m20s, new CMake build takes ~4m46s. Possible causes:
- scikit-build-core overhead (sdist → temp dir → wheel)
- Different parallelization
- Variance in measurements

### 2. Speed optimizations

- **2a. Try NVCC_THREADS=4** - currently default is 2, matching old setup.py
- **2b. Try `--split-compile`** - commented out in old setup.py as "faster", worth testing
  ```
  # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
  ```

### 3. Handle warnings

#### 3a. Remove `cmake` from build-system.requires
```
scikit_build_core - WARNING - cmake should not be in build-system.requires - scikit-build-core will inject it as needed
```
**Action:** Remove `cmake>=3.26` from `build-system.requires` in pyproject.toml

#### 3b. kineto_LIBRARY-NOTFOUND
```
CMake Warning: static library kineto_LIBRARY-NOTFOUND not found.
```
**Context:** Kineto is PyTorch's profiling library. This is likely harmless but should verify.

#### 3c. Ninja detection - reproducibility in clean docker?
scikit-build-core auto-detects Ninja if installed. Questions:
- Will it work in a clean docker without Ninja?
- Should we explicitly add `ninja` to build-system.requires?
- Or set `cmake.generator = "Unix Makefiles"` to force Make?

#### 3d. Fix `-std=c++20` redefinition warning
```
nvcc warning : incompatible redefinition for option 'std', the last value of this option was used
```
**Ideas:**
- Remove our `-std=c++20` flag (CMake/PyTorch may already set it)
- Use `CMAKE_CUDA_STANDARD 20` instead of `-std=c++20` flag
- Check what PyTorch's cmake sets and avoid conflict

#### 3e. libnvrtc.so shorthash warning
```
CMake Warning: Failed to compute shorthash for libnvrtc.so
```
**Context:** From PyTorch's cmake. Investigate if this affects anything.

#### 3f. ptxas C7512 performance warning (hdim256 int8)
```
ptxas info : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources
```
**Context:** This appears for hdim256 int8 kernels. High register pressure. Likely expected behavior but should verify performance is acceptable.

### 4. Verify compilation works

#### 4a. numpy warning - add to optional deps?
```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```
**Context:** PyTorch warning when numpy not installed. Options:
- Add numpy to dependencies (increases install size)
- Add to optional-dependencies
- Ignore (it's just a warning)

#### 4b. Run pytest
```bash
cd lite_attention/tests
pip install einops pytest
pytest test_flash_attn.py::test_lite_attn_output -v
```

### 5. ccache - added, not tested

Added to CMakeLists.txt:
```cmake
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
```
**TODO:** Install ccache on remote and verify rebuild is fast

### 6. cibuildwheel CI

#### 6a. Wheel matrix: python/torch/cuda combos
Like flash_attn, we may need pre-built wheels for:
- Python: 3.10, 3.11, 3.12
- Torch: 2.5, 2.6, 2.7
- CUDA: 12.8, 12.9
- CXX11 ABI: TRUE/FALSE

#### 6b. Test with other torch versions
Currently only tested with torch installed by uv. Should test compatibility with different torch versions.

### 7. Update README (CUDA 12.8+)

Document new requirement: CUDA 12.8+ (was 12.3+). This eliminated 107 lines of nvcc/ptxas download logic.

### 8. Commit / PR

Once tasks are complete:
- Review all changes
- Create commit with descriptive message
- Open PR

### 9. Check if we need cuDNN/cuSPARSELt/cuDSS/cuFile

Build showed these warnings:
```
-- USE_CUDNN is set to 0. Compiling without cuDNN support
-- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
-- USE_CUDSS is set to 0. Compiling without cuDSS support
-- USE_CUFILE is set to 0. Compiling without cuFile support
```
**TODO:** Verify LiteAttention doesn't need any of these features.

### 10. CMAKE_CUDA_ARCHITECTURES ignored by PyTorch

```
CMake Warning: pytorch is not compatible with `CMAKE_CUDA_ARCHITECTURES` and will ignore its value. Please configure `TORCH_CUDA_ARCH_LIST` instead.
```
**Context:** We set `CMAKE_CUDA_ARCHITECTURES "90a"` but PyTorch ignores it.
**Options:**
- Set `TORCH_CUDA_ARCH_LIST` environment variable instead
- Or just ignore since PyTorch auto-detects (showed "Autodetected CUDA architecture(s): 9.0")

---

## Files Changed

| File | Status |
|------|--------|
| `pyproject.toml` | Created (repo root) |
| `CMakeLists.txt` | Created (repo root) |
| `lite_attention/` | Renamed from `hopper/` |
| `lite_attention/.gitignore` | Created |
| `hopper/setup.py` | Deleted |
| `hopper/instantiations/*.cu` | Removed from git (531 files) |

---

## Build Commands

```bash
# On remote H100
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

# Clean build
rm -rf build .venv dist
uv build

# With higher parallelism
NVCC_THREADS=4 uv build

# Install and test
uv pip install numpy einops pytest
uv run pytest lite_attention/tests/test_flash_attn.py::test_lite_attn_output -v
```

---

## References

- [scikit-build-core docs](https://scikit-build-core.readthedocs.io/)
- [cibuildwheel docs](https://cibuildwheel.pypa.io/)
- Old plan: `~/.claude/plans/sleepy-prancing-planet.md`
