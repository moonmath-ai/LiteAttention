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
- Reduced kernels from 18 to 14 (removed batch files; hdim64_256/512 and hdim192_128 required by linker)
- Added `--threads` flag for nvcc intra-file parallelism

---

## Tasks
Once a task is complete:
- Review all changes
- Create commit with descriptive message


### 1. Investigate 1.5 minute gap (3m20s → 4m46s)

Old setup.py built in ~3m20s, new CMake build takes ~4m46s. Undertand why.
Possible causes:
- scikit-build-core overhead (sdist → temp dir → wheel)
- Different parallelization
- Variance in measurements
(look for more ideas)

### 2. Test build with minimal environment variables ✅

**Tested.** Results:
- `CUDA_HOME` - **NOT required**. CMake finds CUDA toolkit from nvcc location.
- `nvcc` in PATH - **Required**. Build fails without it.

Only requirement: `export PATH=/usr/local/cuda-12.8/bin:$PATH`

### 3. Speed optimizations

Any way to improve the build time is welcome.

Tried, not helping:
- **3a. Try NVCC_THREADS=4** - currently default is 2, matching old setup.py
- **3b. Try `--split-compile`** - commented out in old setup.py as "faster", worth testing
  ```
  # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
  ```
Ideas to try:
- (look for more ideas)

### 4. Handle all build warnings

Build and look for warnings. Address warnings chronologically.

Some warnings we saw:

#### 4a. Remove `cmake` from build-system.requires ✅ Fixed
```
scikit_build_core - WARNING - cmake should not be in build-system.requires - scikit-build-core will inject it as needed
```
**Fixed:** Removed `cmake>=3.26` from `build-system.requires` in pyproject.toml. Committed in 693c791.

#### 4b. kineto_LIBRARY-NOTFOUND ✅ Known issue, ignore
```
CMake Warning: static library kineto_LIBRARY-NOTFOUND not found.
```
**Status:** PyTorch warning. Kineto is PyTorch's profiling library, not needed for LiteAttention.

#### 4c. Ninja detection - reproducibility in clean docker?
scikit-build-core auto-detects Ninja, even tought we didn't install it (and it is not installed on the server)
If this is the intended work, and ninja will be detected on all macines, it is good.
- Will it work in a clean docker without Ninja?
If not, we want to make sure that ninja is available on all machines.
- Should we explicitly add `ninja` to build-system.requires?
Or that the build is the same.
- Or set `cmake.generator = "Unix Makefiles"` to force Make?
Altough it's a very bad solution - we will have very long build times

#### 4d. Fix `-std=c++20` redefinition warning ✅ Fixed
```
nvcc warning : incompatible redefinition for option 'std', the last value of this option was used
```
**Fixed:** Use `CMAKE_CUDA_STANDARD 20` instead of `-std=c++20` flag. Committed in 166559d.

#### 4e. libnvrtc.so shorthash warning ✅ Known issue, ignore
```
CMake Warning: Failed to compute shorthash for libnvrtc.so
```
**Status:** Known PyTorch bug, harmless. When hash computation fails, PyTorch just sets
`CUDA_NVRTC_SHORTHASH="XXXXXXXX"` as placeholder. Build continues normally.

**References:**
- https://github.com/pytorch/pytorch/issues/129777
- https://github.com/pytorch/pytorch/issues/53350

#### 4f. ptxas C7512 performance warning (hdim256 bf16)
```
ptxas info : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources
```
**Context:** This appears for hdim256 bf16 kernels (24 instances). High register pressure.

Investigate.
Those kernels are hand written for performence. Unless it's a known and expected warning, we should find out how to fix it.
In any case, we should verify performance is acceptable.

#### 4g. Check if we need cuDNN/cuSPARSELt/cuDSS/cuFile

Build showed these warnings:
```
-- USE_CUDNN is set to 0. Compiling without cuDNN support
-- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
-- USE_CUDSS is set to 0. Compiling without cuDSS support
-- USE_CUFILE is set to 0. Compiling without cuFile support
```
**TODO:** Investigate / Verify LiteAttention doesn't need any of these features.

#### 4h. CMAKE_CUDA_ARCHITECTURES ignored by PyTorch ✅ Known issue, ignore
```
CMake Warning: pytorch is not compatible with `CMAKE_CUDA_ARCHITECTURES` and will ignore its value. Please configure `TORCH_CUDA_ARCH_LIST` instead.
```
**Status:** PyTorch ignores this setting and auto-detects instead. We removed our global
`CMAKE_CUDA_ARCHITECTURES` setting since we use per-file `-gencode` flags anyway (more precise).
PyTorch auto-detects correctly: "Autodetected CUDA architecture(s): 9.0"


### 5. Verify compilation works

#### 5a. numpy warning - add to optional deps?
```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```
**Context:** PyTorch warning when numpy not installed. Options:
- Add numpy to dependencies (increases install size)
- Add to optional-dependencies
- Ignore (it's just a warning)

#### 5b. Run pytest ✅ Done
```bash
cd lite_attention/tests
export PYTHONPATH=../utils:../_internal:$PYTHONPATH
pip install einops pytest
pytest test_flash_attn.py::test_lite_attn_output -v
```
**Result:** 730 passed, 470 skipped (skipped due to disabled kernel variants)

#### 5c. Run in WAN2.2
WAN can use this as a requirement, we can generate a video and see that it works, and the speed.
This is the practical usage, hence an important test.

#### 5d. Fix test imports - remove PYTHONPATH requirement
Currently tests require:
```bash
cd lite_attention/tests
export PYTHONPATH=../utils:../_internal:$PYTHONPATH
pytest ...
```
**Goal:** Restructure tests/package so `pytest lite_attention/tests/` works from repo root without env vars.
Options:
- Move `padding.py` and `test_util.py` into tests/ or make them proper package imports
- Add conftest.py that sets up sys.path
- Use relative imports within the package

#### 5e. Test with all kernel variants enabled
Current build disables some features (FP16, FP8, softcap, local, etc.) resulting in ~470 skipped tests.
**Goal:** Run full test suite with all variants to verify complete functionality.
Note: This will increase build time significantly.

#### 5f. Add test dependencies to pyproject.toml
Currently need `uv pip install einops pytest` before running tests.
**Goal:** Add test dependencies to `[project.optional-dependencies]` so `uv sync --extra test` works.
```toml
[project.optional-dependencies]
test = ["pytest", "einops"]
```

### 6. ccache - added, not tested

ccache might improve compile times.
Added to CMakeLists.txt:
```cmake
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
```
**TODO:** Install ccache on remote and verify rebuild is fast
NOTE: only after all build warnings go away. We don't want caching to hide issues.

### 7. Build (and test) for more platforms

We want to verify that code is working for various versions.
We can skip a version if it is
- uncommon to use torch/attention in that version
- not supported by our dependencies, maybe because it's too recent
- too old, we require a newer version for some features
Document which versions are supported and which are not (and why)

- Python: 3.10, 3.11, 3.12, 3.13, 3.14
- Torch: currently 2.10; how old can we support? README says 2.2; flash-attn compiles for 2.4-2.9;
- CUDA: 12.8, 12.9, 13.0, 13.1
- CXX11 ABI: TRUE/FALSE : what is it? is it important to support both?

### 8. Update README

Document new stuff, including:
- requirement: CUDA 12.8+ (was 12.3+). This eliminated 107 lines of nvcc/ptxas download logic.
- new location
- how to compile (can be in a separate file

Also update CLAUDE.md

### 9. cibuildwheel CI

Auto building wheels will improve the user ease of install.
NOTE: after the build is working, fast, stable, tested.

Like flash_attn, we may need pre-built wheels for Python/Torch/CUDA/CXX11ABI versions.
We might not - way are those resulting in different wheels?
Investigate for flash-attn to understand.

---

## Files Changed

| File | Status |
|------|--------|
| `pyproject.toml` | Created (repo root) |
| `CMakeLists.txt` | Created (repo root) |
| `lite_attention/` | Renamed from `hopper/` |
| `lite_attention/.gitignore` | Created |
| `hopper/setup.py` | Deleted |
| `hopper/instantiations/*.cu` | Removed from git (531 files) (auto-generated with generate_kernels.py |

---

## Build Commands

```bash
# On remote H100 (only nvcc in PATH required, CUDA_HOME not needed)
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Clean build
rm -rf build .venv dist
uv build

# With higher parallelism
NVCC_THREADS=4 uv build

# Install and test (see 5d for PYTHONPATH requirement)
uv pip install einops pytest
cd lite_attention/tests
PYTHONPATH=../utils:../_internal uv run pytest test_flash_attn.py -v
```

---

## References

- [scikit-build-core docs](https://scikit-build-core.readthedocs.io/)
- [cibuildwheel docs](https://cibuildwheel.pypa.io/)
- Old plan: `~/.claude/plans/sleepy-prancing-planet.md`
