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

### 2. Speed optimizations

Any way to improve the build time is welcome.
Some ideas:
- **2a. Try NVCC_THREADS=4** - currently default is 2, matching old setup.py
- **2b. Try `--split-compile`** - commented out in old setup.py as "faster", worth testing
  ```
  # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
  ```
(look for more ideas)

### 3. Handle allt build warnings

Build and look for warnings. Address warnings chronologically.

Some warnings we saw:

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
scikit-build-core auto-detects Ninja, even tought we didn't install it (and it is not installed on the server)
If this is the intended work, and ninja will be detected on all macines, it is good.
- Will it work in a clean docker without Ninja?
If not, we want to make sure that ninja is available on all machines.
- Should we explicitly add `ninja` to build-system.requires?
Or that the build is the same.
- Or set `cmake.generator = "Unix Makefiles"` to force Make?
Altough it's a very bad solution - we will have very long build times

#### 3d. Fix `-std=c++20` redefinition warning
```
nvcc warning : incompatible redefinition for option 'std', the last value of this option was used
```
Are both flags the same? How can we check that?

**Ideas:**
- Remove our `-std=c++20` flag (CMake/PyTorch may already set it)
- Use `CMAKE_CUDA_STANDARD 20` instead of `-std=c++20` flag
- Check what PyTorch's cmake sets and avoid conflict

#### 3e. libnvrtc.so shorthash warning
```
CMake Warning: Failed to compute shorthash for libnvrtc.so
```
**Context:** From PyTorch's cmake.

Investigate.

#### 3f. ptxas C7512 performance warning (hdim256 int8)
```
ptxas info : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources
```
**Context:** This appears for hdim256 int8 kernels. High register pressure.

Investigate.
Those kernels are hand written for performence. Unless it's a known and expected warning, we should find out how to fix it.
In any case, we should verify performance is acceptable.

#### 3g. Check if we need cuDNN/cuSPARSELt/cuDSS/cuFile

Build showed these warnings:
```
-- USE_CUDNN is set to 0. Compiling without cuDNN support
-- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
-- USE_CUDSS is set to 0. Compiling without cuDSS support
-- USE_CUFILE is set to 0. Compiling without cuFile support
```
**TODO:** Investigate / Verify LiteAttention doesn't need any of these features.

#### 3h. CMAKE_CUDA_ARCHITECTURES ignored by PyTorch

```
CMake Warning: pytorch is not compatible with `CMAKE_CUDA_ARCHITECTURES` and will ignore its value. Please configure `TORCH_CUDA_ARCH_LIST` instead.
```
**Context:** We set `CMAKE_CUDA_ARCHITECTURES "90a"` but PyTorch ignores it.
**Options:**
- Investigate
- Set `TORCH_CUDA_ARCH_LIST` environment variable instead
- Or just ignore since PyTorch auto-detects (showed "Autodetected CUDA architecture(s): 9.0")


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

#### 4c. Run in WAN2.2
WAN can use this as a requirement, we can generate a video and see that it works, and the speed.
This is the practical usage, hence an important test.

### 5. ccache - added, not tested

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

### 6. Build (and test) for more platforms

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

### 7. Update README

Document new stuff, including:
- requirement: CUDA 12.8+ (was 12.3+). This eliminated 107 lines of nvcc/ptxas download logic.
- new location
- how to compile (can be in a separate file

Also update CLAUDE.md

### 8. cibuildwheel CI

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
Why `uv pip install` and not `uv sync` ?

---

## References

- [scikit-build-core docs](https://scikit-build-core.readthedocs.io/)
- [cibuildwheel docs](https://cibuildwheel.pypa.io/)
- Old plan: `~/.claude/plans/sleepy-prancing-planet.md`
