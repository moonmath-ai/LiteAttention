from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUTLASS path from environment or use common libs folder
cutlass_path = os.environ.get('CUTLASS_PATH',
                               os.path.expanduser('~/libs/cutlass'))
cutlass_path = os.path.expanduser(cutlass_path)

# Normalize the path - handle case where CUTLASS_PATH might point to include directory
cutlass_include = None
if os.path.isdir(os.path.join(cutlass_path, 'include')):
    # Standard case: CUTLASS_PATH points to root, include is at CUTLASS_PATH/include
    cutlass_include = os.path.join(cutlass_path, 'include')
elif os.path.isdir(cutlass_path) and os.path.isdir(os.path.join(cutlass_path, 'cute')):
    # CUTLASS_PATH already points to include directory
    cutlass_include = cutlass_path
else:
    # Try alternative location
    alt_path = os.path.join(os.path.dirname(__file__), 'third_party', 'cutlass', 'include')
    if os.path.isdir(alt_path):
        cutlass_include = alt_path
    else:
        raise RuntimeError(
            f"CUTLASS not found. Please set CUTLASS_PATH environment variable or install CUTLASS.\n"
            f"Tried: {cutlass_path}, {alt_path}"
        )

# Verify required headers exist
cute_tensor = os.path.join(cutlass_include, 'cute', 'tensor.hpp')
cutlass_numeric = os.path.join(cutlass_include, 'cutlass', 'numeric_types.h')
if not os.path.exists(cute_tensor):
    raise RuntimeError(f"CUTLASS CuTe not found. Expected: {cute_tensor}")
if not os.path.exists(cutlass_numeric):
    raise RuntimeError(f"CUTLASS headers not found. Expected: {cutlass_numeric}")

# Build include directories
include_dirs = [cutlass_include]

# Add tools/util/include if it exists
# Resolve symlinks to get the real path
cutlass_include_real = os.path.realpath(cutlass_include)
cutlass_root = os.path.dirname(cutlass_include_real)
tools_util_include = os.path.join(cutlass_root, 'tools', 'util', 'include')
if os.path.isdir(tools_util_include):
    include_dirs.append(tools_util_include)
else:
    # Try alternative: if cutlass_include was a symlink, try from the symlink's parent
    cutlass_include_parent = os.path.dirname(cutlass_include)
    tools_util_include_alt = os.path.join(cutlass_include_parent, 'tools', 'util', 'include')
    if os.path.isdir(tools_util_include_alt):
        include_dirs.append(tools_util_include_alt)

# Toggle device debug (cuda-gdb) vs profiler build
device_debug = os.environ.get('DEVICE_DEBUG', '0') == '1'

nvcc_flags = [
    '-O3',
    '-g',              # host/device symbols
    '-std=c++17',
    '-lineinfo',       # keep for profiler line mapping
    '--use_fast_math', # ok for profiling; remove if it affects accuracy
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-Xcompiler=-fPIC',
    '-Xcompiler=-Wno-float-conversion',
    '-Xcompiler=-fno-strict-aliasing',
    '-gencode=arch=compute_90,code=sm_90',
]

if device_debug:
    # For cuda-gdb: use -G, and remove -lineinfo (ptxas ignores it with -G)
    nvcc_flags = [f for f in nvcc_flags if f != '-lineinfo']
    nvcc_flags.append('-G')

# Path to the shared quant.cu in hopper/_internal/cpp
quant_cu_path = os.path.join(os.path.dirname(__file__), '..', 'hopper', '_internal', 'cpp', 'quant.cu')
quant_cu_path = os.path.abspath(quant_cu_path)

if not os.path.exists(quant_cu_path):
    raise RuntimeError(f"quant.cu not found at {quant_cu_path}")

setup(
    name='quant_tma',
    ext_modules=[
        CUDAExtension(
            name='quant_tma',
            sources=[
                quant_cu_path,
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-g'],
                'nvcc': nvcc_flags + ['-DQUANT_STANDALONE'],
            },
            extra_link_args=['-lcuda', '-lcudart']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch','numpy'],
)
