from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUTLASS path from environment or use common libs folder
cutlass_path = os.environ.get('CUTLASS_PATH',
                               os.path.expanduser('~/libs/cutlass'))

setup(
    name='quant_tma',
    ext_modules=[
        CUDAExtension(
            name='quant_tma',
            sources=[
                'quant.cu',
                'quant_python.cu',
            ],
            include_dirs=[
                f'{cutlass_path}/include',
                f'{cutlass_path}/tools/util/include',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-Xcompiler=-fPIC',
                    '-Xcompiler=-Wno-float-conversion',
                    '-Xcompiler=-fno-strict-aliasing',
                    # Target modern architectures
                    '-gencode=arch=compute_80,code=sm_80',  # Ampere
                    '-gencode=arch=compute_89,code=sm_89',  # Ada Lovelace
                    '-gencode=arch=compute_90,code=sm_90',  # Hopper
                ]
            },
            extra_link_args=['-lcuda', '-lcudart']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'numpy',
    ],
)
