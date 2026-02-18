# Troubleshooting

After cloning, initialize the submodules:
```bash
git submodule update --init
```

# Building and Testing

Build the project and create a virtual environment that includes it:
```bash
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync
```

`CUDA_HOME` and `CXX` must match the compiler and CUDA version used to build PyTorch. To target a different CUDA version, update the PyTorch index in `pyproject.toml` (`tool.uv.sources` / `tool.uv.index`) accordingly.

## Optional Build Flags

Prepend any of the following environment variables to disable specific build components:
```bash
FLASH_ATTENTION_DISABLE_SM80=TRUE \
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE \
FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
```

To control build parallelism for nvcc/ninja, prepend:
```bash
MAX_JOBS=$(nproc) NVCC_THREADS=4 \
```

To display build output, append `-v`.

## Running Tests

After building, use `uv run pytest` to run tests inside the virtual environment that includes the compiled LiteAttention module.

> **Note:** `CUDA_HOME=... CXX=... uv build --no-build-isolation` should work in theory but is currently broken.

# Using LiteAttention as a Dependency in Another Project

LiteAttention is a CUDA extension and must be built against the consuming project's PyTorch to ensure ABI compatibility.

for `CUDA_HOME`, use the directory that match the version used to compile PyTorch for that project

All options below support the same optional environment variables described in [Optional Build Flags](#optional-build-flags).

## Option 1: With setup.py (legacy)

From the root of this repo, with the target project's virtual environment activated:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ python setup.py install
```

## Option 2: With pip

From the root of this repo, with the target project's virtual environment activated:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ pip install .
```

It may also be possible to install from an arbitrary directory:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ pip install /path/to/LiteAttention
```

## Option 3: With uv (recommended)

This approach uses `no-build-isolation-package` so the build runs inside the project environment, against the exact Python, CUDA, and PyTorch versions already installed. Because build isolation is disabled, the build dependencies (`setuptools`, `packaging`, `ninja`) must be listed as regular project dependencies so they are available when LiteAttention is built. See the [uv docs on disabling build isolation](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation).

The consuming project's `pyproject.toml` likely already has `torch` and a PyTorch index configured. Add the LiteAttention-specific parts (marked with `# <-- add` below):
```toml
[project]
dependencies = [
    # existing project dependencies ...
    "torch>=2.2",
    "lite-attention",                        # <-- add
    "setuptools>=64",                        # <-- add (build dep)
    "packaging",                             # <-- add (build dep)
    "ninja",                                 # <-- add (build dep)
]

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true

[tool.uv]
no-build-isolation-package = ["lite-attention"]  # <-- add

[tool.uv.sources]
torch = { index = "pytorch-cu129" }
# Pick one of the following:                     # <-- add
lite-attention = { path = "../LiteAttention" }
# lite-attention = { path = "../LiteAttention", editable = true }
# lite-attention = { git = "https://github.com/moonmath-ai/LiteAttention" }
```

Then sync with the appropriate CUDA environment variables:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ uv sync
```
