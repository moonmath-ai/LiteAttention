# Troubleshooting

After cloning, initialize the submodules:
```bash
git submodule update --init
```

# Building and Testing

Build the project and create a virtual environment that includes it:
```bash
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --extra dev
```

This single command creates the venv, installs all dependencies (including dev extras like pytest), and builds the CUDA extension.
The `no-build-isolation-package` setting in `pyproject.toml` ensures the extension is compiled against the venv's PyTorch.

`CUDA_HOME` and `CXX` must match the compiler and CUDA version used to build PyTorch.
To target a different CUDA version, update the PyTorch index in `pyproject.toml` (`tool.uv.sources` / `tool.uv.index`) accordingly.

The `--extra dev` is not needed in production.

### Alternative: two-step build

If the single-step method has issues, you can split it into two steps:
```bash
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv sync --no-install-project --extra dev
CUDA_HOME=/usr/local/cuda-12.8 CXX=g++ uv pip install -e . --no-build-isolation
```

The first command installs all dependencies; the second builds and installs the project against the venv's PyTorch.

## Optional Build Flags

Prepend the following environment variables to disable unused features and speed up the build:
```bash
LITE_ATTENTION_DISABLE_SOFTCAP=TRUE \
LITE_ATTENTION_DISABLE_CLUSTER=TRUE \
LITE_ATTENTION_DISABLE_SM80=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
LITE_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
LITE_ATTENTION_DISABLE_PACKGQA=TRUE \
LITE_ATTENTION_DISABLE_PAGEDKV=TRUE \
```

Additionally, disable any of the following if not needed:
```bash
LITE_ATTENTION_DISABLE_FP16=TRUE \
LITE_ATTENTION_DISABLE_FP8=TRUE \
LITE_ATTENTION_DISABLE_BACKWARD=TRUE \
LITE_ATTENTION_DISABLE_HDIM64=TRUE \
LITE_ATTENTION_DISABLE_HDIM96=TRUE \
LITE_ATTENTION_DISABLE_HDIM192=TRUE \
LITE_ATTENTION_DISABLE_HDIM256=TRUE \
```

To control build parallelism for nvcc/ninja, prepend:
```bash
MAX_JOBS=$(nproc) NVCC_THREADS=4 \
```

To display build output, append `-v`.

> **Note:** `CUDA_HOME=... CXX=... uv build --no-build-isolation` should work in theory but is currently broken.

## Running Tests

Run tests with:
```bash
uv run pytest
```

If you built without `--extra dev`, install it first:
```bash
uv run --extra dev pytest
```

# Using LiteAttention as a Dependency in Another Project

LiteAttention is a CUDA extension and must be built against the consuming project's PyTorch to ensure ABI compatibility.

`CUDA_HOME` must point to the CUDA version that matches the consuming project's PyTorch.
All options below support the same optional environment variables described in [Optional Build Flags](#optional-build-flags).

Options 1 and 2 require the build dependencies (`packaging`, `ninja`, `wheel`) to be installed in the target virtual environment beforehand:
```bash
pip install packaging ninja wheel
```

## Option 1: With setup.py (legacy)

From the root of this repo, with the target project's virtual environment activated:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ python setup.py install
```

> **Note:** You must run this from the LiteAttention directory.

## Option 2: With pip

With the target project's virtual environment activated:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ pip install --no-build-isolation /path/to/LiteAttention
```

`--no-build-isolation` ensures the extension is compiled against the venv's PyTorch rather than a potentially different version resolved by pip's build isolation.

## Option 3: With uv (recommended)

This approach uses `no-build-isolation-package` so the build runs inside the project environment, against the exact Python, CUDA, and PyTorch versions already installed.
Because build isolation is disabled, the build dependencies (`setuptools`, `packaging`, `ninja`) must be listed as regular project dependencies so they are available when LiteAttention is built.
Additionally, static dependency metadata must be provided so uv can resolve dependencies without executing LiteAttention's `setup.py` (which imports `torch`).
See the [uv docs on disabling build isolation](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation).

The consuming project's `pyproject.toml` likely already has `torch` and a PyTorch index configured.
Add the LiteAttention-specific parts (marked with `# <-- add` below):
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

# Static metadata so uv can resolve without building:  # <-- add
[[tool.uv.dependency-metadata]]
name = "lite-attention"
version = "0.4.0"
requires-dist = ["torch>=2.2", "einops", "structlog", "tomli-w"]
```

Then sync with the appropriate CUDA environment variables:
```bash
CUDA_HOME=/usr/local/cuda-12.9 CXX=g++ uv sync
```
