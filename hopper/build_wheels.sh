#!/bin/bash
# Script to automate LiteAttention wheel creation for different Python versions and architectures.
# Usage: ./build_wheels.sh [python_versions...]
# Example: ./build_wheels.sh 3.10 3.11 3.12

set -e

# Argument parsing for python versions and torch version
PYTHON_VERSIONS=()
TORCH_VERSION_REQ="latest"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --python)
            shift
            if [[ "$1" == *","* ]]; then
                # Comma-separated list
                IFS="," read -ra PYTHON_VERSIONS <<< "$1"
            else
                # Space-separated, accumulate until next flag or end-of-args
                while [[ $# -gt 0 && "$1" != --* ]]; do
                    PYTHON_VERSIONS+=("$1")
                    shift
                done
                continue  # Skip incrementing $# at bottom of loop, as we already advanced
            fi
            ;;
        --torch)
            shift
            TORCH_VERSION_REQ="$1"
            ;;
        *)
            PYTHON_VERSIONS+=("$1")
            ;;
    esac
    shift
done

if [ ${#PYTHON_VERSIONS[@]} -eq 0 ]; then
    PYTHON_VERSIONS=("3.10" "3.12")
fi

WHEELS_OUTPUT_DIR="$(pwd)/dist_wheels"
mkdir -p "$WHEELS_OUTPUT_DIR"
mkdir -p "$(pwd)/build_lite_attention"

    echo "Building wheels for Python versions: ${PYTHON_VERSIONS[*]}"
    echo "Requested Torch version: $TORCH_VERSION_REQ"
    echo "Base build directory: $(pwd)/build_lite_attention"
    echo "Output directory: $WHEELS_OUTPUT_DIR"

# Ensure clean state in the current directory
rm -rf build/ dist/ *.egg-info

build_version() {
    local py_ver=$1
    local env_name="env_py${py_ver//./}"
    local env_path="$BASE_BUILD_DIR/$env_name"

    echo ""
    echo "================================================================================"
    echo " BUILDING FOR PYTHON $py_ver"
    echo "================================================================================"
    
    # Create or update conda environment
    if [ ! -d "$env_path" ]; then
        echo "Creating conda environment in $env_path..."
        conda create -p "$env_path" python="$py_ver" -y
    fi

    echo "Installing/Updating dependencies..."
    if [ "$TORCH_VERSION_REQ" == "latest" ]; then
        conda run -p "$env_path" pip install torch einops packaging ninja wheel build
    else
        conda run -p "$env_path" pip install torch=="$TORCH_VERSION_REQ" einops packaging ninja wheel build
    fi

    # Capture installed torch and cuda versions for the wheel filename
    # We strip the '+cuXXX' suffix from torch version to keep the version tag clean
    INSTALLED_TORCH_VER=$(conda run -p "$env_path" python -c "import torch; print(torch.__version__.split('+')[0])")
    INSTALLED_CUDA_VER=$(conda run -p "$env_path" python -c "import torch; print(torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu')")
    
    # This environment variable is read by setup.py:get_package_version()
    export LITE_ATTENTION_LOCAL_VERSION="cu${INSTALLED_CUDA_VER}torch${INSTALLED_TORCH_VER}"

    # Build settings
    export LITE_ATTENTION_FORCE_BUILD=TRUE
    # export LITE_ATTENTION_DISABLE_SM80=FALSE # Uncomment to enable SM80 (Ampere) support
    
    echo "Starting build..."
    # We clean build/ between versions to avoid cache issues
    rm -rf build/
    
    # Using 'build' package (PEP 517) instead of calling setup.py directly.
    # --no-isolation is used because we've already installed dependencies in the conda env.
    conda run -p "$env_path" python -m build --wheel --no-isolation --outdir dist/

    echo "Build complete. Moving wheel..."
    ls dist/*.whl
    cp dist/*.whl "$WHEELS_OUTPUT_DIR/"
    
    # Clean up dist/ to avoid picking up old wheels in next iteration
    rm -rf dist/
}

# Run build for each version
for ver in "${PYTHON_VERSIONS[@]}"; do
    build_version "$ver"
done

echo ""
echo "================================================================================"
echo " ALL BUILDS COMPLETE"
echo "================================================================================"
ls -lh "$WHEELS_OUTPUT_DIR"
