#!/usr/bin/env python3
import json
import os
import sys

DEFAULT_OS_VERSION = "devel-ubuntu22.04"
ABI_VALUES = ("FALSE", "TRUE")

CUDA_FAMILY_VERSIONS = {
    "cu124": ("12.4.1",),
    "cu128": ("12.8.1",),
    "cu129": ("12.9.1",),
    "cu130": ("13.0.0", "13.0.1", "13.0.2", "13.1.0", "13.1.1"),
}

SUPPORTED_WHEEL_FAMILIES = {
    "cu124": {
        "2.4.0": ("3.9", "3.10", "3.11", "3.12"),
    },
    "cu128": {
        "2.7.1": ("3.9", "3.10", "3.11", "3.12", "3.13"),
        "2.8.0": ("3.9", "3.10", "3.11", "3.12", "3.13"),
        "2.9.1": ("3.10", "3.11", "3.12", "3.13"),
        "2.10.0": ("3.10", "3.11", "3.12", "3.13"),
    },
    "cu129": {
        "2.8.0": ("3.9", "3.10", "3.11", "3.12", "3.13"),
        "2.9.1": ("3.10", "3.11", "3.12", "3.13"),
        "2.10.0": ("3.10", "3.11", "3.12", "3.13"),
    },
    "cu130": {
        "2.9.1": ("3.10", "3.11", "3.12", "3.13"),
        "2.10.0": ("3.10", "3.11", "3.12", "3.13"),
    },
}


def parse_csv_env(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [part.strip() for part in value.split(",") if part.strip()]


def cuda_family(cuda_version: str) -> str:
    for family, versions in CUDA_FAMILY_VERSIONS.items():
        if cuda_version in versions:
            return family
    raise KeyError(f"Unsupported CUDA version: {cuda_version}")


def is_supported_combo(python_version: str, cuda_version: str, torch_version: str) -> bool:
    family = cuda_family(cuda_version)
    supported_python_versions = SUPPORTED_WHEEL_FAMILIES.get(family, {}).get(torch_version, ())
    return python_version in supported_python_versions


def build_include_entry(
    os_version: str,
    python_version: str,
    cuda_version: str,
    torch_version: str,
    cxx11_abi: str,
) -> dict[str, str]:
    return {
        "os-version": os_version,
        "python-version": python_version,
        "cuda-version": cuda_version,
        "torch-version": torch_version,
        "cxx11_abi": cxx11_abi,
    }


def full_supported_matrix(
    os_version: str = DEFAULT_OS_VERSION,
    abi_values: tuple[str, ...] = ABI_VALUES,
) -> dict:
    include = []
    for family, torch_support in SUPPORTED_WHEEL_FAMILIES.items():
        for cuda_version in CUDA_FAMILY_VERSIONS[family]:
            for torch_version, python_versions in torch_support.items():
                for python_version in python_versions:
                    for cxx11_abi in abi_values:
                        include.append(
                            build_include_entry(
                                os_version=os_version,
                                python_version=python_version,
                                cuda_version=cuda_version,
                                torch_version=torch_version,
                                cxx11_abi=cxx11_abi,
                            )
                        )
    return {"include": include}


def matrix_48() -> dict:
    cuda13 = list(CUDA_FAMILY_VERSIONS["cu130"])
    combos: list[tuple[str, str, str]] = []

    # Legacy Python coverage on CUDA 12.x with the newest compatible torch.
    for cu in ("12.8.1", "12.9.1"):
        combos.append(("3.9", cu, "2.8.0"))

    # Mid-range Python on CUDA 12.x.
    for py in ("3.10", "3.11"):
        for cu in ("12.8.1", "12.9.1"):
            combos.append((py, cu, "2.8.0"))

    # Newer torch line on CUDA 12.x for newest Python.
    for cu in ("12.8.1", "12.9.1"):
        combos.append(("3.12", cu, "2.10.0"))

    # Extra modern CUDA 12.9 combo for Python 3.11.
    combos.append(("3.11", "12.9.1", "2.10.0"))

    # CUDA 13 subversion coverage.
    for py in ("3.10", "3.11", "3.12"):
        for cu in cuda13:
            combos.append((py, cu, "2.10.0"))

    if len(combos) != 24:
        raise RuntimeError(f"Expected 24 base combos, got {len(combos)}")

    include = []
    for py, cu, torch in combos:
        if not is_supported_combo(py, cu, torch):
            raise RuntimeError(f"Unsupported matrix-48 combo: py={py} cu={cu} torch={torch}")
        for abi in ABI_VALUES:
            include.append(
                build_include_entry(
                    os_version=DEFAULT_OS_VERSION,
                    python_version=py,
                    cuda_version=cu,
                    torch_version=torch,
                    cxx11_abi=abi,
                )
            )

    if len(include) != 48:
        raise RuntimeError(f"Expected 48 wheels, got {len(include)}")

    return {"include": include}


def main() -> int:
    preset = os.environ.get("PRESET", "custom")

    if preset == "bleeding-edge":
        matrix = {
            "include": [
                build_include_entry(
                    os_version=DEFAULT_OS_VERSION,
                    python_version="3.12",
                    cuda_version="13.1.1",
                    torch_version="2.10.0",
                    cxx11_abi="FALSE",
                )
            ]
        }
    elif preset == "matrix-48":
        matrix = matrix_48()
    elif preset == "full":
        matrix = full_supported_matrix()
    else:
        matrix = {
            "os-version": parse_csv_env("OS_VERSION"),
            "python-version": parse_csv_env("PYTHON_VERSIONS"),
            "cuda-version": parse_csv_env("CUDA_VERSIONS"),
            "torch-version": parse_csv_env("TORCH_VERSIONS"),
            "cxx11_abi": parse_csv_env("CXX11_ABI"),
        }

    matrix_json = json.dumps(matrix, separators=(",", ":"))
    print(matrix_json)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={matrix_json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
