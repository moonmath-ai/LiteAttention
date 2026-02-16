"""Shared configuration for la_tests."""

import os

import pytest

# Enable debug mode for all la_tests — allows non-negative thresholds in tests.
os.environ["LITE_ATTENTION_DEBUG"] = "TRUE"


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when CUDA is not available."""
    try:
        import torch

        if torch.cuda.is_available():
            return
    except ImportError:
        pass
    skip_gpu = pytest.mark.skip(reason="No CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
