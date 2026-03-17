"""Shared pytest configuration for calibration tests."""

import os

# Enable debug mode for all tests — allows non-negative thresholds in tests.
os.environ["LITE_ATTENTION_DEBUG"] = "TRUE"
