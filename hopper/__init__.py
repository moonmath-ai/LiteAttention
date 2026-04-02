__version__ = "0.4.0"

import torch

if torch.version.hip is not None:
    # AMD / ROCm: use CK-based LiteAttentionAMD (no CUDA extension needed)
    from .lite_attention_amd import LiteAttentionAMD, patch_model

    __all__ = [
        "LiteAttentionAMD",
        "patch_model",
    ]
else:
    # NVIDIA / CUDA: use Hopper LiteAttention (requires _C extension)
    from .lite_attention import (
        LiteAttention,
        LiteAttentionCalibConfig,
        LiteAttentionRegistry,
        LiteAttentionRunConfig,
        SeqParallelLiteAttention,
    )

    __all__ = [
        "LiteAttention",
        "SeqParallelLiteAttention",
        "LiteAttentionRunConfig",
        "LiteAttentionCalibConfig",
        "LiteAttentionRegistry",
    ]
