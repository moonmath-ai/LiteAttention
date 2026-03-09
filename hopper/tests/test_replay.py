"""GPU-required tests for LiteAttention replay mode."""

import pytest
import torch
import torch.nn as nn
from lite_attention import LiteAttention
from lite_attention.lite_attention import (
    LiteAttentionRegistry,
    LiteAttentionReplayConfig,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Module has no registry"),
]

BATCH = 1
SEQ_LEN = 4096
HEADS = 8
HEAD_DIM = 128


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    q = torch.randn(
        BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    return q, k, v


class SimpleModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attn0 = LiteAttention(**kwargs)
        self.attn1 = LiteAttention(**kwargs)


def run_steps(registry, q, k, v, n):
    """Run n forward steps through all modules, return outputs per step."""
    outputs = []
    for _ in range(n):
        step_out = []
        for mod in registry.named_modules.values():
            step_out.append(mod(q, k, v))
        outputs.append(step_out)
    return outputs


# ===========================================================================
# Basic replay: capture then replay, verify skip patterns match
# ===========================================================================


def test_replay_matches_capture(qkv, tmp_path):
    """Run with const threshold + capture, then replay; skip patterns should match."""
    q, k, v = qkv
    n_steps = 5
    threshold = -8.0

    # --- Phase 1: run with capture ---
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(
        model, mode="const", threshold=threshold
    )
    capture_path = tmp_path / "capture.pt"
    registry.enable_capture(save_path=capture_path, qk_block_map=True)
    outputs_capture = run_steps(registry, q, k, v, n_steps)
    registry.save()

    # --- Phase 2: replay from capture file ---
    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(
        model2, mode="replay", filename=capture_path
    )
    outputs_replay = run_steps(registry2, q, k, v, n_steps)

    # Step 0: both start with "compute all" → outputs must match exactly
    for i, (cap_out, rep_out) in enumerate(zip(outputs_capture[0], outputs_replay[0])):
        torch.testing.assert_close(cap_out, rep_out, msg=f"step 0, module {i}")

    # Steps 1+: replay uses the write_list from the previous capture step as
    # read_list, which is the same data the normal run would have used.
    # Outputs should match exactly.
    for step in range(1, n_steps):
        for i, (cap_out, rep_out) in enumerate(
            zip(outputs_capture[step], outputs_replay[step])
        ):
            torch.testing.assert_close(cap_out, rep_out, msg=f"step {step}, module {i}")


def test_replay_with_disabled_steps(qkv, tmp_path):
    """Replay with disabled_steps should produce disabled output for early steps."""
    q, k, v = qkv
    n_steps = 5
    disabled_steps = 2
    threshold = -8.0

    # Capture with disabled_steps
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(
        model,
        mode="const",
        threshold=threshold,
        disabled_steps=disabled_steps,
    )
    capture_path = tmp_path / "capture.pt"
    registry.enable_capture(save_path=capture_path, qk_block_map=True)
    outputs_capture = run_steps(registry, q, k, v, n_steps)
    registry.save()

    # Replay with same disabled_steps
    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(
        model2,
        mode="replay",
        filename=capture_path,
        disabled_steps=disabled_steps,
    )
    outputs_replay = run_steps(registry2, q, k, v, n_steps)

    # All steps should match
    for step in range(n_steps):
        for i, (cap_out, rep_out) in enumerate(
            zip(outputs_capture[step], outputs_replay[step])
        ):
            torch.testing.assert_close(cap_out, rep_out, msg=f"step {step}, module {i}")


# ===========================================================================
# Validation errors
# ===========================================================================


def test_replay_missing_module_raises(qkv, tmp_path):
    """Replay should error if a module is not in the capture file."""
    q, k, v = qkv

    # Create a capture with one model
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    capture_path = tmp_path / "capture.pt"
    registry.enable_capture(save_path=capture_path, qk_block_map=True)
    run_steps(registry, q, k, v, 2)
    registry.save()

    # Try to replay with a different model (different module names)
    class DifferentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.different_attn = LiteAttention()

    model2 = DifferentModel()
    with pytest.raises(ValueError, match="not found in capture file"):
        LiteAttentionRegistry.from_model(model2, mode="replay", filename=capture_path)


def test_replay_no_skip_lists_raises(qkv, tmp_path):
    """Replay should error if capture file has no skip_lists (only pct)."""
    q, k, v = qkv

    # Capture without qk_block_map or attn_map → no skip_lists saved
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    capture_path = tmp_path / "capture.pt"
    # Only pct capture, no skip_lists
    registry.enable_capture(save_path=capture_path, qk_block_map=False, attn_map=False)
    run_steps(registry, q, k, v, 2)
    registry.save()

    model2 = SimpleModel()
    with pytest.raises(ValueError, match="no skip_lists"):
        LiteAttentionRegistry.from_model(model2, mode="replay", filename=capture_path)


def test_replay_subset_heads_raises(qkv, tmp_path):
    """Replay should error if only a subset of heads was captured."""
    q, k, v = qkv

    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    capture_path = tmp_path / "capture.pt"
    # Capture only heads [0, 1] out of 8
    registry.enable_capture(save_path=capture_path, qk_block_map=True, heads=[0, 1])
    run_steps(registry, q, k, v, 2)
    registry.save()

    model2 = SimpleModel()
    with pytest.raises(ValueError, match="replay requires all"):
        LiteAttentionRegistry.from_model(model2, mode="replay", filename=capture_path)


# ===========================================================================
# Batch expansion
# ===========================================================================


def test_replay_batch_expansion(tmp_path):
    """Replay with larger batch than capture should expand correctly."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Capture with batch=1
    q1 = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k1 = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    model = SimpleModel(max_batch_size=2)
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    capture_path = tmp_path / "capture.pt"
    registry.enable_capture(save_path=capture_path, qk_block_map=True)
    run_steps(registry, q1, k1, v1, 3)
    registry.save()

    # Replay with batch=2
    q2 = torch.randn(2, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k2 = torch.randn(2, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v2 = torch.randn(2, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    model2 = SimpleModel(max_batch_size=2)
    registry2 = LiteAttentionRegistry.from_model(
        model2, mode="replay", filename=capture_path
    )
    # Should not raise — batch expansion handles the mismatch
    run_steps(registry2, q2, k2, v2, 3)


# ===========================================================================
# TOML config path
# ===========================================================================


def test_replay_via_toml_config(qkv, tmp_path):
    """Replay via a TOML config file pointing to a .pt capture."""
    from lite_attention.calibrated_module import CalibratedConfigDict

    q, k, v = qkv
    n_steps = 3

    # Capture
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    capture_path = tmp_path / "capture.pt"
    registry.enable_capture(save_path=capture_path, qk_block_map=True)
    outputs_capture = run_steps(registry, q, k, v, n_steps)
    registry.save()

    # Build TOML config
    names = list(registry.named_modules.keys())
    ccd = CalibratedConfigDict(
        {
            name: LiteAttentionReplayConfig(
                skip_list_file=str(capture_path), write_next=True
            )
            for name in names
        }
    )
    toml_path = tmp_path / "replay_config.toml"
    ccd.save(toml_path)

    # Replay from TOML
    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(
        model2, mode="replay", filename=toml_path
    )
    outputs_replay = run_steps(registry2, q, k, v, n_steps)

    for step in range(n_steps):
        for i, (cap_out, rep_out) in enumerate(
            zip(outputs_capture[step], outputs_replay[step])
        ):
            torch.testing.assert_close(cap_out, rep_out, msg=f"step {step}, module {i}")
