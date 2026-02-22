"""GPU-required tests for LiteAttention debug capture."""

import pytest
import torch
import torch.nn as nn

from lite_attention import LiteAttention, load_capture, render_skip_images
from lite_attention.lite_attention import LiteAttentionRegistry
from lite_attention.debug_capture import skip_list_to_mask

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
    q = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return q, k, v


class SimpleModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attn0 = LiteAttention(**kwargs)
        self.attn1 = LiteAttention(**kwargs)


def warmup_all(registry, q, k, v, n=1):
    for _ in range(n):
        torch.cuda.synchronize()
        for mod in registry.named_modules.values():
            mod(q, k, v)
        torch.cuda.synchronize()


# ===========================================================================
# Basic capture
# ===========================================================================


def test_capture_basic_shapes(qkv, tmp_path):
    """Enable capture on all modules, run forward, save, reload, check shapes."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        heads=[0, 2],
        timesteps=[0, 2],
        batch_indices=[0],
        attn_map_res=64,
    )

    warmup_all(registry, q, k, v, n=3)
    registry.save()

    assert save_path.exists()
    data = load_capture(save_path)

    assert "modules" in data
    for name in registry.named_modules:
        mod_data = data["modules"][name]

        # Timesteps 0 and 2 captured (not 1)
        assert torch.equal(mod_data["timesteps"], torch.tensor([0, 2]))
        assert torch.equal(mod_data["heads"], torch.tensor([0, 2]))
        assert torch.equal(mod_data["batch_indices"], torch.tensor([0]))

        assert mod_data["seq_len_q"] == SEQ_LEN
        assert mod_data["seq_len_k"] == SEQ_LEN
        assert mod_data["head_dim"] == HEAD_DIM

        # skip_lists: [n_captured=2, n_batch=1, n_heads=2, qtiles, ktiles+1]
        sl = mod_data["skip_lists"]
        assert sl.shape[0] == 2  # 2 captured timesteps
        assert sl.shape[1] == 1  # 1 batch
        assert sl.shape[2] == 2  # 2 heads
        assert sl.dtype == torch.int16

        # pct_per_head: [n_captured=2, n_batch=1, n_heads=2]
        assert mod_data["pct_per_head"].shape == (2, 1, 2)

        # attn_maps: [n_captured=2, n_batch=1, n_heads=2, 64, 64]
        assert mod_data["attn_maps"].shape == (2, 1, 2, 64, 64)
        assert mod_data["attn_maps"].dtype == torch.float16

        # thresholds
        assert mod_data["thresholds"].shape == (2,)
        assert (mod_data["thresholds"] == -8.0).all()

        # per-module metadata
        assert mod_data["kBlockM"] > 0
        assert mod_data["kBlockN"] > 0


def test_capture_no_attn_maps(qkv, tmp_path):
    """attn_map_res=0 skips attention map computation."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture_no_maps.pt"

    registry.enable_capture(save_path=save_path, attn_map_res=0)
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert "attn_maps" not in mod_data
        assert mod_data["skip_lists"].shape[0] == 2
        assert mod_data["pct_per_head"].shape[0] == 2


# ===========================================================================
# Module filtering
# ===========================================================================


def test_capture_module_filter_list(qkv, tmp_path):
    """Only capture modules matching the list."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, modules=["attn0"], attn_map_res=0)
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    assert "attn0" in data["modules"]
    assert "attn1" not in data["modules"]


def test_capture_module_filter_callable(qkv, tmp_path):
    """Callable filter on module names."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        modules=lambda name: "1" in name,
        attn_map_res=0,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    assert "attn1" in data["modules"]
    assert "attn0" not in data["modules"]


# ===========================================================================
# Timestep & head filtering
# ===========================================================================


def test_capture_all_timesteps(qkv, tmp_path):
    """timesteps=None captures every forward pass."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, timesteps=None, attn_map_res=0)
    warmup_all(registry, q, k, v, n=5)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert mod_data["timesteps"].shape[0] == 5


def test_capture_all_heads(qkv, tmp_path):
    """heads=None captures all heads."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, heads=None, attn_map_res=0)
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert mod_data["heads"].shape[0] == HEADS
        assert mod_data["skip_lists"].shape[2] == HEADS


# ===========================================================================
# Save behavior
# ===========================================================================


def test_save_preserves_data(qkv, tmp_path):
    """save() does not clear captured data — second save includes all data."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, attn_map_res=0)
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data1 = load_capture(save_path)
    n1 = list(data1["modules"].values())[0]["timesteps"].shape[0]

    # Run more, save again
    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data2 = load_capture(save_path)
    n2 = list(data2["modules"].values())[0]["timesteps"].shape[0]
    assert n2 == n1 + 3


def test_reset_skip_state_warns_unsaved(qkv):
    """reset_skip_state warns when captured data would be lost."""
    q, k, v = qkv
    attn = LiteAttention(threshold=-8.0)
    attn._enable_capture(attn_map_res=0)

    torch.cuda.synchronize()
    attn(q, k, v)
    torch.cuda.synchronize()

    assert len(attn._captured_data) > 0
    with pytest.warns(UserWarning, match="unsaved capture data"):
        attn.reset_skip_state()
    assert len(attn._captured_data) == 0


def test_reset_skip_state_no_warn_when_empty(qkv):
    """reset_skip_state doesn't warn when there's no captured data."""
    q, k, v = qkv
    attn = LiteAttention(threshold=-8.0)
    attn._enable_capture(attn_map_res=0)

    # No forward passes — no captured data
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attn.reset_skip_state()


# ===========================================================================
# Batch bounds
# ===========================================================================


def test_capture_batch_out_of_range(qkv, tmp_path):
    """batch_indices beyond actual batch size are silently skipped."""
    q, k, v = qkv  # BATCH=1
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path, batch_indices=[0, 5], attn_map_res=0,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        # Only batch 0 should be captured (batch 5 > actual batch size 1)
        assert mod_data["skip_lists"].shape[1] == 1


# ===========================================================================
# skip_list_to_mask
# ===========================================================================


def test_skip_list_to_mask_basic():
    """Verify mask decoding from a known skip list pattern."""
    ktiles = 8
    # Format: [length, r1, r2, r1, r2, ...]
    # 2 pairs: range (7,3) and (2,0) — reversed, step=1
    # After +step: (8,4) and (3,1) → sorted: (4,8) and (1,3)
    row = torch.zeros(ktiles + 1, dtype=torch.int16)
    row[0] = 4      # length (4 entries = 2 pairs)
    row[1] = 7      # r1 > r2 → step=1
    row[2] = 3
    row[3] = 2
    row[4] = 0

    skip_list_2d = row.unsqueeze(0)  # [1, ktiles+1]
    mask = skip_list_to_mask(skip_list_2d, ktiles)

    assert mask.shape == (1, ktiles)
    # After +step=1: pairs (8,4) and (3,1), sorted → (4,8) and (1,3)
    # mask[0, 4:8] = True, mask[0, 1:3] = True
    expected = torch.tensor([False, True, True, False, True, True, True, True])
    assert torch.equal(mask[0], expected)


# ===========================================================================
# Offline rendering
# ===========================================================================


def test_render_skip_images(qkv, tmp_path):
    """Capture with attn maps, render to PNGs, verify files created."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        heads=[0],
        timesteps=[0],
        batch_indices=[0],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    vis_dir = tmp_path / "vis"
    render_skip_images(data, output_dir=vis_dir)

    # Check PNGs were created
    for name in registry.named_modules:
        png = vis_dir / name / "batch_0" / "head_0" / "t_0000.png"
        assert png.exists(), f"Expected {png}"


def test_render_skip_images_no_attn_maps_raises(tmp_path):
    """render_skip_images raises if attn_maps are missing."""
    data = {
        "modules": {
            "test": {
                "kBlockM": 64,
                "kBlockN": 128,
                "timesteps": torch.tensor([0]),
                "heads": torch.tensor([0]),
                "batch_indices": torch.tensor([0]),
                "skip_lists": torch.zeros(1, 1, 1, 4, 9, dtype=torch.int16),
                "pct_per_head": torch.zeros(1, 1, 1),
                "seq_len_q": 256,
                "seq_len_k": 256,
                # no "attn_maps" key
            }
        }
    }
    with pytest.raises(ValueError, match="no attention maps"):
        render_skip_images(data, output_dir=tmp_path / "vis")
