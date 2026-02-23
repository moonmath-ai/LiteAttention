"""GPU-required tests for LiteAttention debug capture."""

import pytest
import torch
import torch.nn as nn
from lite_attention import LiteAttention, load_capture, render_skip_images
from lite_attention.debug_capture import skip_list_to_mask
from lite_attention.lite_attention import LiteAttentionRegistry

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


def warmup_all(registry, q, k, v, n=1):
    for _ in range(n):
        torch.cuda.synchronize()
        for mod in registry.named_modules.values():
            mod(q, k, v)
        torch.cuda.synchronize()


# ===========================================================================
# Tier 1: pct_per_head always captured
# ===========================================================================


def test_pct_captured_for_all_modules_and_timesteps(qkv, tmp_path):
    """enable_capture with attn_map_res=0 still captures pct for all modules/timesteps."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, attn_map_res=0)
    warmup_all(registry, q, k, v, n=5)
    registry.save()

    data = load_capture(save_path)
    assert "modules" in data
    for name in registry.named_modules:
        mod_data = data["modules"][name]
        assert mod_data["timesteps"].shape[0] == 5
        # pct_per_head: [T=5, B=1, H=8]
        assert mod_data["pct_per_head"].shape == (5, BATCH, HEADS)
        assert mod_data["thresholds"].shape == (5,)
        assert "attn_maps" not in mod_data
        assert "skip_lists" not in mod_data


def test_pct_shape_with_maps_enabled(qkv, tmp_path):
    """pct_per_head covers ALL timesteps/heads even when maps filter to subset."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_modules=["attn0"],
        attn_map_timesteps=[0, 2],
        attn_map_heads=[0, 3],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data = load_capture(save_path)
    for name in ["attn0", "attn1"]:
        mod_data = data["modules"][name]
        # pct always has ALL timesteps and ALL heads
        assert mod_data["pct_per_head"].shape == (3, BATCH, HEADS)
        assert mod_data["timesteps"].shape[0] == 3

    # attn0 has maps for timesteps 0 and 2 only
    attn0 = data["modules"]["attn0"]
    assert "attn_maps" in attn0
    assert torch.equal(attn0["map_timesteps"], torch.tensor([0, 2]))
    assert torch.equal(attn0["map_heads"], torch.tensor([0, 3]))
    assert attn0["attn_maps"].shape == (2, 1, 2, 32, 32)
    assert attn0["skip_lists"].shape[0] == 2  # 2 map timesteps

    # attn1 has no maps
    assert "attn_maps" not in data["modules"]["attn1"]


# ===========================================================================
# Tier 2: attn map filtering
# ===========================================================================


def test_attn_maps_all_modules_when_modules_none(qkv, tmp_path):
    """attn_map_modules=None captures maps for all modules."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(save_path=save_path, attn_map_res=32)
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    for name in ["attn0", "attn1"]:
        assert "attn_maps" in data["modules"][name]
        assert data["modules"][name]["attn_maps"].shape[0] == 2


def test_attn_map_module_filter_list(qkv, tmp_path):
    """attn_map_modules as list selects specific modules."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_modules=["attn0"],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    assert "attn_maps" in data["modules"]["attn0"]
    assert "attn_maps" not in data["modules"]["attn1"]


def test_attn_map_module_filter_callable(qkv, tmp_path):
    """attn_map_modules as callable filter."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_modules=lambda name: "1" in name,
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    assert "attn_maps" not in data["modules"]["attn0"]
    assert "attn_maps" in data["modules"]["attn1"]


def test_attn_map_timestep_filter(qkv, tmp_path):
    """attn_map_timesteps filters which timesteps get maps."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_timesteps=[1, 3],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=5)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert mod_data["timesteps"].shape[0] == 5  # pct has all 5
        assert torch.equal(mod_data["map_timesteps"], torch.tensor([1, 3]))
        assert mod_data["attn_maps"].shape[0] == 2


def test_attn_map_head_filter(qkv, tmp_path):
    """attn_map_heads filters which heads get maps."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_heads=[0, 5],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert mod_data["pct_per_head"].shape[2] == HEADS  # pct has all heads
        assert torch.equal(mod_data["map_heads"], torch.tensor([0, 5]))
        assert mod_data["attn_maps"].shape[2] == 2  # maps only for 2 heads


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

    assert len(attn._captured_pct) > 0
    with pytest.warns(UserWarning, match="unsaved capture data"):
        attn.reset_skip_state()
    assert len(attn._captured_pct) == 0


def test_reset_skip_state_no_warn_when_empty(qkv):
    """reset_skip_state doesn't warn when there's no captured data."""
    q, k, v = qkv
    attn = LiteAttention(threshold=-8.0)
    attn._enable_capture(attn_map_res=0)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attn.reset_skip_state()


# ===========================================================================
# Batch bounds
# ===========================================================================


def test_capture_batch_out_of_range(qkv, tmp_path):
    """attn_map_batch_indices beyond actual batch size are silently skipped."""
    q, k, v = qkv  # BATCH=1
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map_batch_indices=[0, 5],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        # pct has all batch items (BATCH=1)
        assert mod_data["pct_per_head"].shape[1] == BATCH
        # maps only have batch 0 (batch 5 > actual batch size 1)
        assert mod_data["skip_lists"].shape[1] == 1


# ===========================================================================
# skip_list_to_mask (unchanged)
# ===========================================================================


def test_skip_list_to_mask_basic():
    """Verify mask decoding from a known skip list pattern."""
    ktiles = 8
    row = torch.zeros(ktiles + 1, dtype=torch.int16)
    row[0] = 4
    row[1] = 7
    row[2] = 3
    row[3] = 2
    row[4] = 0

    skip_list_2d = row.unsqueeze(0)
    mask = skip_list_to_mask(skip_list_2d, ktiles)

    assert mask.shape == (1, ktiles)
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
        attn_map_heads=[0],
        attn_map_timesteps=[0],
        attn_map_batch_indices=[0],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    vis_dir = tmp_path / "vis"
    render_skip_images(data, output_dir=vis_dir)

    for name in registry.named_modules:
        png = vis_dir / name / "batch_0" / "head_0" / "t_0000.png"
        assert png.exists(), f"Expected {png}"


def test_render_skip_images_skips_modules_without_maps(tmp_path):
    """render_skip_images silently skips modules with no attn_maps."""
    data = {
        "modules": {
            "no_maps": {
                "timesteps": torch.tensor([0]),
                "thresholds": torch.tensor([-8.0]),
                "pct_per_head": torch.zeros(1, 1, 1),
                "seq_len_q": 256,
                "seq_len_k": 256,
                "head_dim": 128,
                "use_int8": False,
                "kBlockM": 64,
                "kBlockN": 128,
            }
        }
    }
    # Should not raise — just produces no images
    render_skip_images(data, output_dir=tmp_path / "vis")
