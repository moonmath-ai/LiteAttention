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
SHORT_SEQ_LEN = 1024
HEADS = 8
HEAD_DIM = 128


def _make_qkv(batch, seq_len):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    q = torch.randn(
        batch, seq_len, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, seq_len, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch, seq_len, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    return q, k, v


@pytest.fixture
def qkv():
    return _make_qkv(BATCH, SEQ_LEN)


@pytest.fixture
def short_qkv():
    return _make_qkv(BATCH, SHORT_SEQ_LEN)


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
        assert len(mod_data["timesteps"]) == 5
        # pct_per_head: [T=5, B=1, H=8]
        assert mod_data["pct_per_head"].shape == (5, BATCH, HEADS)
        assert len(mod_data["thresholds"]) == 5
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
        attn_map=True,
        attn_map_modules=["attn0"],
        attn_map_timesteps=[0, 2],
        heads=[0, 3],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data = load_capture(save_path)
    for name in ["attn0", "attn1"]:
        mod_data = data["modules"][name]
        # pct always has ALL timesteps and ALL heads
        assert mod_data["pct_per_head"].shape == (3, BATCH, HEADS)
        assert len(mod_data["timesteps"]) == 3

    # attn0 has maps for timesteps 0 and 2 only
    attn0 = data["modules"]["attn0"]
    assert "attn_maps" in attn0
    assert attn0["map_timesteps"] == [0, 2]
    assert attn0["map_heads"] == [0, 3]
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

    registry.enable_capture(save_path=save_path, attn_map=True, attn_map_res=32)
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
        attn_map=True,
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
        attn_map=True,
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
        attn_map=True,
        attn_map_timesteps=[1, 3],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=5)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert len(mod_data["timesteps"]) == 5  # pct has all 5
        assert mod_data["map_timesteps"] == [1, 3]
        assert mod_data["attn_maps"].shape[0] == 2


def test_attn_map_head_filter(qkv, tmp_path):
    """attn_map_heads filters which heads get maps."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        heads=[0, 5],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        assert mod_data["pct_per_head"].shape[2] == HEADS  # pct has all heads
        assert mod_data["map_heads"] == [0, 5]
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

    registry.enable_capture(save_path=save_path, attn_map=True, attn_map_res=0)
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data1 = load_capture(save_path)
    n1 = len(list(data1["modules"].values())[0]["timesteps"])

    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data2 = load_capture(save_path)
    n2 = len(list(data2["modules"].values())[0]["timesteps"])
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


@pytest.mark.parametrize("max_batch_size", [2, 3])
def test_capture_batch_out_of_range(qkv, tmp_path, max_batch_size):
    """attn_map_batch_indices beyond actual batch size are silently skipped."""
    q, k, v = qkv  # BATCH=1
    model = SimpleModel(max_batch_size=max_batch_size)
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        batches=[0, 5],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=1)
    registry.save()

    data = load_capture(save_path)
    for mod_data in data["modules"].values():
        # pct has actual batch items (BATCH=1)
        assert mod_data["pct_per_head"].shape[1] == BATCH
        # skip_lists capture actual batch_size, not max_batch_size
        assert mod_data["skip_lists"].shape[1] == BATCH


# ===========================================================================
# skip_list_to_mask (unchanged)
# ===========================================================================


def test_skip_list_to_mask_basic():
    """Verify mask decoding from a known skip list pattern."""
    ktiles = 8
    row = torch.zeros(ktiles + 2, dtype=torch.int16)
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
        attn_map=True,
        heads=[0],
        attn_map_timesteps=[0],
        batches=[0],
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
                "timesteps": [0],
                "thresholds": [-8.0],
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


# ===========================================================================
# Stats capture
# ===========================================================================


def test_stats_captured_basic(qkv, tmp_path):
    """stats=True captures running statistics at full resolution."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        heads=[0],
        batches=[0],
        attn_map_res=32,
        stats=True,
    )
    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data = load_capture(save_path)
    attn0 = data["modules"]["attn0"]

    # Stats should be present for attn0 (which has map capture + stats)
    assert "stats_mean" in attn0
    assert "stats_std" in attn0
    assert "stats_max" in attn0
    assert "stats_min" in attn0
    assert attn0["stats_count"] == 3

    # Shape: [n_batch_sel=1, n_head_sel=1, seq_q, seq_k]
    assert attn0["stats_mean"].shape == (1, 1, SEQ_LEN, SEQ_LEN)
    assert attn0["stats_std"].shape == (1, 1, SEQ_LEN, SEQ_LEN)
    assert attn0["stats_max"].shape == (1, 1, SEQ_LEN, SEQ_LEN)
    assert attn0["stats_min"].shape == (1, 1, SEQ_LEN, SEQ_LEN)
    assert attn0["stats_batch_indices"] == [0]
    assert attn0["stats_heads"] == [0]

    # attn1 should NOT have stats (not in attn_map_modules)
    attn1 = data["modules"]["attn1"]
    assert "stats_mean" not in attn1


def test_stats_not_captured_when_disabled(qkv, tmp_path):
    """stats=False (default) does not capture statistics."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        attn_map_res=32,
    )
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    assert "stats_mean" not in data["modules"]["attn0"]


def test_stats_values_are_valid(qkv, tmp_path):
    """Stats values are within valid ranges for softmax outputs."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        heads=[0],
        batches=[0],
        attn_map_res=64,
        stats=True,
    )
    warmup_all(registry, q, k, v, n=3)
    registry.save()

    data = load_capture(save_path)
    attn0 = data["modules"]["attn0"]

    # Softmax outputs are in [0, 1]
    assert attn0["stats_mean"].min() >= 0.0
    assert attn0["stats_mean"].max() <= 1.0
    assert attn0["stats_max"].min() >= 0.0
    assert attn0["stats_max"].max() <= 1.0
    assert attn0["stats_min"].min() >= 0.0
    assert attn0["stats_min"].max() <= 1.0

    # std is non-negative
    assert attn0["stats_std"].min() >= 0.0

    # min <= mean <= max
    assert (attn0["stats_min"] <= attn0["stats_mean"] + 1e-6).all()
    assert (attn0["stats_mean"] <= attn0["stats_max"] + 1e-6).all()


def test_stats_accumulate_all_timesteps(qkv, tmp_path):
    """Stats accumulate across ALL forward passes, regardless of attn_map_timesteps."""
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        attn_map_timesteps=[0, 2],  # only capture maps for timesteps 0 and 2
        heads=[0],
        batches=[0],
        attn_map_res=32,
        stats=True,
    )
    warmup_all(registry, q, k, v, n=5)
    registry.save()

    data = load_capture(save_path)
    attn0 = data["modules"]["attn0"]

    # Detailed maps only captured for timesteps 0 and 2
    assert attn0["map_timesteps"] == [0, 2]
    assert attn0["attn_maps"].shape[0] == 2

    # Stats accumulated over ALL 5 timesteps
    assert attn0["stats_count"] == 5


def test_stats_multiple_heads_and_batches(tmp_path):
    """Stats capture with multiple heads and batches has correct shape."""
    batch = 2
    q, k, v = _make_qkv(batch, SHORT_SEQ_LEN)

    model = SimpleModel(max_batch_size=batch)
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    sel_heads = [0, 3, 7]
    sel_batch = [0, 1]
    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        heads=sel_heads,
        batches=sel_batch,
        attn_map_res=64,
        stats=True,
    )
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    attn0 = data["modules"]["attn0"]

    assert attn0["stats_mean"].shape == (
        len(sel_batch),
        len(sel_heads),
        SHORT_SEQ_LEN,
        SHORT_SEQ_LEN,
    )
    assert attn0["stats_batch_indices"] == sel_batch
    assert attn0["stats_heads"] == sel_heads
    assert attn0["stats_count"] == 2


def test_stats_all_heads_all_batches(short_qkv, tmp_path):
    """Stats with None heads/batch captures all heads and batches."""
    q, k, v = short_qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-8.0)
    save_path = tmp_path / "capture.pt"

    registry.enable_capture(
        save_path=save_path,
        attn_map=True,
        attn_map_modules=["attn0"],
        heads=None,
        batches=None,
        attn_map_res=64,
        stats=True,
    )
    warmup_all(registry, q, k, v, n=2)
    registry.save()

    data = load_capture(save_path)
    attn0 = data["modules"]["attn0"]

    assert attn0["stats_mean"].shape == (BATCH, HEADS, SHORT_SEQ_LEN, SHORT_SEQ_LEN)
    assert attn0["stats_batch_indices"] == list(range(BATCH))
    assert attn0["stats_heads"] == list(range(HEADS))
