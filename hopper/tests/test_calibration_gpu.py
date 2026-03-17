"""GPU-required tests for LiteAttention calibration integration."""

import pytest
import torch
import torch.nn as nn
from lite_attention import LiteAttention
from lite_attention.calibrated_module import (
    CalibratedConfigDict,
    ConfigList,
)
from lite_attention.lite_attention import (
    LiteAttentionCalibConfig,
    LiteAttentionDisabledConfig,
    LiteAttentionRegistry,
    LiteAttentionRunConfig,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Module has no registry"),
]


# ---------------------------------------------------------------------------
# Constants — smaller than test_lite_attention.py for speed
# ---------------------------------------------------------------------------

BATCH = 1
SEQ_LEN = 4096
HEADS = 8
HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Model with two LiteAttention sub-modules for registry tests."""

    def __init__(self, **kwargs):
        super().__init__()
        self.attn0 = LiteAttention(**kwargs)
        self.attn1 = LiteAttention(**kwargs)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def small_qkv():
    """Smaller tensors for tests that only check config output types, not numerics."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    sq = 256
    h = 2
    q = torch.randn(BATCH, sq, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(BATCH, sq, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, sq, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return q, k, v


@pytest.fixture
def tmp_toml(tmp_path):
    return tmp_path / "calibrated.toml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def warmup(attn, q, k, v, n=1):
    for _ in range(n):
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()


# ===========================================================================
# LiteAttention config integration
# ===========================================================================


def test_constructor_threshold_creates_run_config():
    attn = LiteAttention(threshold=-5.0)
    cfg = attn.config
    assert isinstance(cfg, LiteAttentionRunConfig)
    assert cfg.threshold == -5.0


def test_constructor_config_param():
    cfg = LiteAttentionRunConfig(threshold=-3.0)
    attn = LiteAttention(config=cfg)
    assert attn.config.threshold == -3.0


def test_constructor_threshold_and_config_raises():
    with pytest.raises(ValueError, match="Cannot specify both"):
        LiteAttention(threshold=-5.0, config=LiteAttentionRunConfig(threshold=-3.0))


def test_threshold_property_reads_config():
    attn = LiteAttention(threshold=-7.0)
    assert attn.threshold == -7.0


def test_set_threshold_warns_deprecated():
    attn = LiteAttention(threshold=-5.0)
    with pytest.warns(UserWarning, match="deprecated"):
        attn.set_threshold(-8.0)
    assert attn.threshold == -8.0


def test_forward_records_calibration_results(qkv):
    q, k, v = qkv
    attn = LiteAttention(threshold=-5.0)
    warmup(attn, q, k, v, 2)
    assert len(attn._config_output) == 2
    assert all(isinstance(r, LiteAttentionRunConfig) for r in attn._config_output)


# ===========================================================================
# Per-timestep configs
# ===========================================================================


def test_per_timestep_config_list(qkv):
    q, k, v = qkv
    thresholds = [-5.0, -8.0, -12.0]
    cl = ConfigList([LiteAttentionRunConfig(threshold=t) for t in thresholds])

    # Wrap in a model + registry so _registry is not None.
    # (Without a registry, add_calibration_results tries self.config after the
    # last timestep, which goes out-of-bounds on the ConfigList.)
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = LiteAttention()

    m = _M()
    _ = LiteAttentionRegistry.from_model(m, mode="const", threshold=-99.0)
    attn = m.attn
    # Override the registry config with our per-timestep list
    attn._registry_config = cl

    for i, expected_th in enumerate(thresholds):
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()
        assert attn._config_output[i].threshold == expected_th


def test_reset_skip_state_resets_config_index(qkv):
    q, k, v = qkv
    attn = LiteAttention(threshold=-5.0)
    warmup(attn, q, k, v, 3)
    assert attn._config_index == 3
    attn.reset_skip_state()
    assert attn._config_index == 0
    assert len(attn._config_output) == 0


# ===========================================================================
# LiteAttentionRegistry
# ===========================================================================


def test_registry_from_model_const(simple_model):
    registry = LiteAttentionRegistry.from_model(
        simple_model, mode="const", threshold=-6.0
    )
    for mod in registry.named_modules.values():
        assert mod._registry_config.threshold == -6.0


def test_registry_from_model_const_default_warns(simple_model):
    with pytest.warns(UserWarning, match="no 'threshold'"):
        registry = LiteAttentionRegistry.from_model(simple_model, mode="const")
    for mod in registry.named_modules.values():
        assert (
            mod._registry_config.threshold == LiteAttentionRunConfig.default().threshold
        )


@pytest.mark.filterwarnings("ignore:no 'threshold'")
def test_registry_from_model_default_mode_warns(simple_model):
    with pytest.warns(UserWarning, match="No 'mode'"):
        LiteAttentionRegistry.from_model(simple_model)


def test_registry_from_model_unknown_mode_raises(simple_model):
    with pytest.raises(ValueError, match="Unknown mode"):
        LiteAttentionRegistry.from_model(simple_model, mode="invalid")


def test_registry_from_model_load_missing_filename_raises(simple_model):
    with pytest.raises(ValueError, match="filename is required"):
        LiteAttentionRegistry.from_model(simple_model, mode="load")


def test_registry_from_model_calib_missing_filename_raises(simple_model):
    with pytest.raises(ValueError, match="filename is required"):
        LiteAttentionRegistry.from_model(simple_model, mode="calib")


def test_registry_from_model_load(simple_model, tmp_toml):
    registry = LiteAttentionRegistry.from_model(
        simple_model, mode="const", threshold=-4.0
    )
    names = list(registry.named_modules.keys())
    ccd = CalibratedConfigDict(
        {name: LiteAttentionRunConfig(threshold=-4.0) for name in names}
    )
    ccd.save(tmp_toml)

    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(model2, mode="load", filename=tmp_toml)
    for mod in registry2.named_modules.values():
        assert mod._registry_config.threshold == -4.0


def test_registry_force_clears_instance_configs():
    model = SimpleModel(threshold=-5.0)
    registry = LiteAttentionRegistry.from_model(
        model, mode="const", threshold=-3.0, force=True
    )
    for mod in registry.named_modules.values():
        assert mod._instance_config is None
        assert mod._registry_config.threshold == -3.0


# ===========================================================================
# Calibration end-to-end
# ===========================================================================


def test_calibration_save(qkv, tmp_toml):
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(
        model,
        mode="calib",
        filename=tmp_toml,
        calib_config={"target_error": 0.01, "metric": "L1"},
    )

    for _ in range(3):
        torch.cuda.synchronize()
        for mod in registry.named_modules.values():
            mod(q, k, v)
        torch.cuda.synchronize()

    registry.save_if_calib()
    assert tmp_toml.exists()

    loaded = CalibratedConfigDict.load(tmp_toml, [LiteAttentionRunConfig])
    for name in registry.named_modules:
        assert name in loaded
        configs = loaded[name]
        assert isinstance(configs, list)
        assert len(configs) == 3
        for c in configs:
            assert isinstance(c, LiteAttentionRunConfig)
            assert c.threshold <= 0  # calibrated thresholds should be non-positive


def test_calibration_loose_vs_tight_target(qkv, tmp_path):
    q, k, v = qkv

    thresholds_by_target = {}
    for target in [0.1, 0.001]:
        model = SimpleModel()
        registry = LiteAttentionRegistry.from_model(
            model,
            mode="calib",
            filename=tmp_path / f"calib_{target}.toml",
            calib_config={"target_error": target, "metric": "L1"},
        )
        name = list(registry.named_modules.keys())[0]
        mod = registry.named_modules[name]
        torch.cuda.synchronize()
        mod(q, k, v)
        torch.cuda.synchronize()
        thresholds_by_target[target] = mod._config_output[0].threshold

    # Loose target (0.1) should allow more aggressive skipping (higher / less negative threshold)
    assert thresholds_by_target[0.1] >= thresholds_by_target[0.001]


def test_calibration_save_then_load(qkv, tmp_toml):
    q, k, v = qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(
        model,
        mode="calib",
        filename=tmp_toml,
        calib_config={"target_error": 0.01, "metric": "L1"},
    )
    # Run forward on all modules
    for mod in registry.named_modules.values():
        torch.cuda.synchronize()
        mod(q, k, v)
        torch.cuda.synchronize()

    saved_thresholds = {
        name: mod._config_output[0].threshold
        for name, mod in registry.named_modules.items()
    }
    registry.save_if_calib()

    # Load and verify values match
    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(model2, mode="load", filename=tmp_toml)
    for name, mod2 in registry2.named_modules.items():
        cfg = mod2._registry_config
        assert cfg is not None
        if isinstance(cfg, ConfigList):
            assert cfg[0].threshold == saved_thresholds[name]
        else:
            assert cfg.threshold == saved_thresholds[name]


def test_save_if_calib_noop_for_const():
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="const", threshold=-5.0)
    registry.save_if_calib()


def test_calib_default_config_warns():
    model = SimpleModel()
    with pytest.warns(UserWarning, match="no 'calib_config'"):
        LiteAttentionRegistry.from_model(
            model, mode="calib", filename="/tmp/dummy.toml"
        )


# ===========================================================================
# calc_error
# ===========================================================================


def test_calc_error_identical_tensors():
    t = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    errors = LiteAttention.calc_error(t, t)
    assert set(errors.keys()) == {"Cossim", "L1", "RMSE"}
    assert errors["Cossim"] == pytest.approx(0.0, abs=1e-6)
    assert errors["L1"] == pytest.approx(0.0, abs=1e-6)
    assert errors["RMSE"] == pytest.approx(0.0, abs=1e-6)


def test_calc_error_different_tensors():
    torch.manual_seed(0)
    a = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
    errors = LiteAttention.calc_error(a, b)
    assert errors["Cossim"] > 0
    assert errors["L1"] > 0
    assert errors["RMSE"] > 0


# ===========================================================================
# LiteAttentionDisabledConfig
# ===========================================================================


def test_disabled_config_in_config_list(qkv):
    """Disabled timesteps run regular attention; skipping timesteps use threshold."""
    q, k, v = qkv
    cl = ConfigList(
        [
            LiteAttentionRunConfig(threshold=-5.0),
            LiteAttentionRunConfig(threshold=-5.0),
            LiteAttentionDisabledConfig(),
            LiteAttentionRunConfig(threshold=-5.0),
        ]
    )

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = LiteAttention()

    m = _M()
    _ = LiteAttentionRegistry.from_model(m, mode="const", threshold=-99.0)
    attn = m.attn
    attn._registry_config = cl

    for i in range(4):
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()

    assert len(attn._config_output) == 4
    # Steps 0, 1, 3 should record LiteAttentionRunConfig with threshold -5.0
    for i in [0, 1, 3]:
        assert type(attn._config_output[i]) is LiteAttentionRunConfig
        assert attn._config_output[i].threshold == -5.0
    # Step 2 should record LiteAttentionDisabledConfig
    assert type(attn._config_output[2]) is LiteAttentionDisabledConfig


def test_disabled_config_output_matches_no_skipping(qkv):
    """Output with LiteAttentionDisabledConfig should match enable_skipping=False."""
    q, k, v = qkv

    # Reference: skipping disabled entirely
    attn_ref = LiteAttention(enable_skipping=False)
    torch.cuda.synchronize()
    out_ref = attn_ref(q, k, v)
    torch.cuda.synchronize()

    # Test: disabled via config on a specific timestep
    cl = ConfigList([LiteAttentionDisabledConfig()])

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = LiteAttention()

    m = _M()
    _ = LiteAttentionRegistry.from_model(m, mode="const", threshold=-99.0)
    m.attn._registry_config = cl
    torch.cuda.synchronize()
    out_test = m.attn(q, k, v)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_test, out_ref)


def test_config_list_clamping_gpu(qkv):
    """ConfigList shorter than timestep count repeats last element."""
    q, k, v = qkv
    cl = ConfigList([LiteAttentionRunConfig(threshold=-5.0)])

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = LiteAttention()

    m = _M()
    _ = LiteAttentionRegistry.from_model(m, mode="const", threshold=-99.0)
    attn = m.attn
    attn._registry_config = cl

    # Step 0: in bounds
    torch.cuda.synchronize()
    attn(q, k, v)
    torch.cuda.synchronize()

    # Steps 1-2: clamp beyond the 1-element list
    with pytest.warns(UserWarning, match="clamping to last entry"):
        for _ in range(2):
            torch.cuda.synchronize()
            attn(q, k, v)
            torch.cuda.synchronize()

    assert len(attn._config_output) == 3
    for r in attn._config_output:
        assert isinstance(r, LiteAttentionRunConfig)
        assert r.threshold == -5.0


def test_disabled_then_clamped_config(qkv):
    """Short ConfigList ending with DisabledConfig clamps to disabled forever."""
    q, k, v = qkv
    cl = ConfigList(
        [
            LiteAttentionRunConfig(threshold=-5.0),
            LiteAttentionDisabledConfig(),
        ]
    )

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = LiteAttention()

    m = _M()
    _ = LiteAttentionRegistry.from_model(m, mode="const", threshold=-99.0)
    attn = m.attn
    attn._registry_config = cl

    # Steps 0-1: in bounds
    for _ in range(2):
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()

    # Steps 2-3: clamp beyond the 2-element list
    with pytest.warns(UserWarning, match="clamping to last entry"):
        for _ in range(2):
            torch.cuda.synchronize()
            attn(q, k, v)
            torch.cuda.synchronize()

    assert len(attn._config_output) == 4
    assert type(attn._config_output[0]) is LiteAttentionRunConfig
    for i in [1, 2, 3]:
        assert type(attn._config_output[i]) is LiteAttentionDisabledConfig


# ===========================================================================
# from_model: mode="disable" and disabled_steps
# ===========================================================================


def test_registry_from_model_disable(small_qkv):
    """mode='disable' produces disabled output on every timestep."""
    q, k, v = small_qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="disable")
    for mod in registry.named_modules.values():
        for _ in range(3):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()
        assert len(mod._config_output) == 3
        for r in mod._config_output:
            assert type(r) is LiteAttentionDisabledConfig


def test_registry_from_model_disable_ignores_disabled_steps(small_qkv):
    """disabled_steps has no effect when mode='disable' (already fully disabled)."""
    q, k, v = small_qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(model, mode="disable", disabled_steps=3)
    for mod in registry.named_modules.values():
        for _ in range(5):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()
        # All steps disabled — disabled_steps doesn't add extra structure
        assert len(mod._config_output) == 5
        for r in mod._config_output:
            assert type(r) is LiteAttentionDisabledConfig


def test_registry_from_model_disabled_steps_with_const(small_qkv):
    """disabled_steps prepends disabled entries to a scalar const config."""
    q, k, v = small_qkv
    model = SimpleModel()
    registry = LiteAttentionRegistry.from_model(
        model, mode="const", threshold=-5.0, disabled_steps=2
    )
    for mod in registry.named_modules.values():
        # Steps 0-2: in bounds (2 disabled + 1 const)
        for _ in range(3):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()

        # Step 3: clamps beyond the 3-element list
        with pytest.warns(UserWarning, match="clamping to last entry"):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()

        out = mod._config_output
        assert len(out) == 4
        assert type(out[0]) is LiteAttentionDisabledConfig
        assert type(out[1]) is LiteAttentionDisabledConfig
        assert type(out[2]) is LiteAttentionRunConfig
        assert out[2].threshold == -5.0
        assert type(out[3]) is LiteAttentionRunConfig
        assert out[3].threshold == -5.0  # clamped last entry


def test_registry_from_model_disabled_steps_with_load(small_qkv, tmp_toml):
    """disabled_steps replaces the first N entries of a loaded ConfigList."""
    q, k, v = small_qkv
    model = SimpleModel()
    names = list(
        LiteAttentionRegistry.from_model(
            model, mode="const", threshold=-1.0
        ).named_modules.keys()
    )
    # Save a 4-step config
    ccd = CalibratedConfigDict(
        {
            name: ConfigList(
                [LiteAttentionRunConfig(threshold=float(-i)) for i in range(4)]
            )
            for name in names
        }
    )
    ccd.save(tmp_toml)

    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(
        model2, mode="load", filename=tmp_toml, disabled_steps=2
    )
    for mod in registry2.named_modules.values():
        for _ in range(4):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()

        out = mod._config_output
        assert len(out) == 4
        assert type(out[0]) is LiteAttentionDisabledConfig
        assert type(out[1]) is LiteAttentionDisabledConfig
        assert type(out[2]) is LiteAttentionRunConfig
        assert out[2].threshold == -2.0
        assert type(out[3]) is LiteAttentionRunConfig
        assert out[3].threshold == -3.0


def test_registry_from_model_disabled_steps_exceeds_list(small_qkv, tmp_toml):
    """When disabled_steps >= list length, last entry is kept as fallback."""
    q, k, v = small_qkv
    model = SimpleModel()
    names = list(
        LiteAttentionRegistry.from_model(
            model, mode="const", threshold=-1.0
        ).named_modules.keys()
    )
    # Save a 2-step config
    ccd = CalibratedConfigDict(
        {
            name: ConfigList(
                [LiteAttentionRunConfig(threshold=float(-i)) for i in range(2)]
            )
            for name in names
        }
    )
    ccd.save(tmp_toml)

    model2 = SimpleModel()
    registry2 = LiteAttentionRegistry.from_model(
        model2, mode="load", filename=tmp_toml, disabled_steps=5
    )
    # ConfigList is 6 elements (5 disabled + 1 fallback)
    for mod in registry2.named_modules.values():
        # Steps 0-5: in bounds
        for _ in range(6):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()

        # Step 6: clamps beyond the 6-element list
        with pytest.warns(UserWarning, match="clamping to last entry"):
            torch.cuda.synchronize()
            mod(q, k, v)
            torch.cuda.synchronize()

        out = mod._config_output
        assert len(out) == 7
        for i in range(5):
            assert type(out[i]) is LiteAttentionDisabledConfig
        # Step 5 onward: fallback to last original entry (threshold=-1.0)
        assert type(out[5]) is LiteAttentionRunConfig
        assert out[5].threshold == -1.0
        assert type(out[6]) is LiteAttentionRunConfig
        assert out[6].threshold == -1.0


def test_mixed_config_list_toml_round_trip(tmp_toml):
    """A ConfigList mixing disabled and run configs survives TOML save/load."""
    original = ConfigList(
        [
            LiteAttentionDisabledConfig(),
            LiteAttentionDisabledConfig(),
            LiteAttentionRunConfig(threshold=-3.0),
            LiteAttentionRunConfig(threshold=-7.0),
        ]
    )
    ccd = CalibratedConfigDict({"attn0": original})
    ccd.save(tmp_toml)

    config_types = [LiteAttentionRunConfig, LiteAttentionDisabledConfig]
    loaded = CalibratedConfigDict.load(tmp_toml, config_types=config_types)

    result = loaded["attn0"]
    assert isinstance(result, ConfigList)
    assert len(result) == 4
    assert type(result[0]) is LiteAttentionDisabledConfig
    assert type(result[1]) is LiteAttentionDisabledConfig
    assert type(result[2]) is LiteAttentionRunConfig
    assert result[2].threshold == -3.0
    assert type(result[3]) is LiteAttentionRunConfig
    assert result[3].threshold == -7.0


def test_registry_from_model_negative_disabled_steps_raises(simple_model):
    with pytest.raises(ValueError, match="non-negative"):
        LiteAttentionRegistry.from_model(
            simple_model, mode="const", threshold=-5.0, disabled_steps=-1
        )
