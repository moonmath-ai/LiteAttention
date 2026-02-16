"""CPU-only tests for the calibration/configuration framework (hopper/calibrated_module.py).

This file imports calibrated_module directly (not via hopper/__init__.py) to avoid
pulling in lite_attention.py and its CUDA extension dependency.
"""

import importlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch.nn as nn

# Import calibrated_module directly from file path to avoid hopper/__init__.py
# which eagerly imports the CUDA-dependent lite_attention module.
_spec = importlib.util.spec_from_file_location(
    "hopper.calibrated_module",
    Path(__file__).resolve().parent.parent / "hopper" / "calibrated_module.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

CalibratedCalibConfig = _mod.CalibratedCalibConfig
CalibratedConfig = _mod.CalibratedConfig
CalibratedConfigDict = _mod.CalibratedConfigDict
CalibratedRunConfig = _mod.CalibratedRunConfig
ConfigList = _mod.ConfigList
ConfigurableModule = _mod.ConfigurableModule
ModuleRegistry = _mod.ModuleRegistry


# ---------------------------------------------------------------------------
# Dummy classes
# ---------------------------------------------------------------------------


@dataclass
class DummyRunConfig(CalibratedRunConfig):
    threshold: float = -10.0

    @classmethod
    def default(cls):
        return cls(threshold=-10.0)


@dataclass
class DummyCalibConfig(CalibratedCalibConfig):
    metric: str = "L1"
    target_error: float = 0.01


@dataclass
class NoDefaultRunConfig(CalibratedRunConfig):
    """A run config that does NOT implement default() — used to test NotImplementedError."""

    value: int = 0


class DummyModule(nn.Module, ConfigurableModule):
    run_config_type = DummyRunConfig

    def __init__(self, config=None):
        nn.Module.__init__(self)
        ConfigurableModule.__init__(self, config)


class DummyModel(nn.Module):
    def __init__(self, n_layers=3, config=None):
        super().__init__()
        self.layers = nn.ModuleList([DummyModule(config=config) for _ in range(n_layers)])
        self.proj = nn.Linear(4, 4)  # non-configurable module


CONFIG_TYPES = {
    "DummyRunConfig": DummyRunConfig,
    "DummyCalibConfig": DummyCalibConfig,
}
CONFIG_TYPE_LIST = [DummyRunConfig, DummyCalibConfig]


# ===========================================================================
# Config serialization
# ===========================================================================


def test_run_config_to_dict_from_dict_roundtrip():
    cfg = DummyRunConfig(threshold=-5.0)
    d = cfg.to_dict()
    assert d["_type"] == "DummyRunConfig"
    assert d["threshold"] == -5.0
    restored = CalibratedConfig.from_dict(dict(d), CONFIG_TYPES)
    assert isinstance(restored, DummyRunConfig)
    assert restored.threshold == -5.0


def test_calib_config_to_dict_from_dict_roundtrip():
    cfg = DummyCalibConfig(metric="RMSE", target_error=0.05)
    d = cfg.to_dict()
    assert d["_type"] == "DummyCalibConfig"
    restored = CalibratedConfig.from_dict(dict(d), CONFIG_TYPES)
    assert isinstance(restored, DummyCalibConfig)
    assert restored.metric == "RMSE"
    assert restored.target_error == 0.05


def test_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown config type"):
        CalibratedConfig.from_dict({"_type": "Nonexistent"}, CONFIG_TYPES)


def test_default_works():
    cfg = DummyRunConfig.default()
    assert cfg.threshold == -10.0


def test_default_not_implemented():
    with pytest.raises(NotImplementedError):
        NoDefaultRunConfig.default()


# ===========================================================================
# ConfigList
# ===========================================================================


def test_config_list_collect():
    cl = ConfigList([DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)])
    collected = cl.collect()
    assert collected["_type"] == "DummyRunConfig"
    assert collected["threshold"] == [-1.0, -2.0]


def test_config_list_explode():
    data = {"_type": "DummyRunConfig", "threshold": [-1.0, -2.0]}
    cl = ConfigList.explode(data, CONFIG_TYPES)
    assert len(cl) == 2
    assert cl[0].threshold == -1.0
    assert cl[1].threshold == -2.0


def test_config_list_collect_explode_roundtrip():
    original = ConfigList([DummyRunConfig(threshold=i * -1.0) for i in range(5)])
    collected = original.collect()
    restored = ConfigList.explode(collected, CONFIG_TYPES)
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        assert a.threshold == b.threshold


def test_config_list_empty_raises():
    with pytest.raises(ValueError, match="Cannot collect empty"):
        ConfigList().collect()


def test_config_list_mixed_types_raises():
    cl = ConfigList([DummyRunConfig(threshold=-1.0), DummyCalibConfig(metric="L1")])
    with pytest.raises(TypeError, match="mixed types"):
        cl.collect()


def test_config_list_mismatched_list_lengths_raises():
    # explode should raise when list fields have different lengths
    data = {"_type": "DummyCalibConfig", "metric": ["L1", "RMSE"], "target_error": [0.01]}
    with pytest.raises(ValueError, match="same length"):
        ConfigList.explode(data, CONFIG_TYPES)


def test_config_list_no_list_fields():
    data = {"_type": "DummyRunConfig", "threshold": -5.0}
    cl = ConfigList.explode(data, CONFIG_TYPES)
    assert len(cl) == 1
    assert cl[0].threshold == -5.0


def test_config_list_explode_unknown_type_raises():
    data = {"_type": "Nonexistent", "foo": [1, 2]}
    with pytest.raises(ValueError, match="Unknown config type"):
        ConfigList.explode(data, CONFIG_TYPES)


# ===========================================================================
# CalibratedConfigDict
# ===========================================================================


def test_calibrated_config_dict_to_dict_from_dict_roundtrip():
    ccd = CalibratedConfigDict(
        {
            "layer0": DummyRunConfig(threshold=-3.0),
            "layer1": DummyCalibConfig(metric="RMSE", target_error=0.1),
        }
    )
    d = ccd.to_dict()
    restored = CalibratedConfigDict.from_dict(d, CONFIG_TYPE_LIST)
    assert isinstance(restored["layer0"], DummyRunConfig)
    assert restored["layer0"].threshold == -3.0
    assert isinstance(restored["layer1"], DummyCalibConfig)
    assert restored["layer1"].metric == "RMSE"


def test_calibrated_config_dict_collect():
    ccd = CalibratedConfigDict(
        {
            "m1": ConfigList([DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)]),
            "m2": DummyCalibConfig(metric="L1"),
        }
    )
    collected = ccd.collect()
    assert collected["m1"]["threshold"] == [-1.0, -2.0]
    assert collected["m2"]["_type"] == "DummyCalibConfig"


def test_calibrated_config_dict_toml_roundtrip(tmp_path):
    ccd = CalibratedConfigDict(
        {
            "layer0": DummyRunConfig(threshold=-3.0),
            "layer1": DummyRunConfig(threshold=-7.0),
        }
    )
    path = tmp_path / "config.toml"
    ccd.save(path)
    loaded = CalibratedConfigDict.load(path, CONFIG_TYPE_LIST)
    assert isinstance(loaded["layer0"], DummyRunConfig)
    assert loaded["layer0"].threshold == -3.0
    assert loaded["layer1"].threshold == -7.0


def test_calibrated_config_dict_toml_roundtrip_with_config_lists(tmp_path):
    """Save ConfigLists via collect(), reload via from_dict with list values."""
    ccd = CalibratedConfigDict(
        {
            "m": ConfigList([DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)]),
        }
    )
    path = tmp_path / "config.toml"
    # Save collected form (dict with list values)
    import tomli_w

    with path.open("wb") as f:
        tomli_w.dump(ccd.collect(), f)

    # Load back — the TOML has dict-with-lists, not list-of-dicts
    import tomllib

    with path.open("rb") as f:
        raw = tomllib.load(f)
    # Reconstruct via explode
    result = {}
    for name, data in raw.items():
        result[name] = ConfigList.explode(data, CONFIG_TYPES)
    assert len(result["m"]) == 2
    assert result["m"][0].threshold == -1.0
    assert result["m"][1].threshold == -2.0


def test_calibrated_config_dict_from_dict_with_list_of_dicts():
    raw = {
        "mod": [
            {"_type": "DummyRunConfig", "threshold": -1.0},
            {"_type": "DummyRunConfig", "threshold": -2.0},
        ]
    }
    ccd = CalibratedConfigDict.from_dict(raw, CONFIG_TYPE_LIST)
    assert isinstance(ccd["mod"], ConfigList)
    assert len(ccd["mod"]) == 2


# ===========================================================================
# ConfigurableModule
# ===========================================================================


def test_config_resolution_instance_over_registry():
    instance_cfg = DummyRunConfig(threshold=-1.0)
    mod = DummyModule(config=instance_cfg)
    registry_cfg = DummyRunConfig(threshold=-99.0)
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = registry_cfg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mod.config.threshold == -1.0  # instance wins


def test_config_resolution_registry_over_default():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    registry_cfg = DummyRunConfig(threshold=-5.0)
    mod._registry_config = registry_cfg
    assert mod.config.threshold == -5.0


def test_config_resolution_falls_to_default():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    # No registry config set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = mod.config
    assert isinstance(cfg, DummyRunConfig)
    assert cfg.threshold == -10.0  # default


def test_config_indexes_into_config_list():
    cl = ConfigList([DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)])
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = cl

    assert mod.config.threshold == -1.0  # index 0
    mod._config_index = 1
    assert mod.config.threshold == -2.0  # index 1


def test_add_calibration_results_advances_index():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    assert mod._config_index == 0
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    assert len(mod._config_output) == 1
    assert mod._config_output[0].threshold == -3.0


def test_add_calibration_results_wrong_type_raises():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    with pytest.raises(TypeError, match="does not match"):
        mod.add_calibration_results(DummyCalibConfig(metric="L1"))


def test_reset_config():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    mod.reset_config()
    assert mod._config_index == 0
    assert len(mod._config_output) == 0


def test_restart_config():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    mod.restart_config()
    assert mod._config_index == 0
    assert len(mod._config_output) == 0


def test_module_name_property():
    mod = DummyModule()
    assert mod.module_name is None  # no registry
    registry = ModuleRegistry(iter([("my_module", mod)]))
    assert mod.module_name == "my_module"


def test_warning_no_registry():
    mod = DummyModule()
    with pytest.warns(match="no registry or local config"):
        _ = mod.config_all


def test_warning_no_registry_config():
    mod = DummyModule()
    registry = ModuleRegistry(iter([("mod", mod)]))
    # registry set but no _registry_config
    with pytest.warns(match="no registry config or local config"):
        _ = mod.config_all


def test_warning_instance_overrides_registry():
    mod = DummyModule(config=DummyRunConfig(threshold=-1.0))
    registry = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-99.0)
    with pytest.warns(match="Using local config"):
        _ = mod.config_all


# ===========================================================================
# ModuleRegistry
# ===========================================================================


def test_registry_filters_non_configurable():
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    # Should contain the 2 DummyModule layers but NOT nn.Linear or the model itself
    for name, mod in registry.named_modules.items():
        assert isinstance(mod, ConfigurableModule)
    assert len(registry.named_modules) == 2


def test_registry_set_bulk_config():
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    cfg = DummyRunConfig(threshold=-7.0)
    registry.set_bulk_config(cfg)
    for mod in registry.named_modules.values():
        assert mod._registry_config is cfg


def test_registry_set_module_config():
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    names = list(registry.named_modules.keys())
    cfg = DummyRunConfig(threshold=-3.0)
    registry.set_module_config(names[0], cfg)
    assert registry.named_modules[names[0]]._registry_config is cfg
    assert registry.named_modules[names[1]]._registry_config is not cfg


def test_registry_set_module_config_bad_name_raises():
    model = DummyModel(n_layers=1)
    registry = ModuleRegistry(model.named_modules())
    with pytest.raises(KeyError):
        registry.set_module_config("nonexistent_module", DummyRunConfig())


def test_registry_load_config_from_toml(tmp_path):
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    names = list(registry.named_modules.keys())

    ccd = CalibratedConfigDict(
        {
            names[0]: DummyRunConfig(threshold=-1.0),
            names[1]: DummyRunConfig(threshold=-2.0),
        }
    )
    path = tmp_path / "config.toml"
    ccd.save(path)

    registry.load_config(path, config_types=CONFIG_TYPE_LIST)
    assert registry.named_modules[names[0]]._registry_config.threshold == -1.0
    assert registry.named_modules[names[1]]._registry_config.threshold == -2.0


def test_registry_config_property():
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    cfg = DummyRunConfig(threshold=-5.0)
    registry.set_bulk_config(cfg)
    result = registry.config
    assert isinstance(result, CalibratedConfigDict)
    for name in registry.named_modules:
        assert result[name].threshold == -5.0


def test_registry_config_output_property():
    model = DummyModel(n_layers=1)
    registry = ModuleRegistry(model.named_modules())
    name = list(registry.named_modules.keys())[0]
    mod = registry.named_modules[name]
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    result = registry.config_output
    assert isinstance(result, CalibratedConfigDict)
    assert len(result[name]) == 1
    assert result[name][0].threshold == -3.0
