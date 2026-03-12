"""Tests for the calibration/configuration framework (calibrated_module)."""

import copy
from dataclasses import dataclass

import pytest
import torch.nn as nn
from lite_attention.calibrated_module import (
    CalibratedCalibConfig,
    CalibratedConfig,
    CalibratedConfigDict,
    CalibratedRunConfig,
    ConfigList,
    ConfigurableModule,
    ModuleRegistry,
)

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
        self.layers = nn.ModuleList(
            [DummyModule(config=config) for _ in range(n_layers)]
        )
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


def test_from_dict_does_not_mutate_input():
    original = {"_type": "DummyRunConfig", "threshold": -5.0}
    original_copy = dict(original)
    CalibratedConfig.from_dict(original, CONFIG_TYPES)
    assert original == original_copy


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
    data = {
        "_type": "DummyCalibConfig",
        "metric": ["L1", "RMSE"],
        "target_error": [0.01],
    }
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


def test_calibrated_config_dict_from_dict_does_not_mutate_raw():
    raw = {"mod": {"_type": "DummyRunConfig", "threshold": -3.0}}
    raw_copy = copy.deepcopy(raw)
    CalibratedConfigDict.from_dict(raw, CONFIG_TYPE_LIST)
    assert raw == raw_copy


def test_calibrated_config_dict_collect():
    ccd = CalibratedConfigDict(
        {
            "m1": ConfigList(
                [DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)]
            ),
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


def test_calibrated_config_dict_save_load_roundtrip_with_config_lists(tmp_path):
    """Test actual save() -> load() roundtrip with ConfigLists (via to_dict)."""
    ccd = CalibratedConfigDict(
        {
            "m": ConfigList(
                [DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)]
            ),
        }
    )
    path = tmp_path / "config.toml"
    ccd.save(path)
    loaded = CalibratedConfigDict.load(path, CONFIG_TYPE_LIST)
    assert isinstance(loaded["m"], ConfigList)
    assert len(loaded["m"]) == 2
    assert loaded["m"][0].threshold == -1.0
    assert loaded["m"][1].threshold == -2.0


def test_calibrated_config_dict_toml_roundtrip_with_collect(tmp_path):
    """Save ConfigLists via collect(), reload via explode."""
    ccd = CalibratedConfigDict(
        {
            "m": ConfigList(
                [DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)]
            ),
        }
    )
    path = tmp_path / "config.toml"
    import tomli_w

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with path.open("wb") as f:
        tomli_w.dump(ccd.collect(), f)

    with path.open("rb") as f:
        raw = tomllib.load(f)
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


@pytest.mark.filterwarnings("ignore:Module has both local config")
def test_config_resolution_instance_over_registry():
    instance_cfg = DummyRunConfig(threshold=-1.0)
    mod = DummyModule(config=instance_cfg)
    registry_cfg = DummyRunConfig(threshold=-99.0)
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = registry_cfg
    assert mod.config.threshold == -1.0  # instance wins


def test_config_resolution_registry_over_default():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    registry_cfg = DummyRunConfig(threshold=-5.0)
    mod._registry_config = registry_cfg
    assert mod.config.threshold == -5.0


@pytest.mark.filterwarnings("ignore:Module has no registry config")
def test_config_resolution_falls_to_default():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    cfg = mod.config
    assert isinstance(cfg, DummyRunConfig)
    assert cfg.threshold == -10.0  # default


def test_config_indexes_into_config_list():
    cl = ConfigList([DummyRunConfig(threshold=-1.0), DummyRunConfig(threshold=-2.0)])
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = cl

    assert mod.config.threshold == -1.0  # index 0
    mod._config_index = 1
    assert mod.config.threshold == -2.0  # index 1


def test_config_index_out_of_bounds_on_config_list():
    cl = ConfigList([DummyRunConfig(threshold=-1.0)])
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = cl
    assert mod.config.threshold == -1.0  # index 0
    mod._config_index = 1  # beyond list
    with pytest.raises(IndexError):
        _ = mod.config


def test_add_calibration_results_advances_index():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    assert mod._config_index == 0
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    assert len(mod._config_output) == 1
    assert mod._config_output[0].threshold == -3.0


def test_add_calibration_results_multi_step():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    thresholds = [-3.0, -4.0, -5.0]
    for th in thresholds:
        mod.add_calibration_results(DummyRunConfig(threshold=th))
    assert mod._config_index == 3
    assert len(mod._config_output) == 3
    for i, th in enumerate(thresholds):
        assert mod._config_output[i].threshold == th


def test_add_calibration_results_wrong_type_raises():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    with pytest.raises(TypeError, match="does not match"):
        mod.add_calibration_results(DummyCalibConfig(metric="L1"))


def test_reset_config():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    mod.reset_config()
    assert mod._config_index == 0
    assert len(mod._config_output) == 0


def test_restart_config():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    assert mod._config_index == 1
    mod.restart_config()
    assert mod._config_index == 0
    assert len(mod._config_output) == 0


def test_restart_config_warns_on_calib_config():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))  # registers mod
    mod._registry_config = DummyCalibConfig(metric="L1", target_error=0.01)
    mod._config_index = 1
    mod._config_output = ConfigList([DummyRunConfig(threshold=-5.0)])
    with pytest.warns(UserWarning, match="calibration config.*data will be lost"):
        mod.restart_config()


def test_module_name_property():
    mod = DummyModule()
    assert mod.module_name is None  # no registry
    _ = ModuleRegistry(iter([("my_module", mod)]))
    assert mod.module_name == "my_module"


def test_warning_no_registry():
    mod = DummyModule()
    with pytest.warns(UserWarning, match="no registry or local config"):
        _ = mod.config_all


def test_warning_no_registry_config():
    mod = DummyModule()
    _ = ModuleRegistry(iter([("mod", mod)]))
    with pytest.warns(UserWarning, match="no registry config or local config"):
        _ = mod.config_all


def test_warning_instance_overrides_registry():
    mod = DummyModule(config=DummyRunConfig(threshold=-1.0))
    _ = ModuleRegistry(iter([("mod", mod)]))
    mod._registry_config = DummyRunConfig(threshold=-99.0)
    with pytest.warns(UserWarning, match="Using local config"):
        _ = mod.config_all


# ===========================================================================
# ModuleRegistry
# ===========================================================================


def test_registry_filters_non_configurable():
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
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


def test_registry_load_config_extra_module_raises(tmp_path):
    model = DummyModel(n_layers=1)
    registry = ModuleRegistry(model.named_modules())
    ccd = CalibratedConfigDict({"nonexistent_module": DummyRunConfig(threshold=-1.0)})
    path = tmp_path / "config.toml"
    ccd.save(path)
    with pytest.raises(KeyError):
        registry.load_config(path, config_types=CONFIG_TYPE_LIST)


def test_registry_load_config_partial_modules(tmp_path):
    model = DummyModel(n_layers=2)
    registry = ModuleRegistry(model.named_modules())
    names = list(registry.named_modules.keys())
    ccd = CalibratedConfigDict({names[0]: DummyRunConfig(threshold=-1.0)})
    path = tmp_path / "config.toml"
    ccd.save(path)
    registry.load_config(path, config_types=CONFIG_TYPE_LIST)
    assert registry.named_modules[names[0]]._registry_config.threshold == -1.0
    assert registry.named_modules[names[1]]._registry_config is None


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


def test_config_output_save_roundtrip(tmp_path):
    model = DummyModel(n_layers=1)
    registry = ModuleRegistry(model.named_modules())
    name = list(registry.named_modules.keys())[0]
    mod = registry.named_modules[name]
    mod._registry_config = DummyRunConfig(threshold=-5.0)
    mod.add_calibration_results(DummyRunConfig(threshold=-3.0))
    mod.add_calibration_results(DummyRunConfig(threshold=-4.0))

    path = tmp_path / "output.toml"
    registry.config_output.save(path)

    loaded = CalibratedConfigDict.load(path, CONFIG_TYPE_LIST)
    assert isinstance(loaded[name], ConfigList)
    assert len(loaded[name]) == 2
    assert loaded[name][0].threshold == -3.0
    assert loaded[name][1].threshold == -4.0


def test_creating_second_registry_resets_config():
    model = DummyModel(n_layers=1)
    registry1 = ModuleRegistry(model.named_modules())
    registry1.set_bulk_config(DummyRunConfig(threshold=-5.0))
    name = list(registry1.named_modules.keys())[0]
    assert registry1.named_modules[name]._registry_config.threshold == -5.0

    # Creating a new registry should clear the config
    registry2 = ModuleRegistry(model.named_modules())
    assert registry2.named_modules[name]._registry_config is None
