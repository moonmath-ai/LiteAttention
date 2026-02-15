"""
Configuration and calibration framework for PyTorch modules.

This module provides a system for managing configurable parameters in neural network
modules, supporting both runtime configuration and calibration workflows. It enables:

- Defining typed configuration dataclasses for module parameters
- Switching between calibration mode (finding optimal parameters)
  and run mode (using fixed parameters)
- Centralized configuration management via ModuleRegistry
- Serialization/deserialization of configurations to TOML files

Quick Start
-----------

1. Define a config dataclass for your module::

    @dataclass
    class MyRunConfig(CalibratedRunConfig):
        threshold: float | list[float] | None = None

        @classmethod
        def default(cls):
            return cls(threshold=0.1)

2. Add the ConfigurableModule mixin to your module::

    class MyModule(nn.Module, ConfigurableModule):
        run_config_type = MyRunConfig

        def __init__(self):
            super().__init__()
            ConfigurableModule.__init__(self)

        def forward(self, x):
            cfg = self.config  # timestep-indexed config
            # use cfg.threshold...
            self.add_calibration_results(MyRunConfig(threshold=computed_value))
            return x

3. Set up a registry and configure modules::

    model = MyModel()
    registry = ModuleRegistry(model.named_modules())

    # Option A: Set config for all modules
    registry.set_bulk_config(MyRunConfig(threshold=0.2))

    # Option B: Set config per module
    registry.set_module_config("layer1", MyRunConfig(threshold=0.1))

    # Option C: Load from TOML file
    registry.load_config(Path("config.toml"), config_types=[MyRunConfig])

4. Run inference and save calibration results::

    for timestep in range(num_timesteps):
        output = model(input)
    registry.config_output.save(Path("calibrated.toml"))
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import structlog
import tomli_w
import tomllib

logger = structlog.get_logger()


@dataclass
class CalibratedConfig:
    """
    Base dataclass for all configuration types.

    Provides serialization (to_dict) and deserialization (from_dict).

    Do not subclass directly; use CalibratedRunConfig or CalibratedCalibConfig instead.
    """

    @classmethod
    def from_dict(
        cls, cfg: dict[str, typing.Any], config_types: dict[str, type[CalibratedConfig]]
    ) -> CalibratedConfig:
        """
        Deserialize a config from a dictionary.

        Args:
            cfg: Dictionary with "_type" key indicating the config class name.
            config_types: Mapping from class names to config classes.

        Returns:
            An instance of the appropriate config subclass.

        Raises:
            ValueError: If the _type is not found in config_types.

        """
        _type = cfg.pop("_type")
        _class = config_types.get(_type)
        if _class is None:
            raise ValueError(f"Unknown config type: {_type}")
        assert issubclass(_class, CalibratedConfig)
        return _class(**cfg)

    def to_dict(self) -> dict[str, typing.Any]:
        """Serialize this config to a dictionary with a _type field."""
        return {"_type": type(self).__name__} | self.__dict__


class CalibratedRunConfig(CalibratedConfig):
    """
    Configuration for runtime parameters.

    Subclass this to define your module's runtime parameters. Parameters can be
    single values (applied to all timesteps) or lists (one value per timestep).
    Optionally implement the default() classmethod to provide fallback values.
    """

    @classmethod
    def default(cls) -> typing.Self:
        """
        Return a default configuration.

        Override this to provide fallback values when no config is explicitly set.
        If not implemented, modules without explicit config will raise an error.
        """
        raise NotImplementedError


class CalibratedCalibConfig(CalibratedConfig):
    """
    Configuration for calibration parameters.

    Subclass this to define calibration settings (e.g., target error, metric type).
    During calibration, the module uses these settings to find optimal runtime values,
    which are then saved as a CalibratedRunConfig.

    Example::

        @dataclass
        class MyCalibConfig(CalibratedCalibConfig):
            metric: Literal["l1", "l2"] = "l1"
            target_error: float = 0.01


        # In forward(), check config type to determine mode:
        cfg = self.config
        if isinstance(cfg, MyCalibConfig):
            threshold = self.find_optimal_threshold(cfg.target_error, cfg.metric)
        else:
            threshold = cfg.threshold
    """

    pass


class ConfigList(list[CalibratedConfig]):
    """
    List of config objects, one per timestep.

    Provides methods to convert between list-of-configs and dict-with-lists
    representations.
    """

    def collect(self) -> dict[str, typing.Any]:
        """
        Collapse list of same-type configs to dict with list values.

        Example: [Config(a=1, b=0), Config(a=2, b=0)] → {"_type": "Config", "a": [1, 2], "b": [0, 0]}

        Returns a dict (not a Config) to prevent accidental use as config.

        Raises:
            ValueError: If the list is empty.
            TypeError: If configs have mixed types.
        """
        if not self:
            raise ValueError("Cannot collect empty ConfigList")
        first_type = type(self[0])
        if not all(type(c) is first_type for c in self):
            raise TypeError("Cannot collect configs with mixed types")
        collected: dict[str, str | list[typing.Any]] = {"_type": first_type.__name__}
        for key in self[0].__dict__:
            collected[key] = [cfg.__dict__[key] for cfg in self]
        return collected

    @classmethod
    def explode(
        cls,
        data: dict[str, typing.Any],
        config_types: dict[str, type[CalibratedConfig]],
    ) -> ConfigList:
        """
        Expand dict with list values to list of configs.

        Example: {"_type": "Config", "a": [1, 2], "b": [0, 0]} → [Config(a=1, b=0), Config(a=2, b=0)]

        Args:
            data: Dict with "_type" key and list-valued fields.
            config_types: Mapping from type names to config classes.

        All list-valued fields must have the same length.
        """
        type_name = data["_type"]
        config_type = config_types.get(type_name)
        if config_type is None:
            raise ValueError(f"Unknown config type: {type_name}")
        fields = {k: v for k, v in data.items() if k != "_type"}
        # Find the length from any list-valued field
        length = None
        for value in fields.values():
            if isinstance(value, list):
                if length is None:
                    length = len(value)
                elif len(value) != length:
                    raise ValueError("All list fields must have the same length")
        if length is None:
            # No list fields - return single-element ConfigList
            return cls([config_type(**fields)])
        # Create one config per index
        result = cls()
        for i in range(length):
            new_dict = {
                k: (v[i] if isinstance(v, list) else v) for k, v in fields.items()
            }
            result.append(config_type(**new_dict))
        return result


class CalibratedConfigDict(dict[str, ConfigList | CalibratedConfig]):
    """
    Dictionary mapping module names to their configurations.

    Provides TOML serialization via load() and save() methods.
    Keys are module names (as returned by model.named_modules()), values are
    ConfigList or CalibratedConfig instances.
    """

    @classmethod
    def from_dict(
        cls,
        config: dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]],
        config_types: list[type[CalibratedConfig]],
    ) -> typing.Self:
        """
        Deserialize configs from a nested dictionary.

        Args:
            config: Nested dict mapping module names to config dicts or list of dicts.
                Each config dict must have a "_type" key.
            config_types: List of config classes for deserialization.
        """
        type_map = {ct.__name__: ct for ct in config_types}
        result: dict[str, ConfigList | CalibratedConfig] = {}
        for name, cfg_data in config.items():
            if isinstance(cfg_data, list):
                result[name] = ConfigList(
                    CalibratedConfig.from_dict(dict(cfg), type_map) for cfg in cfg_data
                )
            else:
                result[name] = CalibratedConfig.from_dict(cfg_data, type_map)
        return cls(result)

    @classmethod
    def load(
        cls, filename: Path, config_types: list[type[CalibratedConfig]]
    ) -> typing.Self:
        """
        Load configs from a TOML file.

        Args:
            filename: Path to the TOML file.
            config_types: List of config classes that may appear in the file.

        """
        with filename.open("rb") as f:
            loaded_config = tomllib.load(f)
        return cls.from_dict(loaded_config, config_types=config_types)

    def to_dict(self) -> dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]]:
        """Serialize all configs to a nested dictionary."""
        result: dict[str, list[dict[str, typing.Any]] | dict[str, typing.Any]] = {}
        for name, cfg in self.items():
            if isinstance(cfg, ConfigList):
                result[name] = [c.to_dict() for c in cfg]
            else:
                result[name] = cfg.to_dict()
        return result

    def collect(self) -> dict[str, typing.Any]:
        """
        Collapse all ConfigLists to dicts with list values.

        Example: {"module1": [Config(a=1), Config(a=2)]} → {"module1": {"_type": "Config", "a": [1, 2]}}

        Single configs are converted via to_dict().
        Raises TypeError if any ConfigList has mixed types.
        """
        result = {}
        for name, cfg in self.items():
            if isinstance(cfg, ConfigList):
                result[name] = cfg.collect()
            else:
                result[name] = cfg.to_dict()
        return result

    def save(self, filename: Path) -> None:
        """Save all configs to a TOML file."""
        with filename.open("wb") as f:
            tomli_w.dump(self.to_dict(), f)


class ConfigurableModule:
    """
    Mixin class that adds configuration support to PyTorch modules.

    To use, inherit from both nn.Module and ConfigurableModule, set the
    run_config_type class attribute, and call ConfigurableModule.__init__()
    in your __init__ method. See module docstring for full example.

    Attributes:
        run_config_type: The CalibratedRunConfig subclass for this module.
        config: Property returning the config for the current timestep.
        config_all: Property returning the full config (lists not indexed).

    Config Resolution Order:
        1. Instance config (passed to __init__)
        2. Registry config (set via ModuleRegistry)
        3. Default config (from run_config_type.default())
    """

    run_config_type: type[CalibratedRunConfig] | None = None

    def __init__(self, config: CalibratedConfig | ConfigList | None = None):
        """
        Initialize the configurable module.

        Args:
            config: Optional instance-level config that overrides registry config.
                Can be a single config (applies to all timesteps) or a ConfigList.

        """
        self._instance_config = config
        self.reset_config()
        # set by ModuleRegistry:
        self._registry: ModuleRegistry | None = None
        self.logger = logger.bind(module_id=id(self))

    def reset_config(self) -> None:
        """
        Reset configuration state to prepare for a new run.

        Called automatically by ModuleRegistry on registration.
        """
        self._registry_config: CalibratedConfig | ConfigList | None = None
        self._config_index = 0
        self._warned_messages: set[str] = set()
        self._config_output: ConfigList = ConfigList()
        if self.run_config_type is None:
            warnings.warn(
                f"Module {type(self)} has no run_config_type defined. "
                "Cannot save calibration results.",
                stacklevel=2,
            )
    def restart_config(self):
        if self._config_index == 0:
            assert not self._config_output
            return
        self._config_index = 0
        self._config_output = ConfigList()
        if (
           isinstance(self.config, CalibratedCalibConfig)
           or isinstance(self.config_all, ConfigList)
           and any(isinstance(c, CalibratedCalibConfig) for c in self.config_all)
        ):
            warnings.warn(
                "Using restart_config() with a calibration config; data will be lost.",
                stacklevel=2,
            )

    @property
    def module_name(self) -> str | None:
        """Get the registered name of this module, or None if unregistered."""
        if self._registry is None:
            return None
        return self._registry.id_to_name[id(self)]

    @property
    def config_all(self) -> CalibratedConfig | ConfigList:
        """
        Get the full config (ConfigList or single config).

        Resolution order: instance config > registry config > default config.
        """
        if self._instance_config is not None:
            # self._instance_config overrides registry config, but we warn about it
            if self._registry is None:
                warnings.warn(
                    "Module has no registry. Using local config.", stacklevel=2
                )
            elif self._registry_config is None:
                warnings.warn(
                    "Module has no registry config. Using local config.", stacklevel=2
                )
            else:
                warnings.warn(
                    "Module has both local config and registry config. "
                    "Using local config.",
                    stacklevel=2,
                )
            return self._instance_config
        if self._registry is None:
            warnings.warn(
                "Module has no registry or local config. Using default config.",
                stacklevel=2,
            )
        elif self._registry_config is None:
            warnings.warn(
                "Module has no registry config or local config. Using default config.",
                stacklevel=2,
            )
        else:
            return self._registry_config

        if self.run_config_type is None:
            raise ValueError(f"Module {type(self)} has no run_config_type defined.")
        return self.run_config_type.default()

    @property
    def config(self) -> CalibratedConfig:
        """
        Get the config for the current timestep.

        If config_all is a ConfigList, returns the config at _config_index.
        If config_all is a single config, returns it directly (same for all timesteps).

        This is the primary way to access config in forward().
        """
        cfg = self.config_all
        if isinstance(cfg, ConfigList):
            return cfg[self._config_index]
        else:
            return cfg

    def add_calibration_results(self, results: CalibratedRunConfig) -> None:
        """
        Record calibration results and advance the timestep index.

        Must be called exactly once per forward pass (even when not calibrating).

        Args:
            results: A RunConfig instance for this timestep.

        """
        self._config_index += 1
        if self.run_config_type is None:
            return
        if self._registry is None and isinstance(self.config, CalibratedCalibConfig):
            warnings.warn(
                "Module has no registry. Cannot save calibration results.", stacklevel=2
            )
        if not isinstance(results, self.run_config_type):
            raise TypeError(
                f"Results type {type(results)} does not match "
                f"module run_config_type {self.run_config_type}."
            )
        self._config_output.append(results)


class ModuleRegistry:
    """
    Central registry for managing configurations across all ConfigurableModules.

    Creates a mapping from module names to modules, and provides methods to set
    configs in bulk or per-module, as well as load/save configs to TOML files.
    See module docstring for full example.

    Attributes:
        named_modules: Dict mapping module names to ConfigurableModule instances.
        config: Property returning current input configs for all modules.
        config_output: Property returning calibration results for all modules.
    """

    def __init__(
        self, named_modules: Iterator[tuple[str, ConfigurableModule | typing.Any]]
    ):
        """
        Create a registry from a model's named_modules().

        Args:
            named_modules: Iterator of (name, module) pairs, typically from
                model.named_modules(). Non-ConfigurableModule entries are ignored.

        Note:
            The model must not add or remove ConfigurableModules after creation.

        """
        self.named_modules = {
            name: module
            for name, module in named_modules
            if isinstance(module, ConfigurableModule)
        }
        self.id_to_name = {
            id(module): name for name, module in self.named_modules.items()
        }
        for module in self.named_modules.values():
            module._registry = self
            module.reset_config()

    def set_bulk_config(self, config: CalibratedConfig | ConfigList) -> None:
        """Set the same config (or config list) for all registered modules."""
        for module in self.named_modules.values():
            module._registry_config = config

    def set_module_config(
        self, name: str, config: CalibratedConfig | ConfigList
    ) -> None:
        """Set config for a specific module by name."""
        module = self.named_modules[name]
        module._registry_config = config

    def load_config(
        self, filename: Path, config_types: list[type[CalibratedConfig]]
    ) -> None:
        """
        Load configs from a TOML file and apply to modules.

        Args:
            filename: Path to the TOML config file.
            config_types: List of config classes that may appear in the file.

        """
        loaded_config = CalibratedConfigDict.load(filename, config_types=config_types)
        for name, cfg in loaded_config.items():
            self.set_module_config(name, cfg)

    @property
    def config(self) -> CalibratedConfigDict:
        """
        Get the current input configs for all modules.

        Returns a dict mapping module names to their config_all (full configs
        with list values, not timestep-indexed). Useful for inspecting or
        saving the input configuration.
        """
        return CalibratedConfigDict(
            {name: module.config_all for name, module in self.named_modules.items()}
        )

    @property
    def config_output(self) -> CalibratedConfigDict:
        """
        Get the accumulated calibration results for all modules.

        Returns a dict mapping module names to their _config_output, which
        contains list values accumulated via add_calibration_results().
        Use .save() on the result to persist calibration results to TOML.
        """
        return CalibratedConfigDict(
            {
                name: module._config_output
                for name, module in self.named_modules.items()
            }
        )
