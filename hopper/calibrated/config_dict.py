"""
Dictionary of module name -> config(s) with TOML load/save.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .config import CalibratedConfig
from .config_list import ConfigList


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
