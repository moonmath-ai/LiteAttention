"""
Base configuration types for the calibration framework.

Defines CalibratedConfig, CalibratedRunConfig, and CalibratedCalibConfig
with serialization (to_dict / from_dict).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass


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
