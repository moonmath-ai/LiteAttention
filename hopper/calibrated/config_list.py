"""
List of configs (one per timestep) with collect/explode for list-dict conversion.
"""

from __future__ import annotations

import typing

from .config import CalibratedConfig


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
