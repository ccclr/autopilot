from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class _FeatureSpec:
    key: str
    kind: str


class ActionFeatureEncoder:
    """Encode the discrete Autobahn arm catalog as structured features.

    Ordered protocol parameters are min-max normalized over the current arm
    catalog.  ``cut_condition_type`` is kept categorical because treating its
    integer labels as a smooth quantity would impose an unjustified distance.
    """

    VERSION = 1
    _SPECS = (
        _FeatureSpec("batch_size", "ordered"),
        _FeatureSpec("header_size", "ordered"),
        _FeatureSpec("cut_condition_type", "categorical"),
        _FeatureSpec("fast_path_timeout", "ordered"),
        _FeatureSpec("k", "ordered"),
    )

    def __init__(self, arms: Sequence[str]) -> None:
        if not arms:
            raise ValueError("action feature encoder requires at least one arm")

        self.arms = tuple(str(arm) for arm in arms)
        parsed = tuple(self._parse_arm(arm) for arm in self.arms)
        self._value_sets = {
            spec.key: tuple(sorted({values[spec.key] for values in parsed}))
            for spec in self._SPECS
        }

        feature_names: list[str] = []
        for spec in self._SPECS:
            if spec.kind == "ordered":
                feature_names.append(f"{spec.key}_norm")
            else:
                feature_names.extend(
                    f"{spec.key}={value}" for value in self._value_sets[spec.key]
                )
        self.feature_names = tuple(feature_names)
        self.features = np.stack(
            [self._encode_values(values) for values in parsed]
        ).astype(np.float32, copy=False)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def schema_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "specs": [(spec.key, spec.kind) for spec in self._SPECS],
            "value_sets": {
                key: list(values) for key, values in self._value_sets.items()
            },
            "feature_names": list(self.feature_names),
        }

    def _encode_values(self, values: dict[str, int]) -> np.ndarray:
        features: list[float] = []
        for spec in self._SPECS:
            value = values[spec.key]
            choices = self._value_sets[spec.key]
            if spec.kind == "ordered":
                lower = float(choices[0])
                upper = float(choices[-1])
                normalized = (
                    0.0
                    if upper == lower
                    else (value - lower) / (upper - lower)
                )
                features.append(float(normalized))
            else:
                features.extend(1.0 if value == choice else 0.0 for choice in choices)
        return np.asarray(features, dtype=np.float32)

    @classmethod
    def _parse_arm(cls, arm: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for part in arm.split(","):
            if "=" not in part:
                raise ValueError(f"invalid DQN arm component: {part!r}")
            key, raw_value = part.split("=", 1)
            if key in values:
                raise ValueError(f"duplicate DQN arm key: {key!r}")
            try:
                values[key] = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"DQN arm value must be an integer: {part!r}"
                ) from error

        required = {spec.key for spec in cls._SPECS}
        missing = required.difference(values)
        unexpected = set(values).difference(required)
        if missing or unexpected:
            raise ValueError(
                "DQN arm schema mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return values
