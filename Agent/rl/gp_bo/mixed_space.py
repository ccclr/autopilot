"""
Mixed action space helpers for GP-BO / KernelUCB.

Discrete dims are enumerated; fast_path_timeout is continuous in
codec.fast_path_timeout_ms_bounds and rounded to int ms when emitted.
"""

from __future__ import annotations

import itertools
from typing import Any, List, Sequence, Tuple

from actions.action_encode import ActionCodec

Arm = str

# Keys in arm id / decode_arm dict (matches ArmCatalog).
ARM_KEYS = [
    "batch_size",
    "header_size",
    "cut_condition_type",
    "fast_path_timeout",
    "k",
]
TIMEOUT_KEY = "fast_path_timeout"
TIMEOUT_KEY_INDEX = ARM_KEYS.index(TIMEOUT_KEY)


def encode_arm_values(values: Sequence[float | int]) -> Arm:
    parts = []
    for key, value in zip(ARM_KEYS, values):
        if key == TIMEOUT_KEY:
            parts.append(f"{key}={int(round(float(value)))}")
        else:
            parts.append(f"{key}={int(value)}")
    return ",".join(parts)


def decode_arm_values(arm_id: Arm) -> Tuple[float, ...]:
    values = {}
    for part in arm_id.split(","):
        key, value = part.split("=", 1)
        values[key] = float(value)
    return tuple(values[key] for key in ARM_KEYS)


def decode_arm_params(arm_id: Arm) -> dict[str, Any]:
    values = decode_arm_values(arm_id)
    params = dict(zip(ARM_KEYS, values))
    params[TIMEOUT_KEY] = int(round(params[TIMEOUT_KEY]))
    for key in ARM_KEYS:
        if key != TIMEOUT_KEY:
            params[key] = int(params[key])
    return params


class MixedActionSpace:
    """
    Discrete bases × continuous timeout.

    A "base" is a full arm string with timeout set to the lower bound (placeholder).
    """

    def __init__(self, codec: ActionCodec | None = None):
        self.codec = codec or ActionCodec()
        bounds = getattr(self.codec, "fast_path_timeout_ms_bounds", None)
        if bounds is None:
            vals = self.codec.fast_path_timeout_ms_values
            bounds = (min(vals), max(vals))
        self.timeout_lo = float(bounds[0])
        self.timeout_hi = float(bounds[1])
        if self.timeout_hi < self.timeout_lo:
            raise ValueError("fast_path_timeout_ms_bounds must be (lo, hi) with lo <= hi")
        # Search bound may be pruned; keep timeout_hi for feature normalization.
        self.timeout_search_hi = self.timeout_hi
        self.timeout_search_lo = self.timeout_lo

        discrete_sets = [
            self.codec.batch_size_values,
            self.codec.header_size_values,
            self.codec.cut_condition_type_values,
            self.codec.parallel_proposals_values,
        ]
        self._bases: List[Tuple[int, int, int, int]] = list(itertools.product(*discrete_sets))
        # Placeholder arms (timeout = lo) for catalog-style APIs.
        self._placeholder_arms = [
            encode_arm_values((b, h, c, self.timeout_lo, k)) for (b, h, c, k) in self._bases
        ]

    def list_bases(self) -> List[Tuple[int, int, int, int]]:
        return list(self._bases)

    def list_placeholder_arms(self) -> List[Arm]:
        return list(self._placeholder_arms)

    def set_timeout_search_hi(self, cap: float | None) -> None:
        if cap is None:
            self.timeout_search_hi = self.timeout_hi
            return
        self.timeout_search_hi = float(
            np_clip(float(cap), self.timeout_lo, self.timeout_hi)
        )

    def set_timeout_search_lo(self, bound: float | None) -> None:
        # Emitted timeouts are integer milliseconds. Do not clamp an empty
        # range back onto the excluded boundary.
        self.timeout_search_lo = self.timeout_lo if bound is None else max(
            self.timeout_lo, float(bound) + 1.0
        )

    @property
    def timeout_search_empty(self) -> bool:
        return self.timeout_search_lo > self.timeout_search_hi

    def project_timeout(self, timeout_ms: float) -> float:
        if self.timeout_search_lo == self.timeout_lo:
            return float(timeout_ms)
        return float(np_clip(timeout_ms, self.timeout_search_lo, self.timeout_search_hi))

    def make_arm(self, base: Tuple[int, int, int, int], timeout_ms: float) -> Arm:
        if self.timeout_search_empty:
            raise ValueError("No selectable timeout remains after RTT filtering")
        b, h, c, k = base
        timeout = float(np_clip(timeout_ms, self.timeout_search_lo, self.timeout_search_hi))
        timeout = self.project_timeout(timeout)
        return encode_arm_values((b, h, c, timeout, k))

    def normalize_timeout(self, timeout_ms: float) -> float:
        span = self.timeout_hi - self.timeout_lo
        if span <= 0:
            return 0.0
        return (float(np_clip(timeout_ms, self.timeout_lo, self.timeout_hi)) - self.timeout_lo) / span

    def denormalize_timeout(self, unit: float) -> float:
        u = float(np_clip(unit, 0.0, 1.0))
        return self.timeout_search_lo + u * (self.timeout_search_hi - self.timeout_search_lo)


def np_clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


class MixedArmCatalog:
    """
    Drop-in decode helper for CMABTrainer: decode_arm(arm_id) -> params dict.
    list_arms() returns placeholder arms (not the full continuous set).
    """

    def __init__(self, space: MixedActionSpace):
        self.space = space
        self.codec = space.codec

    def list_arms(self) -> List[Arm]:
        return self.space.list_placeholder_arms()

    def decode_arm(self, arm: Arm) -> dict[str, Any]:
        return decode_arm_params(arm)
