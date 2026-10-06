from __future__ import annotations

import itertools
import random
from typing import Any, List, Tuple

from actions.action_encode import ActionCodec

Arm = str


class ArmCatalog:
    def __init__(self, codec: ActionCodec | None = None, max_arms: int | None = None, seed: int = 0, reserve_zero_timeout: bool = False):
        self.codec = codec or ActionCodec()
        self.action_dims = list(self.codec.action_dims)
        self._arm_keys = [
            "batch_size",
            "header_size",
            "cut_condition_type",
            "fast_path_timeout",
            "k",
            # "use_optimistic_tips",
        ]
        # use_optimistic_tips: 1=True, 0=False for arm encoding (int required by _decode_arm)
        self._use_optimistic_tips_arm_values = [1, 0]
        self._value_sets = [
            self.codec.batch_size_values,
            self.codec.header_size_values,
            self.codec.cut_condition_type_values,
            self.codec.fast_path_timeout_ms_values,
            self.codec.parallel_proposals_values,
            # self._use_optimistic_tips_arm_values,
        ]
        self._arms = self._build_arms(max_arms=max_arms, seed=seed)
        if reserve_zero_timeout:
            # Reserve fallback features before fixed action encodings/models
            # are initialized, including sampled and default-only catalogs.
            existing = {arm_id for arm_id, _ in self._arms}
            for _, values in list(self._arms):
                zero_values = (*values[:3], 0, *values[4:])
                zero_arm = self._encode_arm(zero_values)
                if zero_arm not in existing:
                    self._arms.append((zero_arm, zero_values))
                    existing.add(zero_arm)
        self._arm_lookup = {arm_id: arm_tuple for arm_id, arm_tuple in self._arms}

    def _build_arms(self, max_arms: int | None, seed: int) -> List[Tuple[Arm, Tuple[int, ...]]]:
        all_arms = list(itertools.product(*self._value_sets))
        if max_arms is None or max_arms >= len(all_arms):
            return [(self._encode_arm(arm), arm) for arm in all_arms]
        rng = random.Random(seed)
        sampled = rng.sample(all_arms, k=max_arms)
        return [(self._encode_arm(arm), arm) for arm in sampled]

    def _encode_arm(self, arm: Tuple[int, ...]) -> Arm:
        parts = []
        for key, value in zip(self._arm_keys, arm):
            parts.append(f"{key}={int(value)}")
        return ",".join(parts)

    def _decode_arm(self, arm_id: Arm) -> Tuple[int, ...]:
        values = {}
        for part in arm_id.split(","):
            key, value = part.split("=", 1)
            values[key] = int(value)
        return tuple(values[key] for key in self._arm_keys)

    def list_arms(self) -> List[Arm]:
        return list(self._arm_lookup.keys())

    def decode_arm(self, arm: Arm) -> dict[str, Any]:
        # Decoding an externally supplied configuration must not expand the
        # legal catalog. In particular, the parameters file may contain a
        # timeout outside the discrete grid; accelerator filtering later calls
        # list_arms(), which must still return only the original legal arms.
        values = self._arm_lookup.get(arm)
        if values is None:
            values = self._decode_arm(arm)
        return dict(zip(self._arm_keys, values))

