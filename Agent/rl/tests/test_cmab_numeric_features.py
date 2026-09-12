from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from actions.action_encode import ActionCodec
from cmab.arm_catalog import ArmCatalog
from cmab.policy import CMABPolicy


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=4,"
    "fast_path_timeout=300,k=4",
)


class CMABNumericFeatureTests(unittest.TestCase):
    def test_policy_uses_raw_protocol_parameter_values(self) -> None:
        policy = CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            fit_every=100,
        )

        np.testing.assert_array_equal(
            policy._arm_to_vector(ARMS[1]),
            np.asarray([500000, 64, 4, 300, 4], dtype=np.float32),
        )

    def test_current_catalog_contains_96_actions_and_200ms_timeout(self) -> None:
        catalog = ArmCatalog(codec=ActionCodec(policy="rf_ts"))

        self.assertEqual(len(catalog.list_arms()), 96)
        self.assertIn(200, catalog.codec.fast_path_timeout_ms_values)


if __name__ == "__main__":
    unittest.main()
