from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
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


class CMABActionEncodingTests(unittest.TestCase):
    def _policy(self, action_encoding: str = "numeric") -> CMABPolicy:
        return CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            fit_every=100,
            action_encoding=action_encoding,
        )

    def test_numeric_remains_the_default_and_preserves_raw_values(self) -> None:
        policy = self._policy()

        np.testing.assert_array_equal(
            policy._arm_to_vector(ARMS[1]),
            np.asarray([500000, 64, 4, 300, 4], dtype=np.float32),
        )

    def test_one_hot_uses_one_feature_per_complete_arm(self) -> None:
        policy = self._policy("one_hot")

        np.testing.assert_array_equal(
            policy._arm_to_vector(ARMS[0]),
            np.asarray([1, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            policy._arm_to_vector(ARMS[1]),
            np.asarray([0, 1], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            policy._feature_row([0.25, 0.75], ARMS[1]),
            np.asarray([0.25, 0.75, 0, 1], dtype=np.float64),
        )

    def test_current_catalog_produces_96_one_hot_features(self) -> None:
        catalog = ArmCatalog(codec=ActionCodec(policy="rf_ts"))
        arms = catalog.list_arms()
        policy = CMABPolicy(
            arms=arms,
            feature_dim=5,
            action_encoding="one_hot",
        )

        encoded = policy._arm_to_vector(arms[40])

        self.assertEqual(len(arms), 96)
        self.assertEqual(encoded.shape, (96,))
        self.assertEqual(float(encoded.sum()), 1.0)
        self.assertEqual(float(encoded[40]), 1.0)

    def test_current_catalog_includes_200ms_timeout(self) -> None:
        catalog = ArmCatalog(codec=ActionCodec(policy="rf_ts"))

        self.assertEqual(catalog.timeout_values, (0, 100, 200, 300))
        self.assertEqual(len(catalog.list_arms()), 96)

    def test_recent_arm_matching_supports_one_hot_features(self) -> None:
        policy = self._policy("one_hot")
        policy.update([ARMS[1]], [1.0], contexts=[[0.1, 0.2]])

        counts, matched = policy._window_arm_counts_from_replay()

        self.assertEqual(matched, 1)
        self.assertEqual(counts, {ARMS[0]: 0, ARMS[1]: 1})

    def test_checkpoint_cannot_cross_action_encodings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cmab.pkl"
            self._policy("numeric").save(checkpoint)

            with self.assertRaisesRegex(ValueError, "encoding mismatch"):
                self._policy("one_hot").load(checkpoint)

    def test_legacy_checkpoint_is_treated_as_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pkl"
            source = self._policy("numeric")
            joblib.dump(
                {
                    'rf': source._rf,
                    'X': source._X,
                    'y': source._y,
                    'is_fitted': source._is_fitted,
                    'update_count': source._update_count,
                },
                checkpoint,
            )

            target = self._policy("numeric")
            target.load(checkpoint)

            self.assertEqual(target.action_encoding, "numeric")

    def test_rejects_unknown_encoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "action encoding"):
            self._policy("ordinal")


if __name__ == "__main__":
    unittest.main()
