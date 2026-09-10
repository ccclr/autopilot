from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from cmab.policy import CMABPolicy


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=4,"
    "fast_path_timeout=200,k=4",
)


class _FixedTree:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.float64)

    def predict(self, features):
        return self.predictions[: len(features)]


class CMABPairwiseResidualTests(unittest.TestCase):
    @staticmethod
    def _policy(enabled: bool) -> CMABPolicy:
        return CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            n_estimators=5,
            random_state=7,
            enable_pairwise_residual_rf=enabled,
        )

    def test_disabled_mode_keeps_pairwise_model_unused(self) -> None:
        policy = self._policy(False)

        policy.update([ARMS[0]], [1.25], contexts=[[0.2, 0.4]])

        self.assertEqual(policy._pair_X, [])
        self.assertEqual(policy._pair_y, [])
        self.assertFalse(policy._pair_rf_is_fitted)

    def test_pair_features_use_only_cut_and_timeout(self) -> None:
        policy = self._policy(True)

        np.testing.assert_array_equal(
            policy._pair_feature_row(ARMS[1]),
            np.asarray([4, 200], dtype=np.float32),
        )

    def test_pairwise_rf_starts_with_zero_cold_start_residual(self) -> None:
        policy = self._policy(True)

        policy.update([ARMS[0]], [1.25], contexts=[[0.2, 0.4]])

        self.assertEqual(policy._pair_y, [0.0])
        self.assertTrue(policy._pair_rf_is_fitted)

    def test_new_residual_uses_global_model_before_update(self) -> None:
        policy = self._policy(True)
        context = [0.2, 0.4]
        policy.update([ARMS[0]], [1.25], contexts=[context])
        feature = policy._feature_row(context, ARMS[1])
        expected = 2.0 - float(policy._rf.predict([feature])[0])

        policy.update([ARMS[1]], [2.0], contexts=[context])

        self.assertAlmostEqual(policy._pair_y[-1], expected)

    def test_optional_pair_model_does_not_change_global_rf(self) -> None:
        global_only = self._policy(False)
        factorized = self._policy(True)
        contexts = [[0.2, 0.4], [0.3, 0.5]]
        for arm, reward, context in zip(ARMS, [1.25, 2.0], contexts):
            global_only.update([arm], [reward], contexts=[context])
            factorized.update([arm], [reward], contexts=[context])

        probe = global_only._feature_matrix([0.25, 0.45])
        np.testing.assert_allclose(
            global_only._rf.predict(probe),
            factorized._rf.predict(probe),
        )

    def test_selection_adds_pair_residual_without_weight(self) -> None:
        policy = self._policy(True)
        policy._is_fitted = True
        policy._y = [1.0]
        policy._rf = SimpleNamespace(estimators_=[_FixedTree([2.0, 1.0])])
        policy._pair_rf = SimpleNamespace(
            estimators_=[_FixedTree([-2.0, 2.0])]
        )
        policy._pair_rf_is_fitted = True

        selected = policy.select_arm([0.2, 0.4], shared_seed_hex="seed")

        # Global scores [2, 1] plus pair residuals [-2, 2] give [0, 3].
        self.assertEqual(selected, ARMS[1])

    def test_pairwise_checkpoint_round_trip(self) -> None:
        source = self._policy(True)
        source.update([ARMS[0]], [1.25], contexts=[[0.2, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cmab_pair.pkl"
            source.save(checkpoint)
            target = self._policy(True)

            target.load(checkpoint)

            self.assertTrue(target._pair_rf_is_fitted)
            self.assertEqual(target._pair_y, [0.0])


if __name__ == "__main__":
    unittest.main()
