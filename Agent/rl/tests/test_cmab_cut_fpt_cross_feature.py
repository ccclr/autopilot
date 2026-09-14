from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from cmab.policy import CMABPolicy


def _arm(cut: int, timeout: int) -> str:
    return (
        "batch_size=100000,header_size=32,"
        f"cut_condition_type={cut},fast_path_timeout={timeout},k=1"
    )


ARMS = tuple(
    _arm(cut, timeout)
    for cut in (2, 3, 4)
    for timeout in (0, 100, 200, 300)
)
CONTEXT = [0.2, 0.4, 0.6, 0.8, 1.0]


class CMABCutFptCrossFeatureTests(unittest.TestCase):
    @staticmethod
    def _policy(enabled: bool) -> CMABPolicy:
        return CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            n_estimators=5,
            random_state=7,
            enable_cut_fpt_cross_feature=enabled,
        )

    def test_disabled_mode_preserves_original_numeric_features(self) -> None:
        policy = self._policy(False)

        row = policy._feature_row(CONTEXT, _arm(4, 200))

        self.assertEqual(row.shape, (10,))
        np.testing.assert_allclose(
            row,
            CONTEXT + [100000, 32, 4, 200, 1],
        )

    def test_enabled_mode_appends_one_hot_without_replacing_numeric_values(self) -> None:
        policy = self._policy(True)

        row = policy._feature_row(CONTEXT, _arm(4, 200))

        self.assertEqual(row.shape, (22,))
        np.testing.assert_allclose(
            row[:10],
            CONTEXT + [100000, 32, 4, 200, 1],
        )
        self.assertEqual(float(row[10:].sum()), 1.0)
        self.assertEqual(int(np.argmax(row[10:])), 10)

    def test_all_twelve_pairs_have_distinct_deterministic_columns(self) -> None:
        policy = self._policy(True)

        columns = [
            int(np.argmax(policy._cut_fpt_cross_vector(arm))) for arm in ARMS
        ]

        self.assertEqual(columns, list(range(12)))

    def test_training_and_inference_use_same_augmented_dimension(self) -> None:
        policy = self._policy(True)

        policy.update([ARMS[0]], [1.25], contexts=[CONTEXT])

        self.assertEqual(policy._rf.n_features_in_, 22)
        self.assertEqual(policy._feature_matrix(CONTEXT).shape, (12, 22))
        self.assertIn(policy.select_arm(CONTEXT, shared_seed_hex="seed"), ARMS)

    def test_replay_arm_matching_includes_cross_suffix(self) -> None:
        policy = self._policy(True)
        policy.update(
            [ARMS[0], ARMS[-1]],
            [1.0, 2.0],
            contexts=[CONTEXT, CONTEXT],
        )

        counts, matched = policy._window_arm_counts_from_replay()

        self.assertEqual(matched, 2)
        self.assertEqual(counts[ARMS[0]], 1)
        self.assertEqual(counts[ARMS[-1]], 1)

    def test_cross_checkpoint_round_trip(self) -> None:
        source = self._policy(True)
        source.update([ARMS[0]], [1.25], contexts=[CONTEXT])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cmab_cross.pkl"
            source.save(checkpoint)
            target = self._policy(True)

            target.load(checkpoint)

            self.assertTrue(target._is_fitted)
            np.testing.assert_allclose(target._X, source._X)
            self.assertEqual(target._rf.n_features_in_, 22)

    def test_checkpoint_rejects_cross_setting_mismatch(self) -> None:
        source = self._policy(False)
        source.update([ARMS[0]], [1.25], contexts=[CONTEXT])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cmab_numeric.pkl"
            source.save(checkpoint)

            with self.assertRaisesRegex(ValueError, "setting does not match"):
                self._policy(True).load(checkpoint)

    def test_old_pairwise_checkpoint_remains_usable_as_global_only(self) -> None:
        source = self._policy(False)
        source.update([ARMS[0]], [1.25], contexts=[CONTEXT])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cmab_old_pairwise.pkl"
            source.save(checkpoint)

            import joblib

            data = joblib.load(checkpoint)
            data.pop("enable_cut_fpt_cross_feature")
            data["enable_pairwise_residual_rf"] = True
            data["pair_rf"] = object()
            joblib.dump(data, checkpoint)
            target = self._policy(False)

            target.load(checkpoint)

            self.assertTrue(target._is_fitted)
            self.assertEqual(target._rf.n_features_in_, 10)


if __name__ == "__main__":
    unittest.main()
