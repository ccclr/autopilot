from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from cmab.combined_policy import CombinedCMABPolicy
from cmab.factorized_policy import FactorizedCMABPolicy
from cmab.policy import CMABPolicy


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=3,"
    "fast_path_timeout=200,k=1",
    "batch_size=100000,header_size=32,cut_condition_type=3,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=3,"
    "fast_path_timeout=200,k=4",
    "batch_size=500000,header_size=64,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
)

CTX = [0.10, 0.20, 0.30, 0.40, 0.50]


class CombinedCMABPolicyTests(unittest.TestCase):
    def _policy(self, **kwargs) -> CombinedCMABPolicy:
        defaults = dict(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            random_state=0,
            min_samples_to_fit=4,
            n_estimators=12,
            pair_n_estimators=8,
            pair_max_depth=3,
            pair_min_samples_leaf=2,
        )
        defaults.update(kwargs)
        return CombinedCMABPolicy(**defaults)

    def test_pair_features_are_state_plus_cut_and_timeout(self) -> None:
        policy = self._policy()
        row = policy._pair_feature_row(CTX, ARMS[0])
        np.testing.assert_array_equal(
            row,
            np.asarray([0.10, 0.20, 0.30, 0.40, 0.50, 3.0, 200.0], dtype=np.float32),
        )

    def test_pair_features_ignore_batch_header_and_k(self) -> None:
        policy = self._policy()
        a = policy._pair_feature_row(CTX, ARMS[0])
        b = policy._pair_feature_row(CTX, ARMS[2])
        np.testing.assert_array_equal(a, b)

    def test_pair_features_require_named_arms_and_context(self) -> None:
        policy = self._policy()
        with self.assertRaisesRegex(ValueError, "requires a state context"):
            policy._pair_feature_row(None, ARMS[0])
        with self.assertRaisesRegex(ValueError, "named CMAB arms"):
            policy._pair_feature_row(CTX, (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "missing pairwise parameter"):
            policy._pair_feature_row(CTX, "batch_size=1,header_size=2,k=3")

    def test_high_reward_samples_are_dropped_from_both_histories(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0], ARMS[1]],
            [3.0, 16.0],
            contexts=[CTX, CTX],
            shared_seed_hex="drop",
        )
        self.assertEqual(len(policy._y), 1)
        self.assertEqual(len(policy._pair_X), 1)
        self.assertEqual(len(policy._X), 1)

    def test_residual_targets_match_current_global_rf(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 8 + [ARMS[1]] * 8,
            [8.0] * 8 + [1.0] * 8,
            contexts=[CTX] * 16,
            shared_seed_hex="fit1",
        )
        self.assertTrue(policy._pair_rf_is_fitted)
        X = np.array(policy._X)
        y = np.array(policy._y)
        expected = y - policy._rf.predict(X)
        np.testing.assert_allclose(np.array(policy._pair_y), expected)

        policy.update(
            [ARMS[2]] * 4,
            [6.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="fit2",
        )
        X = np.array(policy._X)
        y = np.array(policy._y)
        expected = y - policy._rf.predict(X)
        np.testing.assert_allclose(np.array(policy._pair_y), expected)
        self.assertEqual(len(policy._pair_y), len(policy._y))

    def test_select_arm_uses_global_plus_pair_mean(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 8 + [ARMS[1]] * 8,
            [8.0] * 8 + [1.0] * 8,
            contexts=[CTX] * 16,
            shared_seed_hex="score",
        )
        mean, _std, global_mean, pair_mean = policy._score_arms(CTX)
        self.assertIsNotNone(pair_mean)
        np.testing.assert_allclose(mean, global_mean + pair_mean)
        self.assertFalse(np.allclose(mean, global_mean))
        chosen = policy.select_arm(CTX, shared_seed_hex="score2")
        max_pred = float(mean.max())
        max_arms = {
            ARMS[i] for i in np.flatnonzero(np.isclose(mean, max_pred))
        }
        self.assertIn(chosen, max_arms)

    def test_select_arm_without_pair_fit_uses_global_only(self) -> None:
        policy = self._policy(min_samples_to_fit=100)
        policy._X = [policy._feature_row(CTX, ARMS[0])]
        policy._y = [1.0]
        policy._is_fitted = False
        chosen = policy.select_arm(CTX, shared_seed_hex="cold")
        self.assertIn(chosen, ARMS)

    def test_checkpoint_roundtrip(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 8 + [ARMS[1]] * 8,
            [5.0] * 8 + [1.0] * 8,
            contexts=[CTX] * 16,
            shared_seed_hex="save",
        )
        before = policy._score_arms(CTX)[0]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pkl"
            policy.save(path)
            restored = self._policy()
            restored.load(path)

        self.assertTrue(restored._is_fitted)
        self.assertTrue(restored._pair_rf_is_fitted)
        self.assertEqual(len(restored._pair_X), 16)
        np.testing.assert_allclose(restored._score_arms(CTX)[0], before)

    def test_can_load_global_checkpoint_and_start_pair_later(self) -> None:
        global_policy = CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            action_encoding="numeric",
            min_samples_to_fit=4,
            n_estimators=12,
        )
        global_policy.update(
            [ARMS[0]] * 8,
            [3.0] * 8,
            contexts=[CTX] * 8,
            shared_seed_hex="g",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global.pkl"
            global_policy.save(path)
            combined = self._policy()
            combined.load(path)

        self.assertTrue(combined._is_fitted)
        self.assertFalse(combined._pair_rf_is_fitted)
        self.assertEqual(combined._pair_start_index, 8)
        self.assertEqual(combined._pair_X, [])

        combined.update(
            [ARMS[1]] * 8,
            [1.0] * 8,
            contexts=[CTX] * 8,
            shared_seed_hex="after",
        )
        self.assertEqual(len(combined._pair_X), 8)
        self.assertEqual(len(combined._pair_y), 8)
        self.assertTrue(combined._pair_rf_is_fitted)

    def test_cannot_load_factorized_checkpoint(self) -> None:
        factorized = FactorizedCMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            min_samples_to_fit=4,
            n_estimators=8,
            max_depth=3,
            min_samples_leaf=2,
        )
        factorized.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="f",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factorized.pkl"
            factorized.save(path)
            with self.assertRaisesRegex(ValueError, "reward model mismatch"):
                self._policy().load(path)

    def test_global_policy_cannot_load_combined_checkpoint(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="x",
        )
        global_policy = CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            action_encoding="numeric",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pkl"
            policy.save(path)
            with self.assertRaisesRegex(ValueError, "reward model mismatch"):
                global_policy.load(path)

    def test_schema_mismatch_is_rejected(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="schema",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pkl"
            policy.save(path)
            import joblib

            data = joblib.load(path)
            data["pair_feature_schema"] = "old_cut_timeout_v1"
            joblib.dump(data, path)
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                self._policy().load(path)

    def test_round_robin_is_used_before_combined_inference(self) -> None:
        policy = self._policy(policy_name="round_robin")
        first = [policy.select_arm(CTX) for _ in range(len(ARMS))]
        self.assertEqual(sorted(first), sorted(ARMS))
        self.assertFalse(policy._pair_rf_is_fitted)

    def test_missing_arm_raises(self) -> None:
        policy = self._policy()
        with self.assertRaises(ValueError):
            policy._arm_values("not-an-arm")

    def test_target_schema_mismatch_is_rejected(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="target",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pkl"
            policy.save(path)
            import joblib

            data = joblib.load(path)
            data["pair_target_schema"] = "stale_residual_v0"
            joblib.dump(data, path)
            with self.assertRaisesRegex(ValueError, "target schema mismatch"):
                self._policy().load(path)

    def test_action_encoding_mismatch_is_rejected(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="enc",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.pkl"
            policy.save(path)
            other = self._policy(action_encoding="one_hot")
            with self.assertRaisesRegex(ValueError, "action encoding mismatch"):
                other.load(path)

    def test_pair_does_not_fit_when_global_skips_fit_every(self) -> None:
        policy = self._policy(fit_every=2, min_samples_to_fit=4)
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="skip1",
        )
        self.assertEqual(len(policy._pair_X), 4)
        self.assertFalse(policy._is_fitted)
        self.assertFalse(policy._pair_rf_is_fitted)

        policy.update(
            [ARMS[1]] * 4,
            [1.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="skip2",
        )
        self.assertTrue(policy._is_fitted)
        self.assertTrue(policy._pair_rf_is_fitted)

    def test_misaligned_pair_history_raises(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 4,
            [2.0] * 4,
            contexts=[CTX] * 4,
            shared_seed_hex="align",
        )
        policy._pair_X.pop()
        with self.assertRaisesRegex(ValueError, "not aligned"):
            policy._fit_pairwise_residual("align2")

    def test_pair_and_global_bootstrap_use_different_labels(self) -> None:
        policy = self._policy()
        policy._update_count = 3
        global_idx = policy._bootstrap_indices(10, "seed")
        pair_idx = policy._bootstrap_indices(10, "seed", label="pair_outer_bootstrap")
        self.assertFalse(np.array_equal(global_idx, pair_idx))
        default_idx = policy._bootstrap_indices(10, "seed", label="outer_bootstrap")
        np.testing.assert_array_equal(global_idx, default_idx)


if __name__ == "__main__":
    unittest.main()
