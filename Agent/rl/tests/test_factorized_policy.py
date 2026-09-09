from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from cmab.factorized_policy import FactorizedCMABPolicy, parse_arm_factors
from cmab.policy import CMABPolicy


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=3,"
    "fast_path_timeout=200,k=1",
    "batch_size=100000,header_size=32,cut_condition_type=3,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=3,"
    "fast_path_timeout=200,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=3,"
    "fast_path_timeout=0,k=1",
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=200,k=4",
)

CTX = [0.10, 0.20, 0.30, 0.40, 0.50]


class FactorizedCMABPolicyTests(unittest.TestCase):
    def _policy(self, **kwargs) -> FactorizedCMABPolicy:
        defaults = dict(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            random_state=0,
            min_samples_to_fit=4,
            n_estimators=12,
            max_depth=4,
            min_samples_leaf=2,
        )
        defaults.update(kwargs)
        return FactorizedCMABPolicy(**defaults)

    def test_parse_arm_factors(self) -> None:
        self.assertEqual(
            parse_arm_factors(ARMS[0]),
            {
                "batch_size": 100000,
                "header_size": 32,
                "cut_condition_type": 3,
                "fast_path_timeout": 200,
                "k": 1,
            },
        )

    def test_cut_is_one_hot_not_scalar(self) -> None:
        policy = self._policy()
        factors = parse_arm_factors(ARMS[0])
        cut_vec = policy._factor_vec("cut_condition_type", factors["cut_condition_type"])
        timeout_vec = policy._factor_vec("fast_path_timeout", factors["fast_path_timeout"])

        self.assertEqual(cut_vec.shape, (2,))
        self.assertEqual(float(cut_vec.sum()), 1.0)
        self.assertEqual(timeout_vec.shape, (1,))
        self.assertEqual(float(timeout_vec[0]), 200.0)

    def test_timeout_features_are_shared_across_arms(self) -> None:
        policy = self._policy()
        high_a = parse_arm_factors(ARMS[0])
        high_b = parse_arm_factors(ARMS[2])
        low = parse_arm_factors(ARMS[1])

        timeout_a = policy._main_row(CTX, high_a, "fast_path_timeout")
        timeout_b = policy._main_row(CTX, high_b, "fast_path_timeout")
        timeout_low = policy._main_row(CTX, low, "fast_path_timeout")

        self.assertTrue((timeout_a == timeout_b).all())
        self.assertFalse((timeout_a == timeout_low).all())

    def test_timeout_effect_transfers_to_unseen_arm(self) -> None:
        policy = self._policy()
        high = ARMS[0]
        low = ARMS[1]
        unseen_high = ARMS[2]
        unseen_low = ARMS[3]
        decisions = [high] * 10 + [low] * 10
        rewards = [8.0] * 10 + [1.0] * 10
        contexts = [CTX] * 20

        policy.update(decisions, rewards, contexts=contexts, shared_seed_hex="abc")

        self.assertTrue(policy._is_fitted)
        high_pred = policy._predict_reward(CTX, unseen_high)
        low_pred = policy._predict_reward(CTX, unseen_low)
        self.assertGreater(high_pred, low_pred)

    def test_select_arm_prefers_shared_high_timeout(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 10 + [ARMS[1]] * 10,
            [8.0] * 10 + [1.0] * 10,
            contexts=[CTX] * 20,
            shared_seed_hex="sel",
        )
        chosen = policy.select_arm(CTX, shared_seed_hex="sel2")
        self.assertIn(parse_arm_factors(chosen)["fast_path_timeout"], (200,))

    def test_select_arm_does_not_bonus_underexplored_factors(self) -> None:
        policy = self._policy()
        high = ARMS[0]
        rare_low = ARMS[1]
        policy.update(
            [high] * 16 + [rare_low],
            [8.0] * 16 + [1.0],
            contexts=[CTX] * 17,
            shared_seed_hex="counts",
        )
        chosen = policy.select_arm(CTX, shared_seed_hex="counts2")
        self.assertEqual(parse_arm_factors(chosen)["fast_path_timeout"], 200)

    def test_batched_scores_match_per_arm_predict(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 8 + [ARMS[1]] * 8,
            [8.0] * 8 + [1.0] * 8,
            contexts=[CTX] * 16,
            shared_seed_hex="batch",
        )
        batched = policy._predict_rewards_all_arms(CTX)
        per_arm = np.asarray(
            [policy._predict_reward(CTX, arm) for arm in ARMS],
            dtype=np.float64,
        )
        np.testing.assert_allclose(batched, per_arm, rtol=1e-10, atol=1e-10)

    def test_checkpoint_roundtrip(self) -> None:
        policy = self._policy()
        policy.update(
            [ARMS[0]] * 8,
            [3.0] * 8,
            contexts=[CTX] * 8,
            shared_seed_hex="save",
        )
        before = policy._predict_reward(CTX, ARMS[2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factorized.pkl"
            policy.save(path)
            restored = self._policy()
            restored.load(path)

        self.assertTrue(restored._is_fitted)
        self.assertEqual(len(restored._records), 8)
        self.assertAlmostEqual(restored._predict_reward(CTX, ARMS[2]), before)

    def test_cannot_load_global_checkpoint(self) -> None:
        global_policy = CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="rf_ts",
            action_encoding="numeric",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global.pkl"
            global_policy.save(path)
            with self.assertRaisesRegex(ValueError, "reward model mismatch"):
                self._policy().load(path)

    def test_global_policy_cannot_load_factorized_checkpoint(self) -> None:
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
            path = Path(directory) / "factorized.pkl"
            policy.save(path)
            with self.assertRaisesRegex(ValueError, "reward model mismatch"):
                global_policy.load(path)

    def test_round_robin_is_used_before_factorized_inference(self) -> None:
        policy = self._policy(policy_name="round_robin")
        first = [policy.select_arm(CTX) for _ in range(len(ARMS))]
        self.assertEqual(sorted(first), sorted(ARMS))


if __name__ == "__main__":
    unittest.main()
