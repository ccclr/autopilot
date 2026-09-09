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
from cmab.trainer import CMABTrainer


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

    def test_current_catalog_produces_72_one_hot_features(self) -> None:
        catalog = ArmCatalog(codec=ActionCodec(policy="rf_ts"))
        arms = catalog.list_arms()
        policy = CMABPolicy(
            arms=arms,
            feature_dim=5,
            action_encoding="one_hot",
        )

        encoded = policy._arm_to_vector(arms[40])

        self.assertEqual(len(arms), 72)
        self.assertEqual(encoded.shape, (72,))
        self.assertEqual(float(encoded.sum()), 1.0)
        self.assertEqual(float(encoded[40]), 1.0)

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

    def test_param_apply_ok_requires_explicit_true(self) -> None:
        self.assertTrue(CMABTrainer._param_apply_ok_from_payload({"param_apply_ok": True}))
        self.assertFalse(CMABTrainer._param_apply_ok_from_payload({"param_apply_ok": False}))
        self.assertFalse(CMABTrainer._param_apply_ok_from_payload({}))
        self.assertFalse(CMABTrainer._param_apply_ok_from_payload({"param_apply_ok": 1}))

    def test_wait_returns_consecutive_epoch_not_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            (metrics / "global_state_epoch_3.json").write_text("{}")
            (metrics / "global_state_epoch_4.json").write_text("{}")
            (metrics / "global_state_epoch_9.json").write_text("{}")
            trainer = CMABTrainer(
                metrics_dir=str(metrics),
                parameters_file=str(metrics / "params.json"),
                checkpoint_dir=str(metrics / "ckpt"),
                policy=None,
                context_builder=None,
                arm_catalog=None,
                metrics_timeout=1,
            )
            # Target epoch 4 exists, but a later file also exists: jump to live.
            jumped = trainer._wait_for_new_metrics_file(
                metrics / "global_state_epoch_3.json",
                timeout=1,
            )
            self.assertEqual(jumped.name, "global_state_epoch_9.json")

            (metrics / "global_state_epoch_9.json").unlink()
            consecutive = trainer._wait_for_new_metrics_file(
                metrics / "global_state_epoch_3.json",
                timeout=1,
            )
            self.assertEqual(consecutive.name, "global_state_epoch_4.json")

    def test_first_metrics_file_is_earliest_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp)
            (metrics / "global_state_epoch_7.json").write_text("{}")
            (metrics / "global_state_epoch_2.json").write_text("{}")
            trainer = CMABTrainer(
                metrics_dir=str(metrics),
                parameters_file=str(metrics / "params.json"),
                checkpoint_dir=str(metrics / "ckpt"),
                policy=None,
                context_builder=None,
                arm_catalog=None,
            )
            earliest = trainer._get_earliest_metrics_file()
            self.assertEqual(earliest.name, "global_state_epoch_2.json")

    def test_round_robin_covers_catalog_then_repeats(self) -> None:
        policy = CMABPolicy(
            arms=ARMS,
            feature_dim=5,
            policy_name="round_robin",
            random_state=0,
        )
        first = [policy.select_arm(None) for _ in range(len(ARMS))]
        self.assertEqual(sorted(first), sorted(ARMS))
        second = [policy.select_arm(None) for _ in range(len(ARMS))]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
