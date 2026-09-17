from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmab.policy import CMABPolicy
from cmab.trainer import CMABTrainer
from offline_dataset import AsyncTransitionDatasetWriter


ARM = "batch_size=100000,header_size=32,cut_condition_type=2,fast_path_timeout=200,k=1"


class RunCheckpointTests(unittest.TestCase):
    def test_latest_is_saved_at_epoch_10_and_20_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"global_state_epoch_{epoch}.json" for epoch in (8, 9, 10, 11, 20)]
            for path in files:
                path.write_text(json.dumps({"global_reward": 2.0}))
            policy = mock.Mock(policy_name="rf_ts", uses_context=True)
            policy.select_arm.return_value = ARM
            catalog = mock.Mock()
            catalog.decode_arm.return_value = {}
            trainer = CMABTrainer(
                metrics_dir=directory, parameters_file=str(root / "params"),
                checkpoint_dir=str(root / "legacy"), policy=policy,
                context_builder=None, arm_catalog=catalog, warmup_iterations=0,
            )
            trainer._connect_param_socket = lambda: None
            trainer._get_latest_metrics_file = lambda: files[0]
            trainer._load_initial_arm_from_parameters_file = lambda: ARM
            trainer._write_parameters_to_file = lambda *_: None
            following = iter(files[1:])
            trainer._wait_for_new_metrics_file = lambda *a, **kw: next(following)
            trainer._build_context_from_global_state = lambda _: np.zeros(5)
            trainer._compute_shared_seed_hex = lambda _: "00"
            saved_after_updates = []
            trainer._save_latest_checkpoint = lambda: saved_after_updates.append(policy.update.call_count)
            trainer.run(num_iterations=4, checkpoint_freq=10)
            self.assertEqual(saved_after_updates, [2, 4])

    def test_runs_are_isolated_and_latest_can_be_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for run in ("first", "second"):
                writer = AsyncTransitionDatasetWriter(
                    root_dir=directory, environment="A", run_id=run,
                    arms=[ARM], node_index=0,
                )
                try:
                    path = writer.run_dir / "cmab_checkpoint_latest.pkl"
                    policy = CMABPolicy([ARM], 5, enable_cut_fpt_cross_feature=True)
                    policy._rf.verbose = 0
                    trainer = CMABTrainer(
                        metrics_dir=directory, parameters_file=str(Path(directory)/"params"),
                        checkpoint_dir=str(Path(directory)/"legacy"), policy=policy,
                        context_builder=None, arm_catalog=None,
                        latest_checkpoint_path=str(path),
                    )
                    for reward in (1.0, 2.0):
                        policy.update([ARM], [reward], [np.zeros(5)])
                        trainer._save_latest_checkpoint()
                    restored = CMABPolicy([ARM], 5, enable_cut_fpt_cross_feature=True)
                    restored.load(str(path))
                    self.assertEqual(restored._y, [1.0, 2.0])
                    np.testing.assert_array_equal(
                        restored._rf.predict(policy._X), policy._rf.predict(policy._X)
                    )
                    before = path.read_bytes()
                    def broken_save(temporary):
                        Path(temporary).write_bytes(b"partial")
                        raise OSError("simulated full disk")
                    with mock.patch.object(policy, "save", side_effect=broken_save):
                        with self.assertLogs("cmab.trainer", level="ERROR"):
                            trainer._save_latest_checkpoint()
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(list(path.parent.glob("*.tmp")), [])
                    paths.append(path)
                finally:
                    writer.close()
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(joblib.load(paths[0])["update_count"], 2)


if __name__ == "__main__":
    unittest.main()
