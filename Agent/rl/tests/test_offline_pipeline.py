from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from dqn.policy import DQNPolicy
from cmab.trainer import CMABTrainer
from offline_dataset import (
    AsyncTransitionDatasetWriter,
    BalancedTransitionSampler,
    TransitionDatasetWriter,
    discover_transition_files,
    load_transition_files,
)


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=3,"
    "fast_path_timeout=100,k=4",
)


class OfflineDatasetTests(unittest.TestCase):
    def _write_run(self, root: Path, environment: str, run_id: str) -> Path:
        writer = TransitionDatasetWriter(
            root_dir=root,
            environment=environment,
            run_id=run_id,
            arms=ARMS,
            node_index=0,
        )
        writer.write(
            source_epoch=1,
            reward_epoch=2,
            state=np.asarray([10.0, 0.5], dtype=np.float32),
            arm=ARMS[0],
            reward=1.25,
            next_state=np.asarray([12.0, 0.6], dtype=np.float32),
        )
        writer.close()
        return writer.transition_path

    def test_multiple_runs_are_loaded_without_cross_run_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "A", "run-1")
            self._write_run(root, "A", "run-2")
            self._write_run(root, "B", "run-1")

            paths = discover_transition_files(root)
            records = load_transition_files(paths)

            self.assertEqual(len(paths), 3)
            self.assertEqual(len(records), 3)
            self.assertEqual(
                {(record.environment, record.run_id) for record in records},
                {("A", "run-1"), ("A", "run-2"), ("B", "run-1")},
            )
            self.assertTrue(
                all(record.reward_epoch == record.source_epoch + 1 for record in records)
            )

            sampler = BalancedTransitionSampler(records, seed=7)
            sample = sampler.sample(50)
            self.assertEqual(len(sample), 50)
            self.assertEqual({record.environment for record in sample}, {"A", "B"})

    def test_async_writer_persists_records_before_close_returns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = AsyncTransitionDatasetWriter(
                root_dir=directory,
                environment="A",
                run_id="async-run",
                arms=ARMS,
                node_index=0,
            )
            path = writer.transition_path
            self.assertTrue(
                writer.write(
                    source_epoch=1,
                    reward_epoch=2,
                    state=np.asarray([10.0, 0.5], dtype=np.float32),
                    arm=ARMS[0],
                    reward=1.25,
                    next_state=np.asarray([12.0, 0.6], dtype=np.float32),
                )
            )
            writer.close()
            records = load_transition_files([path])
            self.assertEqual(len(records), 1)

    def test_epoch_actions_include_abandon_without_creating_a_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = TransitionDatasetWriter(
                root_dir=directory,
                environment="A",
                run_id="action-run",
                arms=ARMS,
                node_index=0,
                enable_epoch_actions=True,
            )
            self.assertTrue(writer.write_epoch_action(
                source_epoch=0,
                reward_epoch=1,
                selected_arm=ARMS[0],
                effective_arm=ARMS[0],
                abandoned=False,
            ))
            self.assertTrue(writer.write_epoch_action(
                source_epoch=1,
                reward_epoch=2,
                selected_arm=ARMS[1],
                effective_arm=ARMS[0],
                abandoned=True,
            ))
            self.assertFalse(writer.write_epoch_action(
                source_epoch=1,
                reward_epoch=2,
                selected_arm=ARMS[1],
                effective_arm=ARMS[0],
                abandoned=True,
            ))
            self.assertTrue(writer.write_epoch_action(
                source_epoch=2,
                reward_epoch=3,
                selected_arm=ARMS[1],
                effective_arm=None,
                abandoned=True,
            ))
            writer.close()

            actions = [
                json.loads(line)
                for line in writer.epoch_action_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(actions), 3)
            self.assertEqual(actions[0]["status"], "applied")
            self.assertEqual(actions[1]["epoch"], 2)
            self.assertEqual(actions[1]["selected_action_id"], 1)
            self.assertEqual(actions[1]["effective_action_id"], 0)
            self.assertEqual(actions[1]["status"], "abandoned_reused_previous")
            self.assertEqual(actions[2]["status"], "abandoned_effective_unknown")
            self.assertIsNone(actions[2]["effective_action_id"])
            self.assertEqual(writer.transition_path.read_text(encoding="utf-8"), "")

    def test_async_epoch_action_is_flushed_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = AsyncTransitionDatasetWriter(
                root_dir=directory,
                environment="A",
                run_id="async-action-run",
                arms=ARMS,
                node_index=0,
                enable_epoch_actions=True,
            )
            self.assertTrue(writer.write_epoch_action(
                source_epoch=4,
                reward_epoch=5,
                selected_arm=ARMS[1],
                effective_arm=ARMS[0],
                abandoned=True,
            ))
            writer.close()
            actions = [
                json.loads(line)
                for line in writer.epoch_action_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["effective_action_id"], 0)

    def test_existing_cmab_writer_keeps_cmab_behavior_policy_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_run(Path(directory), "A", "cmab-run")
            record = json.loads(path.read_text(encoding="utf-8").strip())
            manifest = json.loads(
                (path.parent / "meta.json").read_text(encoding="utf-8")
            )

            self.assertEqual(record["behavior_policy"], "cmab")
            self.assertEqual(manifest["behavior_policy"], "cmab")
            self.assertNotIn("epoch_action_file", manifest)
            self.assertFalse((path.parent / "epoch_actions.jsonl").exists())

    def test_async_writer_failure_does_not_propagate_to_cmab_caller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = AsyncTransitionDatasetWriter(
                root_dir=directory,
                environment="A",
                run_id="failing-run",
                arms=ARMS,
                node_index=0,
            )

            def fail_write(**_kwargs):
                raise OSError("simulated disk failure")

            writer._writer.write = fail_write
            self.assertTrue(
                writer.write(
                    source_epoch=1,
                    reward_epoch=2,
                    state=np.asarray([10.0, 0.5], dtype=np.float32),
                    arm=ARMS[0],
                    reward=1.25,
                    next_state=np.asarray([12.0, 0.6], dtype=np.float32),
                )
            )
            deadline = time.monotonic() + 1.0
            while not writer.failed and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(writer.failed)
            self.assertFalse(
                writer.write(
                    source_epoch=2,
                    reward_epoch=3,
                    state=np.asarray([12.0, 0.6], dtype=np.float32),
                    arm=ARMS[0],
                    reward=1.30,
                    next_state=np.asarray([13.0, 0.7], dtype=np.float32),
                )
            )
            writer.close()


class DQNCheckpointTests(unittest.TestCase):
    def _policy(self, learning_rate: float) -> DQNPolicy:
        return DQNPolicy(
            state_dim=2,
            arms=ARMS,
            learning_rate=learning_rate,
            replay_capacity=8,
            batch_size=1,
            learning_starts=1,
            target_update_interval=2,
            hidden_dim=8,
            seed=3,
        )

    def test_finetune_load_keeps_new_optimizer_and_resets_online_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pt"
            source = self._policy(learning_rate=1e-3)
            source.observe(
                np.asarray([10.0, 0.5]),
                0,
                1.0,
                np.asarray([11.0, 0.6]),
            )
            source.select_action(np.asarray([10.0, 0.5]))
            source.train(updates=1)
            source.save(checkpoint)

            target = self._policy(learning_rate=1e-4)
            target.load(checkpoint, mode="finetune")

            self.assertEqual(len(target.replay_buffer), 0)
            self.assertEqual(target.decision_steps, 0)
            self.assertEqual(target.gradient_steps, 0)
            self.assertEqual(target.transitions_seen, 0)
            self.assertAlmostEqual(
                target.optimizer.param_groups[0]["lr"], 1e-4
            )
            for online, target_value in zip(
                target.online_network.parameters(),
                target.target_network.parameters(),
            ):
                self.assertTrue(torch.equal(online, target_value))


class CMABExportIsolationTests(unittest.TestCase):
    def test_abandoned_epoch_records_reused_action_without_transition(self) -> None:
        class FakePolicy:
            policy_name = "test"
            uses_context = True

            def __init__(self) -> None:
                self.selected = iter(ARMS)
                self.updated = []

            def select_arm(self, _context, shared_seed_hex=None):
                return next(self.selected)

            def update(self, arms, rewards, contexts, shared_seed_hex=None):
                self.updated.append(arms[0])

        class FakeCatalog:
            @staticmethod
            def decode_arm(arm):
                return {"arm": arm}

        class RecordingWriter:
            def __init__(self) -> None:
                self.actions = []
                self.transitions = []

            def write_epoch_action(self, **record):
                self.actions.append(record)
                return True

            def write(self, **record):
                self.transitions.append(record)
                return True

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"global_state_epoch_{epoch}.json" for epoch in range(3)]
            for epoch, path in enumerate(files):
                path.write_text(json.dumps({"global_reward": 1.0 + epoch}), encoding="utf-8")
            policy = FakePolicy()
            writer = RecordingWriter()
            trainer = CMABTrainer(
                metrics_dir=str(root),
                parameters_file=str(root / "parameters.json"),
                checkpoint_dir=str(root / "checkpoints"),
                policy=policy,
                context_builder=object(),
                arm_catalog=FakeCatalog(),
                warmup_iterations=0,
                transition_writer=writer,
            )
            trainer._connect_param_socket = lambda: None
            trainer._get_latest_metrics_file = lambda: files[0]
            trainer._load_initial_arm_from_parameters_file = lambda: ARMS[0]
            trainer._write_parameters_to_file = lambda _params, _epoch: None
            next_files = iter(files[1:])
            trainer._wait_for_new_metrics_file = lambda _last, timeout: next(next_files)
            trainer._build_context_from_global_state = lambda _path: np.asarray([1.0])
            trainer._build_context_from_data = lambda _data: np.asarray([1.0])
            trainer._compute_shared_seed_hex = lambda _path: "00"
            original_exists = Path.exists

            def signal_exists(path):
                return (
                    path.name == "autopilot_rl_param_abandon_1.signal"
                    or original_exists(path)
                )

            with mock.patch.object(Path, "exists", signal_exists):
                with mock.patch.object(trainer, "_save_latest_checkpoint") as save_latest:
                    trainer.run(num_iterations=2, checkpoint_freq=10)
                self.assertEqual(save_latest.call_count, 0)

            self.assertEqual(policy.updated, [ARMS[0], ARMS[0]])
            self.assertEqual(len(writer.transitions), 1)
            self.assertEqual(writer.actions[0]["selected_arm"], ARMS[0])
            self.assertEqual(writer.actions[1]["selected_arm"], ARMS[1])
            self.assertEqual(writer.actions[1]["effective_arm"], ARMS[0])
            self.assertTrue(writer.actions[1]["abandoned"])

    def test_export_failure_cannot_undo_or_stop_cmab_iteration(self) -> None:
        arm = ARMS[0]

        class FakePolicy:
            policy_name = "test"
            uses_context = True

            def __init__(self) -> None:
                self.updates = []

            def select_arm(self, _context, shared_seed_hex=None):
                return arm

            def update(self, arms, rewards, contexts, shared_seed_hex=None):
                self.updates.append((arms, rewards, contexts))

            def save(self, _path):
                raise AssertionError("checkpoint should not run in this test")

        class FakeCatalog:
            @staticmethod
            def decode_arm(selected_arm):
                if selected_arm != arm:
                    raise AssertionError(f"unexpected arm: {selected_arm}")
                return {"batch_size": 100000}

        class FailingWriter:
            run_id = "failing-writer"

            def __init__(self) -> None:
                self.write_calls = 0
                self.close_calls = 0

            def write(self, **_kwargs):
                self.write_calls += 1
                raise OSError("simulated exporter failure")

            def write_epoch_action(self, **_kwargs):
                return True

            def close(self):
                self.close_calls += 1
                raise OSError("simulated close failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "global_state_epoch_987654.json"
            following = root / "global_state_epoch_987655.json"
            current.write_text('{"global_reward": 1.0}\n', encoding="utf-8")
            following.write_text('{"global_reward": 1.25}\n', encoding="utf-8")
            policy = FakePolicy()
            writer = FailingWriter()
            trainer = CMABTrainer(
                metrics_dir=str(root),
                parameters_file=str(root / "parameters.json"),
                checkpoint_dir=str(root / "checkpoints"),
                policy=policy,
                context_builder=object(),
                arm_catalog=FakeCatalog(),
                warmup_iterations=0,
                transition_writer=writer,
            )
            trainer._connect_param_socket = lambda: None
            trainer._get_latest_metrics_file = lambda: current
            trainer._load_initial_arm_from_parameters_file = lambda: None
            trainer._write_parameters_to_file = lambda _params, _epoch: None
            trainer._wait_for_new_metrics_file = (
                lambda _last, timeout: following
            )
            trainer._build_context_from_global_state = (
                lambda path: np.asarray(
                    [10.0 if path == current else 11.0, 0.5],
                    dtype=np.float32,
                )
            )
            trainer._build_context_from_data = lambda _data: np.asarray(
                [11.0, 0.5], dtype=np.float32
            )
            trainer._compute_shared_seed_hex = lambda _path: "00"

            trainer.run(num_iterations=1, checkpoint_freq=10)

            self.assertEqual(len(policy.updates), 1)
            self.assertEqual(trainer.last_metrics_file, following)
            self.assertEqual(list(trainer.reward_history), [1.25])
            self.assertEqual(writer.write_calls, 1)
            self.assertEqual(writer.close_calls, 1)
            self.assertIsNone(trainer.transition_writer)


if __name__ == "__main__":
    unittest.main()
