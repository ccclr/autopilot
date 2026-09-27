from __future__ import annotations
import argparse
import runpy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

RL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_ROOT))
from cmab import ArmCatalog
from actions.action_encode import ActionCodec
from dqn.trainer import DQNTrainer
from controllers.action_receiver import ActionReceiver
from controllers.action_transport import PROTOCOL_VERSION

# Other pre-existing tests stub controllers.controller in sys.modules.
spec = importlib.util.spec_from_file_location('gcp_controller_test_target', RL_ROOT / 'controllers/controller.py')
controller_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller_module)


class DQNGCPTests(unittest.TestCase):
    def test_only_successfully_applied_contiguous_transitions_are_learned(self):
        for apply_ok, broadcast_ok, next_epoch, expected in (
            (True, True, 2, 1), (False, True, 2, 0),
            (None, True, 2, 0), (True, False, 2, 0), (True, True, 3, 0),
        ):
            with self.subTest(apply_ok=apply_ok, broadcast_ok=broadcast_ok, epoch=next_epoch):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    before = root / 'global_state_epoch_1.json'
                    after = root / f'global_state_epoch_{next_epoch}.json'
                    payload = {'global_reward': 1.0, 'global_fast_path_ratio': .5,
                               'state_4_lane_vector': {'growth_rates': {'0': 10.0}},
                               'param_apply_ok': apply_ok}
                    before.write_text(json.dumps(payload))
                    after.write_text(json.dumps(payload))
                    broadcaster = Mock(endpoints=[])
                    broadcaster.broadcast.return_value = SimpleNamespace(success=broadcast_ok, failed_nodes=[])
                    trainer = DQNTrainer(metrics_dir=directory, checkpoint_dir=directory,
                        arm_catalog=ArmCatalog(ActionCodec()), broadcaster=broadcaster,
                        policy_kwargs={'batch_size': 1, 'learning_starts': 1, 'seed': 1})
                    trainer._get_latest_metrics_file = lambda: before
                    trainer._wait_for_new_metrics_file = lambda *_args: after
                    with patch.object(Path, 'exists', return_value=False):
                        trainer.run(1, checkpoint_freq=10)
                    self.assertEqual(len(trainer.policy.replay_buffer), expected)
                    self.assertEqual(trainer.policy.gradient_steps, expected)

    def test_controller_builds_executable_dqn_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(controller_module.subprocess, 'Popen') as popen, \
                 patch.object(controller_module.threading, 'Thread'):
                controller = controller_module.AutopilotController(
                    metrics_dir=directory, parameters_file=directory+'/params.json',
                    log_dir=directory, node_index=0, rl_algo='dqn',
                    dqn_action_endpoints='0@127.0.0.1:19100',
                    dqn_checkpoint_load_mode='finetune', max_training_iterations=3,
                )
                command = popen.call_args.args[0]
                self.assertNotIn('--accelerator-period', command)
                self.assertEqual(command[command.index('--checkpoint-load-mode')+1], 'finetune')
                controller.cleanup_training()
            parse_args = argparse.ArgumentParser.parse_args
            class Parsed(Exception):
                pass
            def validate_args(parser, *args, **kwargs):
                namespace = parse_args(parser, command[2:])
                self.assertEqual(namespace.checkpoint_load_mode, 'finetune')
                raise Parsed()
            with patch.object(argparse.ArgumentParser, 'parse_args', validate_args):
                with self.assertRaises(Parsed):
                    runpy.run_path(str(RL_ROOT / 'train_dqn.py'), run_name='__main__')

    def test_offline_cli_checkpoint_can_be_finetuned(self):
        from offline_dataset import TransitionDatasetWriter
        from dqn.policy import DQNPolicy
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = ArmCatalog(ActionCodec()).list_arms()
            writer = TransitionDatasetWriter(root_dir=root / 'data',
                environment='gcp', run_id='smoke', arms=arms, node_index=0)
            writer.write(source_epoch=1, reward_epoch=2,
                state=np.asarray([10., .5]), arm=arms[0], reward=1.,
                next_state=np.asarray([11., .6]))
            writer.close()
            checkpoint = root / 'offline.pt'
            result = subprocess.run([sys.executable, str(RL_ROOT / 'train_dqn_offline.py'),
                '--dataset-root', str(root / 'data'), '--gradient-steps', '2',
                '--batch-size', '1', '--checkpoint-output', str(checkpoint)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            policy = DQNPolicy(state_dim=2, arms=arms)
            policy.load(checkpoint, mode='finetune')
            self.assertEqual(len(policy.replay_buffer), 0)
            self.assertEqual(policy.decision_steps, 0)

    def test_receiver_deduplicates_decisions(self):
        applier = Mock()
        receiver = ActionReceiver(0, applier)
        catalog = ArmCatalog(ActionCodec())
        arm = catalog.list_arms()[0]
        message = {'protocol_version': PROTOCOL_VERSION, 'type': 'apply_action', 'decision_id': 'test-1',
                   'signal_epoch': 1, 'action_id': 0, 'arm': arm,
                   'params': catalog.decode_arm(arm)}
        first = receiver.handle(message)
        second = receiver.handle(message)
        self.assertTrue(first['ok'])
        self.assertTrue(second['ok'])
        applier.apply.assert_called_once()


if __name__ == '__main__':
    unittest.main()
