from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Agent' / 'rl'))
sys.path.insert(0, str(ROOT / 'benchmark'))

from cmab.accelerator import TrainingAccelerator, max_fast_path_delta_ms
from cmab.arm_catalog import ArmCatalog
from cmab.policy import CMABPolicy
from cmab.trainer import CMABTrainer
from cmab.factorized_policy import FactorizedCMABPolicy
from gp_bo.mixed_space import MixedActionSpace, MixedArmCatalog, decode_arm_params
from benchmark.config import BenchParameters, ConfigError
from benchmark.commands import CommandMaker


def arms(timeouts):
    return [f'fast_path_timeout={t},k={k}' for t in timeouts for k in (1, 4)]


class RTTTimeoutFilterTests(unittest.TestCase):
    def accelerator(self, cap_enabled=False, cap=120.0):
        accelerator = TrainingAccelerator(
            enable_timeout_cap=cap_enabled, enable_rtt_timeout_filter=True,
        )
        accelerator._applied_cap = cap
        return accelerator

    def test_legacy_default_unchanged(self):
        accelerator = TrainingAccelerator()
        accelerator._applied_cap = 120.0
        self.assertEqual(accelerator.filter_arms(arms([0, 100, 200, 300])), arms([0, 100, 200]))

    def test_filter_alone_keeps_strictly_larger(self):
        accelerator = self.accelerator()
        self.assertEqual(accelerator.filter_arms(arms([0, 100, 200, 300])), arms([300]))
        self.assertEqual(accelerator.covering_timeout([0, 100, 200, 300]), 200)

    def test_two_switches_fall_back_to_zero_timeout(self):
        self.assertEqual(self.accelerator(True).filter_arms(arms([0, 100, 200, 300])), arms([0]))

    def test_grid_boundaries_and_above_grid(self):
        grid = arms([0, 100, 200, 300])
        for cap, expected in [(0, [100, 200, 300]), (62, [200, 300]),
                              (100, [200, 300]), (100.1, [300]),
                              (200, [300]), (250, [0]), (400, [0])]:
            with self.subTest(cap=cap):
                self.assertEqual(self.accelerator(cap=cap).filter_arms(grid), arms(expected))
                self.assertEqual(self.accelerator(True, cap=cap).filter_arms(grid), arms([0]))

    def test_no_measurement_keeps_all(self):
        self.assertEqual(self.accelerator(cap=None).filter_arms(arms([0, 100])), arms([0, 100]))

    def test_empty_and_single_timeout_catalog(self):
        self.assertEqual(self.accelerator().filter_arms([]), [])
        self.assertEqual(self.accelerator().filter_arms(arms([100])), arms([0]))

    def test_hint_schedule_and_old_hint_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.accelerator.json'
            accelerator = TrainingAccelerator(hint_path=path, enable_timeout_cap=False,
                                              enable_rtt_timeout_filter=True)
            path.write_text(json.dumps(dict(timeout_cap=120, apply_epoch=15)))
            accelerator.on_epoch(14)
            self.assertEqual(accelerator.filter_arms(arms([0, 100, 200, 300])), arms([0, 100, 200, 300]))
            accelerator.on_epoch(15)
            self.assertEqual(accelerator.filter_arms(arms([0, 100, 200, 300])), arms([300]))

    def test_probe_publishes_existing_cap(self):
        matrix = np.asarray([[0, 20, 40, 150]] * 4, dtype=float)
        np.fill_diagonal(matrix, 0)
        expected = max_fast_path_delta_ms(matrix)
        fab = SimpleNamespace(collect_latency_matrix=lambda **kw: matrix,
                              publish_accelerator_hint=unittest.mock.Mock())
        with patch('cmab.accelerator._fabfile', return_value=fab), \
             patch('cmab.accelerator._prepare_gcp_credentials'):
            self.accelerator()._probe(10)
        hint, _ = fab.publish_accelerator_hint.call_args.args
        self.assertEqual(hint['timeout_cap'], np.ceil(expected) + 10)
        self.assertEqual(hint['apply_epoch'], 15)

    def test_discrete_trainer_restores_original_catalog_on_network_change(self):
        trainer = CMABTrainer.__new__(CMABTrainer)
        trainer.arm_catalog = ArmCatalog()
        original = trainer.arm_catalog.list_arms()
        trainer.policy = CMABPolicy(original, feature_dim=5, action_encoding='one_hot')
        trainer.accelerator = self.accelerator()
        trainer._apply_accelerator()
        self.assertTrue(all(decode_arm_params(a)['fast_path_timeout'] != 100 for a in trainer.policy._arms))
        trainer.accelerator._applied_cap = 0
        trainer._apply_accelerator()
        self.assertTrue(any(decode_arm_params(a)['fast_path_timeout'] == 100 for a in trainer.policy._arms))
        self.assertTrue(all(decode_arm_params(a)['fast_path_timeout'] >= 100 for a in trainer.policy._arms))
        self.assertEqual(trainer.arm_catalog.list_arms(), original)

    def test_filtered_round_robin_uses_only_available_indices(self):
        original = ArmCatalog().list_arms()
        policy = CMABPolicy(original, feature_dim=5, policy_name='round_robin')
        filtered = self.accelerator().filter_arms(original)
        policy.set_available_arms(filtered)
        selected = [policy.select_arm([0] * 5, epoch=i) for i in range(len(filtered))]
        self.assertEqual(set(selected), set(filtered))
        policy.set_available_arms(original)
        self.assertEqual(set(policy._round_robin_order), set(range(len(original))))

    def test_factorized_filter_refreshes_prediction_cache(self):
        original = ArmCatalog().list_arms()
        policy = FactorizedCMABPolicy(original, feature_dim=5)
        vocabulary = dict(policy._factor_catalogs)
        filtered = self.accelerator().filter_arms(original)
        policy.set_available_arms(filtered)
        self.assertEqual(len(policy._arm_factors), len(filtered))
        self.assertTrue(all(len(matrix) == len(filtered) for matrix in policy._main_factor_mat.values()))
        self.assertEqual(policy._factor_catalogs, vocabulary)
        policy.set_available_arms(original)
        self.assertEqual(len(policy._arm_factors), len(original))

    def test_mixed_space_enforces_lower_bound_in_output_and_warmup(self):
        space = MixedActionSpace()
        base = space.list_bases()[0]
        space.set_timeout_search_lo(200)
        for t in np.linspace(0, 300, 1201):
            emitted = decode_arm_params(space.make_arm(base, t))['fast_path_timeout']
            self.assertGreaterEqual(emitted, 201)
            self.assertLessEqual(emitted, 300)
        self.assertEqual(space.denormalize_timeout(0), 201)
        self.assertEqual(space.denormalize_timeout(1), 300)
        space.set_timeout_search_lo(None)
        self.assertEqual(decode_arm_params(space.make_arm(base, 110))['fast_path_timeout'], 110)

    def test_mixed_trainer_composes_and_restores_bounds(self):
        trainer = CMABTrainer.__new__(CMABTrainer)
        space = MixedActionSpace()
        trainer.policy = SimpleNamespace(mixed_space=space)
        trainer.arm_catalog = MixedArmCatalog(space)
        trainer.accelerator = self.accelerator(True)
        self.assertTrue(trainer._apply_accelerator())
        self.assertEqual((space.timeout_search_lo, space.timeout_search_hi), (0, 0))
        for value in (0, 100, 200, 300):
            self.assertEqual(decode_arm_params(space.make_arm(space.list_bases()[0], value))['fast_path_timeout'], 0)
        trainer.accelerator.enable_timeout_cap = False
        self.assertTrue(trainer._apply_accelerator())
        self.assertEqual((space.timeout_search_lo, space.timeout_search_hi), (201, 300))
        trainer.accelerator._applied_cap = 62
        self.assertTrue(trainer._apply_accelerator())
        self.assertEqual((space.timeout_search_lo, space.timeout_search_hi), (101, 300))
        trainer.accelerator._applied_cap = 400
        self.assertTrue(trainer._apply_accelerator())
        self.assertEqual((space.timeout_search_lo, space.timeout_search_hi), (0, 0))
        trainer.accelerator.enable_rtt_timeout_filter = False
        self.assertTrue(trainer._apply_accelerator())
        self.assertEqual((space.timeout_search_lo, space.timeout_search_hi), (0, 300))

    def test_empty_factorized_candidates_can_restore(self):
        original = ArmCatalog().list_arms()
        policy = FactorizedCMABPolicy(original, feature_dim=5)
        policy.set_available_arms([])
        self.assertEqual(policy._arms, [])
        self.assertTrue(all(len(m) == 0 for m in policy._main_factor_mat.values()))
        policy.set_available_arms(original)
        self.assertEqual(len(policy._arm_factors), len(original))

    def test_training_selects_and_writes_zero_timeout_after_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            original = ArmCatalog().list_arms()
            policy = CMABPolicy(original, feature_dim=5)
            trainer = CMABTrainer(
                metrics_dir=directory, parameters_file=str(Path(directory) / '.parameters.json'),
                checkpoint_dir=directory, policy=policy, context_builder=SimpleNamespace(),
                arm_catalog=ArmCatalog(), accelerator=self.accelerator(True),
            )
            file = Path(directory) / 'global_state_epoch_15.json'
            trainer._get_earliest_metrics_file = Mock(return_value=file)
            trainer._connect_param_socket = Mock()
            trainer._load_initial_arm_from_parameters_file = Mock(return_value=None)
            trainer._build_context_from_global_state = Mock(return_value=np.zeros(5))
            trainer._write_parameters_to_file = Mock()
            trainer._compute_shared_seed_hex = Mock(return_value='0' * 64)
            policy.select_arm = Mock(wraps=policy.select_arm)
            def stop_wait(*args, **kwargs):
                trainer.training_active = False
                return None
            trainer._wait_for_new_metrics_file = Mock(side_effect=stop_wait)
            with self.assertLogs('cmab.trainer', level='WARNING'):
                trainer.run(num_iterations=1, checkpoint_freq=1)
            self.assertTrue(policy._arms)
            self.assertTrue(all(decode_arm_params(a)['fast_path_timeout'] == 0 for a in policy._arms))
            policy.select_arm.assert_called_once()
            trainer._write_parameters_to_file.assert_called_once()
            self.assertEqual(trainer._write_parameters_to_file.call_args.args[0]['fast_path_timeout'], 0)
            trainer._wait_for_new_metrics_file.assert_called_once()

    def test_zero_fallback_reserved_for_sampled_one_hot_and_default_catalogs(self):
        from actions.action_encode import ActionCodec
        for codec in (ActionCodec(), ActionCodec(policy='default')):
            catalog = ArmCatalog(codec=codec, max_arms=1, seed=0, reserve_zero_timeout=True)
            original = catalog.list_arms()
            policy = CMABPolicy(original, feature_dim=5, action_encoding='one_hot')
            fallback = self.accelerator(True).filter_arms(original)
            policy.set_available_arms(fallback)
            self.assertTrue(fallback)
            self.assertTrue(all(decode_arm_params(a)['fast_path_timeout'] == 0 for a in fallback))
            self.assertEqual(catalog.list_arms(), original)
            self.assertEqual(len(set(fallback)), len(fallback))

    def test_benchmark_config_and_command_forwarding(self):
        params = dict(faults=0, nodes=4, workers=1, rate=50000, tx_size=512,
                      duration=10, simulate_partition=False, partition_nodes=0,
                      partition_start=0, partition_duration=0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(BenchParameters(params).enable_rtt_timeout_filter)
            self.assertTrue(BenchParameters(dict(params, enable_rtt_timeout_filter=True)).enable_rtt_timeout_filter)
            with self.assertRaises(ConfigError):
                BenchParameters(dict(params, rl_algo='dqn', enable_rtt_timeout_filter=True))
        cmd = CommandMaker.run_controller(repo_name="autopilot", enable_rtt_timeout_filter=True)
        self.assertIn('--enable-rtt-timeout-filter', cmd)
        self.assertNotIn('--enable-accelerator', cmd)
        self.assertNotIn('--enable-rtt-timeout-filter', CommandMaker.run_controller(repo_name="autopilot"))


if __name__ == '__main__':
    unittest.main()
