from unittest import TestCase

from benchmark.commands import CommandMaker
from benchmark.config import BenchParameters, ConfigError


def _parameters(**overrides):
    values = {
        'faults': 0,
        'nodes': [4],
        'rate': [40_000],
        'workers': 1,
        'collocate': True,
        'tx_size': 512,
        'duration': 900,
        'runs': 1,
        'rl_algo': 'cmab',
        'simulate_partition': False,
        'partition_nodes': 2,
        'partition_start': 5,
        'partition_duration': 5,
        'enable_hotspot': False,
    }
    values.update(overrides)
    return values


class CMABActionEncodingConfigTests(TestCase):
    def test_numeric_is_backward_compatible_default(self):
        parameters = BenchParameters(_parameters())

        self.assertEqual(parameters.cmab_action_encoding, 'numeric')

    def test_accepts_one_hot(self):
        parameters = BenchParameters(
            _parameters(cmab_action_encoding='one_hot')
        )

        self.assertEqual(parameters.cmab_action_encoding, 'one_hot')

    def test_rejects_unknown_encoding(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_action_encoding'):
            BenchParameters(_parameters(cmab_action_encoding='ordinal'))

    def test_accepts_xgboost_algo(self):
        parameters = BenchParameters(_parameters(rl_algo='xgboost'))

        self.assertEqual(parameters.rl_algo, 'xgboost')

    def test_controller_command_contains_encoding(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            cmab_action_encoding='one_hot',
        )

        self.assertIn('--cmab-action-encoding one_hot', command)

    def test_cmab_seed_defaults_to_zero(self):
        parameters = BenchParameters(_parameters())

        self.assertEqual(parameters.cmab_seed, 0)

    def test_accepts_cmab_seed(self):
        parameters = BenchParameters(_parameters(cmab_seed=1))

        self.assertEqual(parameters.cmab_seed, 1)

    def test_rejects_negative_cmab_seed(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_seed'):
            BenchParameters(_parameters(cmab_seed=-1))

    def test_controller_command_contains_seed(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            cmab_seed=0,
        )

        self.assertIn('--cmab-seed 0', command)

    def test_factorized_reward_defaults_to_false(self):
        parameters = BenchParameters(_parameters())

        self.assertFalse(parameters.enable_factorized_reward)

    def test_accepts_enable_factorized_reward(self):
        parameters = BenchParameters(
            _parameters(enable_factorized_reward=True)
        )

        self.assertTrue(parameters.enable_factorized_reward)

    def test_controller_command_contains_factorized_flag(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            enable_factorized_reward=True,
        )

        self.assertIn('--enable-factorized-reward', command)

    def test_controller_command_omits_factorized_flag_when_disabled(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            enable_factorized_reward=False,
        )

        self.assertNotIn('--enable-factorized-reward', command)

    def test_cmab_policy_defaults_to_rf_ts(self):
        parameters = BenchParameters(_parameters())

        self.assertEqual(parameters.cmab_policy, 'rf_ts')

    def test_accepts_round_robin_policy(self):
        parameters = BenchParameters(_parameters(cmab_policy='round_robin'))

        self.assertEqual(parameters.cmab_policy, 'round_robin')

    def test_rejects_unknown_cmab_policy(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_policy'):
            BenchParameters(_parameters(cmab_policy='greedy'))

    def test_controller_command_contains_policy(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            cmab_policy='round_robin',
        )

        self.assertIn('--policy round_robin', command)

    def test_cmab_start_pos_defaults_to_zero(self):
        parameters = BenchParameters(_parameters())

        self.assertEqual(parameters.cmab_start_pos, 0)

    def test_accepts_cmab_start_pos(self):
        parameters = BenchParameters(_parameters(cmab_start_pos=39))

        self.assertEqual(parameters.cmab_start_pos, 39)

    def test_rejects_negative_cmab_start_pos(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_start_pos'):
            BenchParameters(_parameters(cmab_start_pos=-1))

    def test_controller_command_contains_start_pos(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            cmab_policy='round_robin',
            cmab_start_pos=39,
        )

        self.assertIn('--start-pos 39', command)


if __name__ == '__main__':
    import unittest

    unittest.main()
