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
        'enable_rl': True,
        'rl_algo': 'cmab',
        'enable_checkpoint': False,
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
        self.assertEqual(parameters.cmab_seed, 0)
        self.assertFalse(parameters.enable_cmab_pairwise_residual_rf)

    def test_accepts_one_hot(self):
        parameters = BenchParameters(
            _parameters(cmab_action_encoding='one_hot')
        )

        self.assertEqual(parameters.cmab_action_encoding, 'one_hot')

    def test_rejects_unknown_encoding(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_action_encoding'):
            BenchParameters(_parameters(cmab_action_encoding='ordinal'))

    def test_accepts_non_negative_cmab_seed(self):
        parameters = BenchParameters(_parameters(cmab_seed=17))

        self.assertEqual(parameters.cmab_seed, 17)

    def test_rejects_negative_cmab_seed(self):
        with self.assertRaisesRegex(ConfigError, 'cmab_seed'):
            BenchParameters(_parameters(cmab_seed=-1))

    def test_accepts_pairwise_residual_rf_switch(self):
        parameters = BenchParameters(
            _parameters(enable_cmab_pairwise_residual_rf=True)
        )

        self.assertTrue(parameters.enable_cmab_pairwise_residual_rf)

    def test_rejects_non_boolean_pairwise_residual_rf_switch(self):
        with self.assertRaisesRegex(
            ConfigError, 'enable_cmab_pairwise_residual_rf'
        ):
            BenchParameters(
                _parameters(enable_cmab_pairwise_residual_rf='yes')
            )

    def test_controller_command_contains_encoding(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
            cmab_action_encoding='one_hot',
            cmab_seed=17,
            enable_cmab_pairwise_residual_rf=True,
        )

        self.assertIn('--cmab-action-encoding one_hot', command)
        self.assertIn('--cmab-seed 17', command)
        self.assertIn('--enable-cmab-pairwise-residual-rf', command)

    def test_controller_command_omits_pairwise_switch_by_default(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
        )

        self.assertNotIn('--enable-cmab-pairwise-residual-rf', command)


if __name__ == '__main__':
    import unittest

    unittest.main()
