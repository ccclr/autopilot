import shlex
from unittest import TestCase
from unittest.mock import Mock
from test_cmab_action_encoding_config import _parameters
from benchmark.config import BenchParameters, ConfigError
from benchmark.commands import CommandMaker
from benchmark.rl_runtime import controller_options, start_dqn_receivers


class DQNRuntimeTests(TestCase):
    def params(self, **kwargs):
        return BenchParameters(_parameters(**kwargs))

    def test_dqn_options_reach_controller(self):
        parameters = self.params(rl_algo='dqn', dqn_checkpoint_load_mode='finetune', dqn_seed=9)
        options = controller_options(parameters, 0, '0@localhost:19100')
        command = CommandMaker.run_controller(node_index=0, repo_name='autopilot',
                    parameters_file='/tmp/params', log_dir='/tmp/logs', rl_algo='dqn', **options)
        tokens = shlex.split(command)
        self.assertEqual(tokens[tokens.index('--dqn-checkpoint-load-mode') + 1], 'finetune')
        self.assertEqual(tokens[tokens.index('--dqn-seed') + 1], '9')
        self.assertEqual(tokens[tokens.index('--dqn-action-endpoints') + 1], '0@localhost:19100')

    def test_local_receivers_have_unique_ports_and_node_parameters(self):
        launch, wait = Mock(), Mock()
        endpoints = start_dqn_receivers(self.params(rl_algo='dqn'), ['127.0.0.1'] * 4,
            repo_name='/tmp/worktree', python_bin='/tmp/python',
            parameters_file=lambda i: f'/tmp/params-{i}.json', launch=launch, wait=wait, local=True)
        self.assertEqual(launch.call_count, 4)
        self.assertIn('3@127.0.0.1:19103', endpoints)
        self.assertIn('/tmp/worktree/Agent/rl/controllers/action_receiver.py', launch.call_args.args[2])
        self.assertIn('/tmp/params-3.json', launch.call_args.args[2])
        wait.assert_called_once_with([f'127.0.0.1:{19100+i}' for i in range(4)])

    def test_cmab_does_not_start_receivers(self):
        launch, wait = Mock(), Mock()
        self.assertIsNone(start_dqn_receivers(self.params(), ['host'], repo_name='autopilot',
            python_bin='python', parameters_file=lambda i: 'params', launch=launch, wait=wait))
        launch.assert_not_called()
        wait.assert_not_called()

    def test_export_only_on_node_zero(self):
        parameters = self.params(enable_cmab_transition_export=True)
        self.assertIn('cmab_transition_export_dir', controller_options(parameters, 0))
        self.assertNotIn('cmab_transition_export_dir', controller_options(parameters, 1))

    def test_cross_feature_preserves_gcp_one_hot_option(self):
        parameters = self.params(enable_cmab_cut_fpt_cross_feature=True, cmab_action_encoding='one_hot')
        self.assertTrue(parameters.enable_cmab_cut_fpt_cross_feature)
        self.assertEqual(parameters.cmab_action_encoding, 'one_hot')

    def test_invalid_options_rejected(self):
        for options in ({'dqn_gamma': 2}, {'dqn_batch_size': 3000},
                        {'dqn_checkpoint_load_mode': 'bad'}, {'dqn_action_port': 65536},
                        {'enable_cmab_cut_fpt_cross_feature': 'true'},
                        {'enable_cmab_cut_fpt_cross_feature': True, 'cmab_policy': 'combined'}):
            with self.subTest(options=options), self.assertRaises(ConfigError):
                self.params(**options)
