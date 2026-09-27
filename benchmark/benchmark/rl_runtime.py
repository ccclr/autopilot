"""Shared options and receiver startup for local, GCP and CloudLab runs."""
from .commands import CommandMaker
from .config import ConfigError

DQN_OPTIONS = (
    'dqn_action_timeout', 'dqn_action_retries', 'dqn_learning_rate', 'dqn_gamma',
    'dqn_replay_capacity', 'dqn_batch_size', 'dqn_learning_starts',
    'dqn_target_update_interval', 'dqn_epsilon_start', 'dqn_epsilon_end',
    'dqn_epsilon_decay_steps', 'dqn_gradient_updates', 'dqn_gradient_clip',
    'dqn_hidden_dim', 'dqn_seed', 'dqn_checkpoint_load_mode',
)


def controller_options(parameters, node_index, endpoints=None):
    options = {
        'enable_cmab_cut_fpt_cross_feature': parameters.enable_cmab_cut_fpt_cross_feature,
        'cmab_fixed_k': parameters.cmab_fixed_k,
        'cmab_factor_freeze_samples': parameters.cmab_factor_freeze_samples,
        'cmab_factor_freeze_margin_ms': parameters.cmab_factor_freeze_margin_ms,
        'max_training_iterations': parameters.rl_max_training_iterations,
    }
    if parameters.rl_algo == 'dqn':
        options.update({name: getattr(parameters, name) for name in DQN_OPTIONS})
        options['dqn_action_endpoints'] = endpoints
    if parameters.enable_cmab_transition_export and node_index == 0:
        options.update(
            cmab_transition_export_dir=parameters.cmab_transition_export_dir,
            cmab_environment_label=parameters.cmab_environment_label,
        )
    return options


def start_dqn_receivers(parameters, hosts, *, repo_name, python_bin,
                        parameters_file, launch, wait, local=False):
    """Start one receiver per primary, including node0, before its trainer.

    launch(index, host, command) and wait(addresses) are backend callbacks.
    """
    if parameters.rl_algo != 'dqn':
        return None
    if not local and len(set(hosts)) != len(hosts):
        raise ConfigError('DQN remote runs require one primary per host')
    ports = [parameters.dqn_action_port + (i if local else 0)
             for i in range(len(hosts))]
    if ports and max(ports) > 65535:
        raise ConfigError('DQN receiver port range exceeds 65535')
    endpoints = []
    for i, (host, port) in enumerate(zip(hosts, ports)):
        command = CommandMaker.run_action_receiver(
            node_index=i, repo_name=repo_name, parameters_file=parameters_file(i),
            python_bin=python_bin, bind_host='127.0.0.1' if local else '0.0.0.0',
            port=port,
        )
        launch(i, host, command)
        endpoints.append(f'{i}@{host}:{port}')
    wait([f'{host}:{port}' for host, port in zip(hosts, ports)])
    return ','.join(endpoints)
