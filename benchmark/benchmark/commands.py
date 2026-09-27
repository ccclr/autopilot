# Copyright(C) Facebook, Inc. and its affiliates.
import os
import shlex
import subprocess
from os.path import join

from benchmark.utils import PathMaker


class CommandMaker:
    # GCP login home. Bench.__init__ overrides this via set_home().
    HOME = f"/home/{os.environ.get('USER', 'ccclr0302')}"
    AGENT_VENV_PATH = f'{HOME}/autopilot-venv'

    @classmethod
    def set_home(cls, home):
        """Set the remote/local home directory used by path-dependent commands."""
        assert isinstance(home, str) and home
        cls.HOME = home.rstrip('/')
        cls.AGENT_VENV_PATH = f'{cls.HOME}/autopilot-venv'

    @staticmethod
    def agent_venv_python():
        return f'{CommandMaker.AGENT_VENV_PATH}/bin/python'

    @staticmethod
    def agent_venv_pip():
        return f'{CommandMaker.AGENT_VENV_PATH}/bin/pip'

    @staticmethod
    def agent_python(python_bin=None):
        if python_bin:
            return python_bin
        venv_python = CommandMaker.agent_venv_python()
        if os.path.isfile(venv_python):
            return venv_python
        return 'python3'

    @staticmethod
    def _remove_tree(path):
        """Remove a tree, using sudo when a previous user owns it."""
        import shutil

        try:
            shutil.rmtree(path)
        except OSError:
            subprocess.run(['sudo', '-n', 'rm', '-rf', path], check=True)

    @staticmethod
    def _venv_writable(venv_path):
        bin_dir = os.path.join(venv_path, 'bin')
        return os.path.isdir(venv_path) and os.access(venv_path, os.W_OK) and os.access(bin_dir, os.W_OK)

    @staticmethod
    def ensure_agent_venv(requirements_file):
        """Create the Agent venv locally and install dependencies."""
        venv_path = CommandMaker.AGENT_VENV_PATH
        venv_python = CommandMaker.agent_venv_python()
        # A root-owned venv still has a python binary, but pip cannot replace it.
        unusable = os.path.isdir(venv_path) and (
            not os.path.isfile(venv_python) or not CommandMaker._venv_writable(venv_path)
        )
        if unusable:
            CommandMaker._remove_tree(venv_path)
        if not os.path.isdir(venv_path):
            subprocess.run(
                ['python3', '-m', 'venv', venv_path],
                check=True,
                capture_output=True,
                text=True,
            )
        # Prefer `python -m pip` so we do not depend on a bin/pip symlink.
        try:
            subprocess.run(
                [venv_python, '-m', 'ensurepip', '--upgrade'],
                check=False,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [venv_python, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [venv_python, '-m', 'pip', 'install', '-q', '-r', requirements_file],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or e.stdout or str(e)).strip()
            raise RuntimeError(
                f'Failed to install Agent dependencies into {venv_path}: {detail}'
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f'Agent venv python missing at {venv_python}; '
                f'install python3-venv and recreate {venv_path}'
            ) from e
        return CommandMaker.agent_venv_python()

    @staticmethod
    def remote_agent_venv_setup_cmds(repo_name):
        """Shell commands to create the Agent venv on a remote host."""
        venv_path = CommandMaker.AGENT_VENV_PATH
        venv_python = CommandMaker.agent_venv_python()
        # Parenthesize `|| true` so it cannot break a longer `cmd1 && cmd2 && ...` chain.
        return [
            'sudo apt-get update -qq',
            'sudo apt-get -y -qq install python3-pip python3-venv',
            (
                f'if [ ! -x {venv_python} ] || [ ! -w {venv_path} ]; then '
                f'rm -rf {venv_path} 2>/dev/null || sudo -n rm -rf {venv_path}; '
                f'python3 -m venv {venv_path}; fi'
            ),
            f'({venv_python} -m ensurepip --upgrade || true)',
            f'{venv_python} -m pip install -q --upgrade pip',
            (
                f'(cd {CommandMaker.HOME}/{repo_name} && '
                f'{venv_python} -m pip install -q -r Agent/requirements.txt)'
            ),
        ]

    @staticmethod
    def cleanup():
        return (
            f'rm -r .db-* ; rm .*.json ; mkdir -p {PathMaker.results_path()}'
        )

    @staticmethod
    def clean_logs():
        return f'rm -r {PathMaker.logs_path()} ; mkdir -p {PathMaker.logs_path()}'

    @staticmethod
    def clean_metrics(repo_name, node_id):
        home = CommandMaker.HOME
        return (
            f'rm -f {home}/{repo_name}/benchmark/latency_full_matrix_*.npy ; '
            f'rm -f {home}/{repo_name}/benchmark/latency_region_matrix_*.npy ; '
            f'rm -f {home}/{repo_name}/benchmark/latency_vector_*.npy ; '
            # f'rm -rf {home}/{repo_name}/metrics-{node_id}; '
            f'rm -f /tmp/autopilot_rl_param_*.sock ; '
            f'rm -f /tmp/autopilot_rl_param_abandon_*.signal ; '
            f'rm -f /tmp/autopilot_core_*.sock ; '
            f'rm -f /tmp/autopilot_controller_*.sock ; '
            f'sudo rm -rf {home}/metrics-* {home}/{repo_name}/metrics-* || true'
        )

    @staticmethod
    def compile():
        return 'cargo build --quiet --release --features benchmark'

    @staticmethod
    def generate_key(filename):
        assert isinstance(filename, str)
        return f'./node generate_keys --filename {filename}'

    @staticmethod
    def run_primary(keys, committee, store, parameters, debug=False, node_index=None):
        assert isinstance(keys, str)
        assert isinstance(committee, str)
        assert isinstance(parameters, str)
        assert isinstance(debug, bool)
        v = '-vvv' if debug else '-vv'
        socket_env = f'RUST_STATE_SOCKET_PATH=/tmp/autopilot_core_{node_index}.sock' if node_index is not None else ''
        cmd = f'./node {v} run --keys {keys} --committee {committee} '
        cmd += f'--store {store} --parameters {parameters} primary'
        if socket_env:
            cmd = f'{socket_env} {cmd}'
        return cmd

    @staticmethod
    def run_worker(keys, committee, store, parameters, id, debug=False):
        assert isinstance(keys, str)
        assert isinstance(committee, str)
        assert isinstance(parameters, str)
        assert isinstance(debug, bool)
        v = '-vvv' if debug else '-vv'
        return (f'./node {v} run --keys {keys} --committee {committee} '
                f'--store {store} --parameters {parameters} worker --id {id}')

    @staticmethod
    def run_client(address, size, rate, nodes, node_id=None, hotspot_config=None):
        assert isinstance(address, str)
        assert isinstance(size, int) and size > 0
        assert isinstance(rate, int) and rate >= 0
        assert isinstance(nodes, list)
        assert all(isinstance(x, str) for x in nodes)
        
        # Build base command
        nodes_str = f'--nodes {" ".join(nodes)}' if nodes else ''
        cmd = f'./benchmark_client {address} --size {size} --rate {rate} {nodes_str}'
        
        cmd += f' --node-id {node_id}'
        
        # Add hotspot configuration if provided
        if hotspot_config and hotspot_config.get('enable_hotspot'):
            windows = hotspot_config.get('hotspot_windows', [])
            hotspot_node_ids_per_window = hotspot_config.get('hotspot_node_ids_per_window', [])
            hotspot_node_rates_per_window = hotspot_config.get('hotspot_node_rates_per_window', [])

            if windows:
                windows_str = ' '.join([f'{w[0]}:{w[1]}' for w in windows])
                cmd += f' --hotspot-windows {windows_str}'

                # Region-based: pass resolved node ids per window (format "2,5|3,4")
                if hotspot_node_ids_per_window:
                    ids_str = '|'.join([','.join(str(i) for i in ids) for ids in hotspot_node_ids_per_window])
                    cmd += f' --hotspot-node-ids "{ids_str}"'
                    if hotspot_node_rates_per_window:
                        rates_by_node_str = '|'.join(
                            [','.join(str(float(r)) for r in rates) for rates in hotspot_node_rates_per_window]
                        )
                        cmd += f' --hotspot-node-rates "{rates_by_node_str}"'
                else:
                    # Original: pass hotspot node count per window
                    hotspot_nodes = hotspot_config.get('hotspot_nodes', [])
                    if hotspot_nodes:
                        def _hotspot_count(n):
                            return sum(n) if isinstance(n, list) else n
                        nodes_str = ' '.join([str(_hotspot_count(n)) for n in hotspot_nodes])
                        cmd += f' --hotspot-nodes {nodes_str}'

        return cmd

    @staticmethod
    def kill():
        return 'tmux kill-server'

    @staticmethod
    def alias_binaries(origin):
        assert isinstance(origin, str)
        node, client = join(origin, 'node'), join(origin, 'benchmark_client')
        # -f -n replaces an existing symlink or file. && keeps this from
        # running when an earlier step in the same chain failed.
        return f'ln -sfn {node} node && ln -sfn {client} benchmark_client'

    @staticmethod
    def run_metrics_collector(epoch_slots, window_size, node_index=None, repo_name=None, log_dir=None, parameters_file=None, python_bin=None, metrics_dir=None):
        """Generate command to run metrics_collector as a background process"""
        assert isinstance(epoch_slots, int) and epoch_slots > 0
        assert isinstance(window_size, int) and window_size > 0
        # Use relative path from benchmark directory (../Agent/metrics_collector.py)
        # or absolute path if repo_name is provided
        metrics_collector_path = join(CommandMaker.HOME, repo_name, 'Agent/metrics_collector.py')
        socket_path = f'/tmp/autopilot_core_{node_index}.sock'
        python = CommandMaker.agent_python(python_bin)
        cmd = f'RUST_STATE_SOCKET_PATH={socket_path} {python} {metrics_collector_path}'
        cmd += f' {epoch_slots} {window_size}'
        cmd += f' --node-index {node_index}'
        cmd += f' --log-dir {log_dir}'
        cmd += f' --parameters-file {parameters_file}'
        cmd += ' --metrics-dir ' + shlex.quote(str(metrics_dir or f'{CommandMaker.HOME}/metrics-{node_index}'))
        return cmd

    @staticmethod
    def run_action_receiver(
        node_index=None,
        repo_name=None,
        parameters_file=None,
        python_bin=None,
        bind_host='0.0.0.0',
        port=19100,
    ):
        """Generate the per-node receiver used by centralized DQN."""
        receiver_path = (
            join(CommandMaker.HOME, repo_name, 'Agent/rl/controllers/action_receiver.py')
        )
        python = CommandMaker.agent_python(python_bin)
        cmd = f'{python} {receiver_path}'
        cmd += f' --node-index {int(node_index)}'
        cmd += f' --parameters-file {shlex.quote(str(parameters_file))}'
        cmd += f' --bind-host {shlex.quote(str(bind_host))}'
        cmd += f' --port {int(port)}'
        return cmd

    @staticmethod
    def run_controller(
        node_index=None,
        repo_name=None,
        log_dir=None,
        parameters_file=None,
        python_bin=None,
        resume_from=None,
        rl_algo=None,
        cmab_action_encoding=None,
        cmab_seed=None,
        warmup_iterations=None,
        enable_accelerator=None,
        accelerator_period=None,
        cmab_policy=None,
        cmab_start_pos=None,
        max_training_iterations=None,
        dqn_action_endpoints=None,
        dqn_action_timeout=None,
        dqn_action_retries=None,
        dqn_learning_rate=None,
        dqn_gamma=None,
        dqn_replay_capacity=None,
        dqn_batch_size=None,
        dqn_learning_starts=None,
        dqn_target_update_interval=None,
        dqn_epsilon_start=None,
        dqn_epsilon_end=None,
        dqn_epsilon_decay_steps=None,
        dqn_gradient_updates=None,
        dqn_gradient_clip=None,
        dqn_hidden_dim=None,
        dqn_seed=None,
        dqn_checkpoint_load_mode=None,
        enable_cmab_cut_fpt_cross_feature=False,
        cmab_fixed_k=None,
        cmab_factor_freeze_samples=0,
        cmab_factor_freeze_margin_ms=30.0,
        cmab_transition_export_dir=None,
        cmab_environment_label=None,
        cmab_transition_run_id=None,
        metrics_dir=None,
    ):
        """Generate command to run controller as a background process"""
        controller_path = join(CommandMaker.HOME, repo_name, 'Agent/rl/controllers/controller.py')
        # update_parameters_path = f'{CommandMaker.HOME}/{repo_name}/Agent/update_parameters_{node_index}.json'
        python = CommandMaker.agent_python(python_bin)
        cmd = f'{python} {controller_path}'
        cmd += ' --metrics-dir ' + shlex.quote(str(metrics_dir or f'{CommandMaker.HOME}/metrics-{node_index}'))
        cmd += f' --node-index {node_index}'
        cmd += f' --log-dir {log_dir}'
        cmd += f' --parameters-file {parameters_file}'
        if rl_algo:
            cmd += f' --rl-algo {rl_algo}'
        if cmab_action_encoding:
            cmd += (
                ' --cmab-action-encoding '
                f'{shlex.quote(str(cmab_action_encoding))}'
            )
        if cmab_seed is not None:
            cmd += f' --cmab-seed {int(cmab_seed)}'
        if resume_from:
            cmd += f' --resume-from {resume_from}'
        if warmup_iterations is not None:
            cmd += f' --warmup-iterations {int(warmup_iterations)}'
        if enable_accelerator:
            cmd += ' --enable-accelerator'
        if cmab_policy:
            cmd += f' --policy {shlex.quote(str(cmab_policy))}'
        if cmab_start_pos is not None:
            cmd += f' --start-pos {int(cmab_start_pos)}'
        if accelerator_period is not None:
            cmd += f' --accelerator-period {int(accelerator_period)}'
        if enable_cmab_cut_fpt_cross_feature:
            cmd += ' --enable-cmab-cut-fpt-cross-feature'
        if cmab_fixed_k is not None:
            cmd += f' --fixed-k {int(cmab_fixed_k)}'
        if cmab_factor_freeze_samples:
            cmd += f' --factor-freeze-samples {int(cmab_factor_freeze_samples)}'
            cmd += f' --factor-freeze-margin-ms {float(cmab_factor_freeze_margin_ms)}'
        if cmab_transition_export_dir:
            cmd += (
                ' --cmab-transition-export-dir '
                f'{shlex.quote(str(cmab_transition_export_dir))}'
            )
        if cmab_environment_label:
            cmd += (
                ' --cmab-environment-label '
                f'{shlex.quote(str(cmab_environment_label))}'
            )
        if cmab_transition_run_id:
            cmd += (
                ' --cmab-transition-run-id '
                f'{shlex.quote(str(cmab_transition_run_id))}'
            )
        if max_training_iterations is not None:
            cmd += f' --max-training-iterations {int(max_training_iterations)}'
        if dqn_action_endpoints:
            cmd += (
                ' --dqn-action-endpoints '
                f'{shlex.quote(str(dqn_action_endpoints))}'
            )
        if dqn_action_timeout is not None:
            cmd += f' --dqn-action-timeout {float(dqn_action_timeout)}'
        if dqn_action_retries is not None:
            cmd += f' --dqn-action-retries {int(dqn_action_retries)}'
        if dqn_learning_rate is not None:
            cmd += f' --dqn-learning-rate {float(dqn_learning_rate)}'
        if dqn_gamma is not None:
            cmd += f' --dqn-gamma {float(dqn_gamma)}'
        if dqn_replay_capacity is not None:
            cmd += f' --dqn-replay-capacity {int(dqn_replay_capacity)}'
        if dqn_batch_size is not None:
            cmd += f' --dqn-batch-size {int(dqn_batch_size)}'
        if dqn_learning_starts is not None:
            cmd += f' --dqn-learning-starts {int(dqn_learning_starts)}'
        if dqn_target_update_interval is not None:
            cmd += (
                ' --dqn-target-update-interval '
                f'{int(dqn_target_update_interval)}'
            )
        if dqn_epsilon_start is not None:
            cmd += f' --dqn-epsilon-start {float(dqn_epsilon_start)}'
        if dqn_epsilon_end is not None:
            cmd += f' --dqn-epsilon-end {float(dqn_epsilon_end)}'
        if dqn_epsilon_decay_steps is not None:
            cmd += f' --dqn-epsilon-decay-steps {int(dqn_epsilon_decay_steps)}'
        if dqn_gradient_updates is not None:
            cmd += f' --dqn-gradient-updates {int(dqn_gradient_updates)}'
        if dqn_gradient_clip is not None:
            cmd += f' --dqn-gradient-clip {float(dqn_gradient_clip)}'
        if dqn_hidden_dim is not None:
            cmd += f' --dqn-hidden-dim {int(dqn_hidden_dim)}'
        if dqn_seed is not None:
            cmd += f' --dqn-seed {int(dqn_seed)}'
        if dqn_checkpoint_load_mode:
            cmd += (
                ' --dqn-checkpoint-load-mode '
                f'{shlex.quote(str(dqn_checkpoint_load_mode))}'
            )
        return cmd