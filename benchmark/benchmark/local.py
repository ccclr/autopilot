# Copyright(C) Facebook, Inc. and its affiliates.
import os
import socket
import subprocess
import shutil
from math import ceil
from os.path import basename, splitext
from time import sleep, time

from fabric import Connection
from benchmark.commands import CommandMaker
from benchmark.config import Key, LocalCommittee, NodeParameters, BenchParameters, ConfigError
from benchmark.logs import LogParser, ParseError
from benchmark.utils import Print, BenchError, PathMaker
from os.path import join


class LocalBench:
    BASE_PORT = 3000

    def __init__(self, bench_parameters_dict, node_parameters_dict):
        try:
            self.bench_parameters = BenchParameters(bench_parameters_dict)
            self.node_parameters = NodeParameters(node_parameters_dict)
        except ConfigError as e:
            raise BenchError('Invalid nodes or bench parameters', e)

    def __getattr__(self, attr):
        return getattr(self.bench_parameters, attr)

    def _background_run(self, command, log_file):
        name = splitext(basename(log_file))[0]
        # Use bash to ensure environment variables are available
        cmd = f'bash -c "{command}" 2> {log_file}'
        subprocess.run(['tmux', 'new', '-d', '-s', name, cmd], check=True)

    def _kill_nodes(self):
        try:
            cmd = CommandMaker.kill().split()
            subprocess.run(cmd, stderr=subprocess.DEVNULL)
        except subprocess.SubprocessError as e:
            raise BenchError('Failed to kill testbed', e)

    @staticmethod
    def _split_host_port(address):
        host, port = address.rsplit(':', 1)
        return host, int(port)

    def _wait_for_tcp_listeners(self, addresses, timeout_sec=30, check_interval_sec=0.1, label='services'):
        """
        Wait until every `host:port` in `addresses` accepts TCP connections.
        """
        assert isinstance(addresses, list)
        deadline = time() + timeout_sec
        remaining = set(addresses)

        while remaining and time() < deadline:
            ready = set()
            for address in remaining:
                host, port = self._split_host_port(address)
                try:
                    with socket.create_connection((host, port), timeout=0.2):
                        ready.add(address)
                except OSError:
                    pass
            remaining -= ready
            if remaining:
                sleep(check_interval_sec)

        if remaining:
            missing = ', '.join(sorted(remaining))
            raise BenchError(f'Timeout while waiting for {label} to start listening: {missing}')

    @staticmethod
    def _resolve_cargo_binary():
        # Prefer the project owner's rustup cargo in sudo/su sessions.
        home = os.path.expanduser('~')
        candidates = []
        preferred = '/home/ccclr0302/.cargo/bin/cargo'
        if preferred not in candidates:
            candidates.append(preferred)
        cargo = shutil.which('cargo')
        if cargo and cargo not in candidates:
            candidates.append(cargo)
        fallback = os.path.join(home, '.cargo', 'bin', 'cargo')
        if fallback not in candidates:
            candidates.append(fallback)

        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        raise BenchError(
            'Cannot find `cargo`. Install Rust toolchain and ensure cargo is in PATH '
            '(for example, source ~/.cargo/env).',
            FileNotFoundError('cargo')
        )

    @staticmethod
    def _cargo_env_for_binary(cargo_binary):
        env = os.environ.copy()
        # rustup chooses toolchains from RUSTUP_HOME, not cargo binary path.
        if cargo_binary.startswith('/home/ccclr0302/.cargo/bin'):
            env['CARGO_HOME'] = '/home/ccclr0302/.cargo'
            env['RUSTUP_HOME'] = '/home/ccclr0302/.rustup'
        return env

    def run(self, debug=False):
        assert isinstance(debug, bool)
        Print.heading('Starting local benchmark')

        # Kill any previous testbed.
        self._kill_nodes()
        
        try:
            Print.info('Setting up testbed...')
            local_requirements = os.path.abspath(join('..', 'Agent', 'requirements.txt'))
            if os.path.isfile(local_requirements):
                Print.info(f'Preparing Agent venv at {CommandMaker.AGENT_VENV_PATH}...')
                CommandMaker.ensure_agent_venv(local_requirements)
            nodes, rate = self.nodes[0], self.rate[0]

            # Cleanup all files.
            cmd = f'{CommandMaker.clean_logs()} ; {CommandMaker.cleanup()}'
            print('before run')
            subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)
            print('after run')
            sleep(0.5)  # Removing the store may take time.

            # Clean up previous metrics and latency matrices
            Print.info('Cleaning up previous metrics and latency matrices...')
            # Note: clean_metrics with node_id parameter is not needed here since we're cleaning shared files
            for i in range(nodes):
                cmd = CommandMaker.clean_metrics('autopilot', i)
                subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

            print('past cleanup')
            # Recompile the latest code.
            cmd = CommandMaker.compile().split()
            cmd[0] = self._resolve_cargo_binary()
            env = self._cargo_env_for_binary(cmd[0])
            subprocess.run(cmd, check=True, cwd=PathMaker.node_crate_path(), env=env)
            print('past compiled and cleaned metrics')

            # Create alias for the client and nodes binary.
            cmd = CommandMaker.alias_binaries(PathMaker.binary_path())
            subprocess.run([cmd], shell=True)

            # Generate configuration files.
            keys = []
            key_files = [PathMaker.key_file(i) for i in range(nodes)]
            for filename in key_files:
                cmd = CommandMaker.generate_key(filename).split()
                subprocess.run(cmd, check=True)
                keys += [Key.from_file(filename)]
            print('past keys')

            names = [x.name for x in keys]
            #print('num workers', self.workers)
            committee = LocalCommittee(names, self.BASE_PORT, self.workers)
            committee.print(PathMaker.committee_file())

            # Generate node-specific parameter files
            for i in range(nodes):
                self.node_parameters.print(PathMaker.local_parameters_file(i))

           # Run the clients (they will wait for the nodes to be ready).
            workers_addresses = committee.workers_addresses(self.faults)
            rate_share = ceil(rate / committee.workers())
            all_worker_nodes = [x for y in workers_addresses for _, x in y]

            hotspot_config = None
            if getattr(self, 'enable_hotspot', False):
                hotspot_config = {
                    'enable_hotspot': True,
                    'hotspot_windows': getattr(self, 'hotspot_windows', []),
                    'hotspot_nodes': getattr(self, 'hotspot_nodes', []),
                    'hotspot_regions': getattr(self, 'hotspot_regions', []),
                    'hotspot_region_rates': getattr(self, 'hotspot_region_rates', []),
                }
            # print(hotspot_config)

            # Run the primaries (except the faulty ones).
            for i, address in enumerate(committee.primary_addresses(self.faults)):
                cmd = CommandMaker.run_primary(
                    PathMaker.key_file(i),
                    PathMaker.committee_file(),
                    PathMaker.db_path(i),
                    PathMaker.local_parameters_file(0),
                    debug=debug,
                    node_index=i
                )
                log_file = PathMaker.primary_log_file(i)
                print(cmd)
                self._background_run(cmd, log_file)

            # Wait until all primaries are listening before proceeding.
            primary_addresses = committee.primary_addresses(self.faults)
            Print.info('Waiting for all primaries to be ready...')
            self._wait_for_tcp_listeners(primary_addresses, timeout_sec=60, label='primaries')

            # Run the workers (except the faulty ones).
            for i, addresses in enumerate(workers_addresses):
                for (id, address) in addresses:
                    cmd = CommandMaker.run_worker(
                        PathMaker.key_file(i),
                        PathMaker.committee_file(),
                        PathMaker.db_path(i, id),
                        PathMaker.local_parameters_file(0),
                        id,  # The worker's id.
                        debug=debug
                    )
                    log_file = PathMaker.worker_log_file(i, id)
                    self._background_run(cmd, log_file)

            # Wait until all workers transaction endpoints are listening.
            worker_listener_addresses = [addr for group in workers_addresses for (_, addr) in group]
            Print.info('Waiting for all workers to be ready...')
            self._wait_for_tcp_listeners(worker_listener_addresses, timeout_sec=60, label='workers')

            # Start controller for RL training.
            agent_python = CommandMaker.agent_venv_python()
            resume_from = getattr(self.bench_parameters, 'cmab_resume_from', None)
            rl_algo = getattr(self.bench_parameters, 'rl_algo', 'cmab')
            cmab_action_encoding = getattr(
                self.bench_parameters, 'cmab_action_encoding', 'numeric'
            )
            cmab_seed = getattr(self.bench_parameters, 'cmab_seed', 0)
            warmup_iterations = getattr(self.bench_parameters, 'rl_warmup_iterations', 5)
            enable_accelerator = getattr(self.bench_parameters, 'enable_accelerator', False)
            accelerator_period = getattr(self.bench_parameters, 'accelerator_period', 100)
            enable_factorized_reward = getattr(
                self.bench_parameters, 'enable_factorized_reward', False
            )
            cmab_policy = getattr(self.bench_parameters, 'cmab_policy', 'rf_ts')
            cmab_start_pos = getattr(self.bench_parameters, 'cmab_start_pos', 0)
            Print.info(f'RL algo: {rl_algo}')
            if rl_algo in ('cmab', 'xgboost'):
                Print.info(f'Action encoding: {cmab_action_encoding}')
            Print.info(f'CMAB seed: {cmab_seed}')
            Print.info(f'RL warmup iterations: {warmup_iterations}')
            Print.info(f'RL factorized reward: enabled={enable_factorized_reward}')
            Print.info(f'CMAB policy: {cmab_policy}')
            Print.info(f'CMAB start pos: {cmab_start_pos}')
            Print.info(f'RL accelerator: enabled={enable_accelerator} period={accelerator_period} epochs')
            if resume_from:
                Print.info(f'RL resume-from: {resume_from}')
            for i, address in enumerate(primary_addresses):
                cmd = CommandMaker.run_controller(
                    node_index=i,
                    repo_name='autopilot',
                    log_dir=os.path.abspath(PathMaker.logs_path()),
                    parameters_file=os.path.abspath(PathMaker.local_parameters_file(i)),
                    python_bin=agent_python,
                    resume_from=resume_from,
                    rl_algo=rl_algo,
                    cmab_action_encoding=(
                        cmab_action_encoding if rl_algo in ('cmab', 'xgboost') else None
                    ),
                    cmab_seed=cmab_seed,
                    warmup_iterations=warmup_iterations,
                    enable_accelerator=enable_accelerator,
                    accelerator_period=accelerator_period,
                    enable_factorized_reward=(
                        enable_factorized_reward if rl_algo == 'cmab' else None
                    ),
                    cmab_policy=(cmab_policy if rl_algo == 'cmab' else None),
                    cmab_start_pos=(cmab_start_pos if rl_algo == 'cmab' else None),
                )
                log_file = join(PathMaker.logs_path(), f'controller-{i}.log')
                self._background_run(cmd, log_file)
            sleep(2)

            # Now that primaries are running and sockets are created, start metrics collectors.
            Print.info('Starting metrics collectors...')
            epoch_slots = self.node_parameters.json.get('epoch_slots', 20)
            window_size = self.node_parameters.json.get('window_size', 5)
            for i, address in enumerate(primary_addresses):
                cmd = CommandMaker.run_metrics_collector(
                    epoch_slots=epoch_slots,
                    window_size=window_size,
                    node_index=i,
                    repo_name='autopilot',  # Use relative path from benchmark directory
                    log_dir=os.path.abspath(PathMaker.logs_path()),
                    parameters_file=os.path.abspath(PathMaker.local_parameters_file(0)),
                    python_bin=agent_python,
                )
                log_file = join(PathMaker.logs_path(), f'metrics_collector-{i}.log')
                self._background_run(cmd, log_file)
            sleep(2)

            # Fix socket permissions to allow metrics_collector to connect.
            Print.info('Fixing socket permissions for metrics collection...')
            for node_idx, address in enumerate(primary_addresses):
                socket_path = f'/tmp/autopilot_core_{node_idx}.sock'
                cmd = f'chmod 666 {socket_path} 2>/dev/null || true'
                try:
                    subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)
                    Print.info(f'Set permissions for socket: {socket_path}')
                except Exception as e:
                    Print.warn(f'Failed to set socket permissions for node {node_idx}: {e}')

            # Start clients only after all primaries/workers are ready.
            for i, addresses in enumerate(workers_addresses):
                for (id, address) in addresses:
                    cmd = CommandMaker.run_client(
                        address,
                        self.tx_size,
                        rate_share,
                        all_worker_nodes,
                        node_id=i,
                        hotspot_config=hotspot_config
                    )
                    log_file = PathMaker.client_log_file(i, id)
                    self._background_run(cmd, log_file)
            print('past workers')

            # Wait for all transactions to be processed.
            Print.info(f'Running benchmark ({self.duration} sec)...')
            sleep(self.duration)
            self._kill_nodes()

            # Parse logs and return the parser.
            Print.info('Parsing logs...')
            return LogParser.process(PathMaker.logs_path(), faults=self.faults)

        except (subprocess.SubprocessError, ParseError) as e:
            self._kill_nodes()
            raise BenchError('Failed to run benchmark', e)