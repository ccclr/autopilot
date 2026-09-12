# Copyright(C) Facebook, Inc. and its affiliates.
import os
import shutil
import socket
from collections import OrderedDict
from fabric import Connection, ThreadingGroup as Group
from fabric.exceptions import GroupException
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException
from os.path import basename, splitext, join
from time import sleep, time
from math import ceil
from copy import deepcopy
import subprocess

from benchmark.config import Committee, Key, NodeParameters, BenchParameters, ConfigError
from benchmark.utils import BenchError, Print, PathMaker, progress_bar
from benchmark.commands import CommandMaker
from benchmark.logs import LogParser, ParseError
from benchmark.gcp_instance import InstanceManager


class FabricError(Exception):
    ''' Wrapper for Fabric exception with a meaningfull error message. '''

    def __init__(self, error):
        assert isinstance(error, GroupException)
        message = list(error.result.values())[-1]
        super().__init__(message)


class ExecutionError(Exception):
    pass


class Bench:
    def __init__(self, ctx):
        self.manager = InstanceManager.make()
        self.settings = self.manager.settings
        self.home = f'/home/{self.settings.username}'
        CommandMaker.set_home(self.home)
        try:
            ctx.connect_kwargs.pkey = RSAKey.from_private_key_file(
                self.manager.settings.key_path
            )
            self.connect = ctx.connect_kwargs
        except (IOError, PasswordRequiredException, SSHException) as e:
            raise BenchError('Failed to load SSH key', e)

    def _check_stderr(self, output):
        if isinstance(output, dict):
            for x in output.values():
                if x.stderr:
                    raise ExecutionError(x.stderr)
        else:
            if output.stderr:
                raise ExecutionError(output.stderr)

    def _resolve_cargo_binary(self):
        # Prefer the project owner's rustup cargo in sudo/su sessions.
        home = self.home
        candidates = []
        preferred = f'{home}/.cargo/bin/cargo'
        if preferred not in candidates:
            candidates.append(preferred)
        cargo = shutil.which('cargo')
        if cargo and cargo not in candidates:
            candidates.append(cargo)
        fallback = os.path.join(os.path.expanduser('~'), '.cargo', 'bin', 'cargo')
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

    def _cargo_env_for_binary(self, cargo_binary):
        env = os.environ.copy()
        # rustup chooses toolchains from RUSTUP_HOME, not cargo binary path.
        cargo_home = f'{self.home}/.cargo'
        if cargo_binary.startswith(f'{cargo_home}/bin'):
            env['CARGO_HOME'] = cargo_home
            env['RUSTUP_HOME'] = f'{self.home}/.rustup'
        return env

    def install(self):
        EXCLUDED_ZONES = ['us-central1-c'] 
        manager = InstanceManager.make()
        settings = manager.settings
        hosts_dict = manager.hosts()

        filtered_hosts = {
            region: nodes for region, nodes in hosts_dict.items()
            if region.lower() not in EXCLUDED_ZONES
        }

        all_nodes = [ip for nodes in filtered_hosts.values() for ip in nodes]
        # Print(all_nodes)

        if not all_nodes:
            print("No hosts remaining after filtering.")
            return
        Print.info('Installing rust and cloning the repo...')
        cmd = [
            'sudo sed -i "/bullseye-backports/d" /etc/apt/sources.list',
            'sudo apt-get update',
            'sudo apt-get -y upgrade',
            'sudo apt-get -y autoremove',
            'sudo apt install -y tmux',

            'sudo apt-get -y install build-essential',
            'sudo apt-get -y install cmake',
            'sudo apt-get -y install clang git curl',
            'sudo apt-get install -y iperf3 python3-pip python3-venv',

            'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y',
            f'source {self.home}/.cargo/env',
            'rustup default stable',

            f'(git clone {self.settings.repo_url} {self.settings.repo_name} || (cd {self.settings.repo_name} && git pull --rebase))',
            f'(cd {self.settings.repo_name} && git checkout {self.settings.branch})',

            f'(cd {self.settings.repo_name} && source {self.home}/.cargo/env && {CommandMaker.compile()})',

            f'ln -sf {self.settings.repo_name}/target/release/node ~/node',
            f'ln -sf {self.settings.repo_name}/target/release/benchmark_client ~/benchmark_client',

            # Agent venv on every remote node (shared helper).
            *CommandMaker.remote_agent_venv_setup_cmds(self.settings.repo_name),
        ]

        hosts = all_nodes
        print(hosts)
        try:
            g = Group(*hosts, user=self.settings.username, connect_kwargs=self.connect)
            g.run(' && '.join(cmd), hide=True)
            Print.heading(f'Initialized testbed of {len(hosts)} nodes')
            Print.info(f'Agent venv ready on remotes: {CommandMaker.AGENT_VENV_PATH}')
        except (GroupException, ExecutionError) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to install repo on testbed', e)

    def kill(self, hosts=[], delete_logs=False):
        assert isinstance(hosts, list)
        assert isinstance(delete_logs, bool)
        hosts = hosts if hosts else self.manager.hosts(flat=True)
        delete_logs = CommandMaker.clean_logs() if delete_logs else 'true'
        cmd = [delete_logs, f'({CommandMaker.kill()} || true)']
        try:
            g = Group(*hosts, user=self.settings.username, connect_kwargs=self.connect)
            g.run(' && '.join(cmd), hide=True)
        except GroupException as e:
            raise BenchError('Failed to kill nodes', FabricError(e))

    def _select_hosts(self, bench_parameters):
        """Returns (hosts, node_regions). node_regions[i] = region for node i (collocate only)."""
        if bench_parameters.collocate:
            nodes = max(bench_parameters.nodes)
            EXCLUDED_ZONES = []
            hosts_dict = self.manager.hosts()
            filtered_hosts = {
                region: ips for region, ips in hosts_dict.items()
                if region.lower() not in EXCLUDED_ZONES
            }
            ordered = []
            node_regions = []
            for region, ips in filtered_hosts.items():
                for ip in ips:
                    ordered.append(ip)
                    node_regions.append(region)
            if len(ordered) < nodes:
                Print.warn(f"Not enough hosts after excluding zones: {len(ordered)} < {nodes}")
                return [], []
            return ordered[:nodes], node_regions[:nodes]
        else:
            primaries = max(bench_parameters.nodes)
            total_needed = primaries * (bench_parameters.workers + 1)
            hosts_dict = self.manager.hosts()
            all_nodes = [ip for ips in hosts_dict.values() for ip in ips]
            all_nodes = sorted(all_nodes, key=lambda ip: tuple(int(x) for x in ip.split('.')))
            if len(all_nodes) < total_needed:
                Print.warn(f"Not enough hosts: {len(all_nodes)} < {total_needed}")
                return [], []
            selected = []
            for i in range(primaries):
                group = all_nodes[i*(bench_parameters.workers+1):(i+1)*(bench_parameters.workers+1)]
                selected.append(group)
            return selected, []

    def _select_hosts_config(self, bench_parameters):
        if bench_parameters.collocate:
            nodes = max(bench_parameters.nodes)
            hosts = self.manager.internal_hosts()
            if sum(len(x) for x in hosts.values()) < nodes:
                return []
            ordered = [x for y in hosts.values() for x in y]
            assert len(ordered) >= nodes, f"Not enough hosts: got {len(ordered)}, need {nodes}"
            return ordered[:nodes]
        else:
            primaries = max(bench_parameters.nodes)
            total_needed = primaries * (bench_parameters.workers + 1)
            hosts = self.manager.internal_hosts()
            all_nodes = [ip for nodes in hosts.values() for ip in nodes]
            all_nodes = sorted(all_nodes, key=lambda ip: tuple(int(x) for x in ip.split('.')))
            if len(all_nodes) < total_needed:
                return []
            selected = []
            for i in range(primaries):
                group = all_nodes[i*(bench_parameters.workers+1):(i+1)*(bench_parameters.workers+1)]
                selected.append(group)
            return selected



    def _background_run(self, host, command, log_file):
        name = splitext(basename(log_file))[0]
        cmd = f'tmux new -d -s "{name}" "{command} |& tee {log_file}"'
        c = Connection(host, user=self.settings.username, connect_kwargs=self.connect)
        output = c.run(cmd, hide=True)
        self._check_stderr(output)

    @staticmethod
    def _split_host_port(address):
        host, port = address.rsplit(':', 1)
        return host, int(port)

    def _wait_for_tcp_listeners(self, addresses, timeout_sec=60, check_interval_sec=0.2, label='services'):
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
                    with socket.create_connection((host, port), timeout=0.4):
                        ready.add(address)
                except OSError:
                    pass
            remaining -= ready
            if remaining:
                sleep(check_interval_sec)

        if remaining:
            missing = ', '.join(sorted(remaining))
            raise BenchError(f'Timeout while waiting for {label} to start listening: {missing}', TimeoutError(missing))

    @staticmethod
    def _flatten_hosts(hosts, collocate):
        if collocate:
            return sorted(set(hosts))
        return sorted(set([x for y in hosts for x in y]))

    def _install_agent_dependencies(self, hosts, collocate):
        """
        Install Agent python dependencies into an isolated venv on local and remote hosts.
        """
        ips = self._flatten_hosts(hosts, collocate)
        venv_python = CommandMaker.agent_venv_python()
        Print.info(f'Installing Agent Python dependencies into venv: {CommandMaker.AGENT_VENV_PATH}')

        local_requirements = os.path.abspath(join('..', 'Agent', 'requirements.txt'))
        if not os.path.isfile(local_requirements):
            raise BenchError(
                f'Cannot find Agent requirements file: {local_requirements}',
                FileNotFoundError(local_requirements)
            )

        try:
            CommandMaker.ensure_agent_venv(local_requirements)
            Print.info(f'Local Agent dependency install succeeded: {venv_python}')
        except Exception as e:
            Print.error(BenchError('Local Agent dependency install failed', e))
            raise

        remote_cmd = CommandMaker.remote_agent_venv_setup_cmds(self.settings.repo_name)
        g = Group(*ips, user=self.settings.username, connect_kwargs=self.connect)
        try:
            g.run(' && '.join(remote_cmd), hide=True)
            Print.info(
                f'Remote Agent dependency install succeeded on {len(ips)} host(s): {venv_python}'
            )
        except Exception as e:
            Print.error(BenchError('Remote Agent dependency install failed', e))
            raise

    def _update(self, hosts, collocate):
        ips = self._flatten_hosts(hosts, collocate)

        Print.info(
            f'Updating {len(ips)} machines (branch "{self.settings.branch}")...'
        )

        branch = self.settings.branch
        repo = self.settings.repo_name
        cmd = [
            # Fetch into origin/<branch>, not into the checked-out local branch
            # (git refuses `fetch origin branch:branch` on a non-bare repo).
            f'(cd {repo} && git fetch origin {branch})',
            f'(cd {repo} && git checkout -B {branch} origin/{branch})',
            f'(cd {repo} && git reset --hard origin/{branch})',
            f'source {self.home}/.cargo/env',
            f'(cd {repo} && {CommandMaker.compile()})',
            CommandMaker.alias_binaries(
                f'./{repo}/target/release/'
            )
        ]

        g = Group(*ips, user=self.settings.username, connect_kwargs=self.connect)
        g.run(' && '.join(cmd), hide=True)

    def _config(self, hosts, node_parameters, bench_parameters):
        Print.info('Generating configuration files...')

        # Cleanup all local configuration files.
        cmd = CommandMaker.cleanup()
        subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

        # Recompile the latest code.
        cmd = CommandMaker.compile().split()
        cmd[0] = self._resolve_cargo_binary()
        env = self._cargo_env_for_binary(cmd[0])
        subprocess.run(cmd, check=True, cwd=PathMaker.node_crate_path(), env=env)

        # Create alias for the client and nodes binary.
        cmd = CommandMaker.alias_binaries(PathMaker.binary_path())
        subprocess.run([cmd], shell=True)

        # Generate configuration files.
        keys = []
        key_files = [PathMaker.key_file(i) for i in range(len(hosts))]
        for filename in key_files:
            cmd = CommandMaker.generate_key(filename).split()
            subprocess.run(cmd, check=True)
            keys += [Key.from_file(filename)]

        names = [x.name for x in keys]

        if bench_parameters.collocate:
            workers = bench_parameters.workers
            addresses = OrderedDict(
                (x, [y] * (workers + 1)) for x, y in zip(names, hosts)
            )
        else:
            addresses = OrderedDict(
                (x, y) for x, y in zip(names, hosts)
            )
        committee = Committee(addresses, self.settings.base_port)
        committee.print(PathMaker.committee_file())

        node_parameters.print(PathMaker.parameters_file())

        # Cleanup all nodes and upload configuration files.
        names = names[:len(names)-bench_parameters.faults]
        progress = progress_bar(names, prefix='Uploading config files:')
        for i, name in enumerate(progress):
            for ip in committee.ips(name):
                c = Connection(ip, user=self.settings.username, connect_kwargs=self.connect)
                c.run(f'{CommandMaker.cleanup()} || true', hide=True)
                c.put(PathMaker.committee_file(), '.')
                c.put(PathMaker.key_file(i), '.')
                c.put(PathMaker.parameters_file(), '.')
                # c.put(PathMaker.local_parameters_file(i), '.')

        return committee

    def _run_single(self, rate, committee, bench_parameters, node_parameters, debug=False, node_regions=None):
        faults = bench_parameters.faults
        node_regions = node_regions or []
        region_based_asynchrony = node_parameters.json.get('simulate_asynchrony', False)
        region_based_hotspot = getattr(bench_parameters, 'enable_hotspot', False)

        # Kill any potentially unfinished run and delete logs.
        hosts = committee.ips()
        self.kill(hosts=hosts, delete_logs=True)

        # Clean up previous metrics and latency matrix files before starting new run
        Print.info('Cleaning up previous metrics and latency matrices...')
        workers_addresses = committee.workers_addresses(faults)
        rate_share = ceil(rate / committee.workers())
        all_worker_nodes = [x for y in workers_addresses for _, x in y]
        
        hotspot_config = None
        if getattr(bench_parameters, 'enable_hotspot', False):
            hotspot_config = {
                'enable_hotspot': True,
                'hotspot_windows': getattr(bench_parameters, 'hotspot_windows', []),
                'hotspot_nodes': getattr(bench_parameters, 'hotspot_nodes', []),
                'hotspot_regions': getattr(bench_parameters, 'hotspot_regions', []),
                'hotspot_region_rates': getattr(bench_parameters, 'hotspot_region_rates', []),
            }
            hotspot_regions = hotspot_config.get('hotspot_regions', [])
            if hotspot_regions and node_regions and region_based_hotspot:
                Print.info('Hotspot selection (region-based):')
                Print.info(f'  Node-to-region mapping: {list(enumerate(node_regions))}')
                hotspot_node_ids_per_window = []
                hotspot_node_rates_per_window = []
                hotspot_region_rates = hotspot_config.get('hotspot_region_rates', [])
                for w, n_counts in enumerate(hotspot_config['hotspot_nodes']):
                    regions = hotspot_regions[w] if w < len(hotspot_regions) else []
                    if regions:
                        ids = []
                        rates = []
                        region_rates = hotspot_region_rates[w] if w < len(hotspot_region_rates) else []
                        if isinstance(n_counts, int):
                            n_counts = [n_counts] * len(regions)
                        for r_idx, r in enumerate(regions):
                            n_pick = n_counts[r_idx] if r_idx < len(n_counts) else 0
                            nodes_in_r = [i for i in range(len(node_regions))
                                          if node_regions[i].lower() == r.lower()]
                            picked = nodes_in_r[:n_pick]
                            ids.extend(picked)
                            # hotspot_region_rates[w][r] is a per-node list
                            # (aligned with egress_penalty[w][r]).
                            per_node_rates = (
                                region_rates[r_idx] if r_idx < len(region_rates) else []
                            )
                            if isinstance(per_node_rates, list):
                                if len(per_node_rates) != len(picked):
                                    raise BenchError(
                                        'Mismatch between hotspot nodes and rates',
                                        ValueError(
                                            f'Window {w}, region {r}: picked {len(picked)} nodes '
                                            f'but {len(per_node_rates)} rates provided'
                                        )
                                    )
                                rates.extend([float(x) for x in per_node_rates])
                            else:
                                rates.extend([float(per_node_rates)] * len(picked))
                            Print.info(
                                f'  Window {w+1}: region "{r}" has nodes {nodes_in_r}, '
                                f'pick first {n_pick} => {picked} with rates '
                                f'{per_node_rates if isinstance(per_node_rates, list) else [per_node_rates] * len(picked)}'
                            )
                        hotspot_node_ids_per_window.append(ids)
                        hotspot_node_rates_per_window.append(rates)
                        Print.info(f'  Window {w+1}: hotspot node ids = {ids}')
                        if rates:
                            Print.info(f'  Window {w+1}: hotspot node rates = {rates}')
                    else:
                        n_total = sum(n_counts) if isinstance(n_counts, list) else n_counts
                        hotspot_node_ids_per_window.append(list(range(n_total)))
                        region_rates = hotspot_region_rates[w] if w < len(hotspot_region_rates) else []
                        flat_rates = []
                        for entry in region_rates:
                            if isinstance(entry, list):
                                flat_rates.extend([float(x) for x in entry])
                            else:
                                flat_rates.append(float(entry))
                        if len(flat_rates) == n_total:
                            hotspot_node_rates_per_window.append(flat_rates)
                        else:
                            fallback_rate = flat_rates[0] if flat_rates else 0.0
                            hotspot_node_rates_per_window.append([fallback_rate] * n_total)
                        Print.info(f'  Window {w+1}: no regions specified, use node indices 0..{n_total-1}')
                hotspot_config['hotspot_node_ids_per_window'] = hotspot_node_ids_per_window
                hotspot_config['hotspot_node_rates_per_window'] = hotspot_node_rates_per_window
            else:
                if hotspot_regions and not region_based_hotspot:
                    Print.info(
                        'Hotspot selection (index-based): region-based disabled '
                        '(requires simulate_asynchrony=true and enable_hotspot=true)'
                    )
                elif not hotspot_regions:
                    Print.info('Hotspot selection (index-based): no hotspot_regions, using first N nodes by index')
                elif not node_regions:
                    Print.info('Hotspot selection: hotspot_regions specified but node_regions empty (non-collocate?), using first N nodes by index')
                hotspot_node_ids_per_window = []
                hotspot_node_rates_per_window = []
                hotspot_region_rates = hotspot_config.get('hotspot_region_rates', [])
                for w, n_counts in enumerate(hotspot_config['hotspot_nodes']):
                    n_total = sum(n_counts) if isinstance(n_counts, list) else n_counts
                    ids = list(range(n_total))
                    hotspot_node_ids_per_window.append(ids)
                    region_rates = hotspot_region_rates[w] if w < len(hotspot_region_rates) else []
                    flat_rates = []
                    for entry in region_rates:
                        if isinstance(entry, list):
                            flat_rates.extend([float(x) for x in entry])
                        else:
                            flat_rates.append(float(entry))
                    if len(flat_rates) == len(ids):
                        hotspot_node_rates_per_window.append(flat_rates)
                    else:
                        fallback_rate = flat_rates[0] if flat_rates else 0.0
                        hotspot_node_rates_per_window.append([fallback_rate] * len(ids))
                    Print.info(f'  Window {w+1}: hotspot nodes = first {n_total} (indices 0..{n_total-1})')
                hotspot_config['hotspot_node_ids_per_window'] = hotspot_node_ids_per_window
                hotspot_config['hotspot_node_rates_per_window'] = hotspot_node_rates_per_window

        # Asynchrony region-based selection is resolved in run() before config upload.
        
        # Run cleanup locally and on every remote host.
        cleanup_cmd = CommandMaker.clean_metrics(self.settings.repo_name, 0)
        subprocess.run([cleanup_cmd], shell=True, stderr=subprocess.DEVNULL)
        g_cleanup = Group(*sorted(set(hosts)), user=self.settings.username, connect_kwargs=self.connect)
        g_cleanup.run(cleanup_cmd, hide=True)

        # Run the primaries (except the faulty ones).
        Print.info('Booting primaries...')
        for i, address in enumerate(committee.primary_addresses(faults)):
            host = Committee.ip(address)
            cmd = CommandMaker.run_primary(
                PathMaker.key_file(i),
                PathMaker.committee_file(),
                PathMaker.db_path(i),
                PathMaker.parameters_file(),
                debug=debug,
                node_index=i
            )
            log_file = PathMaker.primary_log_file(i)
            self._background_run(host, cmd, log_file)

        # Wait until all primaries are listening before proceeding.
        primary_addresses = committee.primary_addresses(faults)
        Print.info('Waiting for all primaries to be ready...')
        self._wait_for_tcp_listeners(primary_addresses, timeout_sec=90, label='primaries')

        # Run the workers (except the faulty ones).
        Print.info('Booting workers...')
        for i, addresses in enumerate(workers_addresses):
            for (id, address) in addresses:
                host = Committee.ip(address)
                cmd = CommandMaker.run_worker(
                    PathMaker.key_file(i),
                    PathMaker.committee_file(),
                    PathMaker.db_path(i, id),
                    f'{self.home}/.parameters.json',
                    id,  # The worker's id.
                    debug=debug
                )
                log_file = PathMaker.worker_log_file(i, id)
                self._background_run(host, cmd, log_file)

        # Wait until all workers transaction endpoints are listening.
        worker_listener_addresses = [addr for group in workers_addresses for (_, addr) in group]
        Print.info('Waiting for all workers to be ready...')
        self._wait_for_tcp_listeners(worker_listener_addresses, timeout_sec=90, label='workers')

        # Start controller for RL training.
        Print.info('Starting RL controllers...')
        resume_from = getattr(bench_parameters, 'cmab_resume_from', None)
        rl_algo = getattr(bench_parameters, 'rl_algo', 'cmab')
        warmup_iterations = getattr(bench_parameters, 'rl_warmup_iterations', 5)
        max_training_iterations = getattr(
            bench_parameters, 'rl_max_training_iterations', 200
        )
        Print.info(f'RL algo: {rl_algo}')
        Print.info(f'RL warmup iterations: {warmup_iterations}')
        Print.info(
            'RL max training iterations: '
            f'{max_training_iterations if max_training_iterations is not None else "continuous"}'
        )
        if resume_from:
            Print.info(f'RL resume-from: {resume_from}')
        for i, address in enumerate(primary_addresses):
            host = Committee.ip(address)
            cmd = CommandMaker.run_controller(
                node_index=i,
                repo_name=self.settings.repo_name,
                log_dir=PathMaker.logs_path(),
                parameters_file=f'{self.home}/.parameters.json',
                python_bin=CommandMaker.agent_venv_python(),
                resume_from=resume_from,
                rl_algo=rl_algo,
                cmab_seed=(
                    getattr(bench_parameters, 'cmab_seed', 0)
                    if rl_algo == 'cmab' else None
                ),
                enable_cmab_pairwise_residual_rf=(
                    getattr(
                        bench_parameters,
                        'enable_cmab_pairwise_residual_rf',
                        False,
                    )
                    if rl_algo == 'cmab' else False
                ),
                warmup_iterations=warmup_iterations,
                max_training_iterations=max_training_iterations,
                kernel_ucb_alpha=getattr(
                    bench_parameters, 'kernel_ucb_alpha', 1.0
                ),
                kernel_ucb_regularization=getattr(
                    bench_parameters, 'kernel_ucb_regularization', 0.1
                ),
                kernel_ucb_length_scale=getattr(
                    bench_parameters, 'kernel_ucb_length_scale', 1.0
                ),
                kernel_ucb_timeout_min=getattr(
                    bench_parameters, 'kernel_ucb_timeout_min', 1.0
                ),
                kernel_ucb_timeout_max=getattr(
                    bench_parameters, 'kernel_ucb_timeout_max', 300.0
                ),
                kernel_ucb_optimizer_restarts=getattr(
                    bench_parameters, 'kernel_ucb_optimizer_restarts', 5
                ),
                kernel_ucb_replay_window=getattr(
                    bench_parameters, 'kernel_ucb_replay_window', 200
                ),
            )
            log_file = join(PathMaker.logs_path(), f'controller-{i}.log')
            self._background_run(host, cmd, log_file)
        sleep(2)

        # Now that primaries are running and sockets are created, start metrics collectors.
        Print.info('Starting metrics collectors...')
        epoch_slots = node_parameters.json.get('epoch_slots', 20)
        window_size = node_parameters.json.get('window_size', 5)
        for i, address in enumerate(primary_addresses):
            host = Committee.ip(address)
            cmd = CommandMaker.run_metrics_collector(
                epoch_slots=epoch_slots,
                window_size=window_size,
                node_index=i,
                repo_name=self.settings.repo_name,
                log_dir=PathMaker.logs_path(),
                parameters_file=PathMaker.parameters_file(),
                python_bin=CommandMaker.agent_venv_python(),
            )
            log_file = join(PathMaker.logs_path(), f'metrics_collector-{i}.log')
            self._background_run(host, cmd, log_file)
        sleep(2)

        # Environment change detection runs independently of RL training.
        enable_reward_change_monitor = getattr(
            bench_parameters, 'enable_reward_change_monitor', True
        )
        Print.info(
            f'Reward change monitors enabled: {enable_reward_change_monitor}'
        )
        reward_change_window_size = getattr(
            bench_parameters, 'reward_change_window_size', 8
        )
        reward_change_lag = getattr(bench_parameters, 'reward_change_lag', 3)
        reward_change_threshold = getattr(
            bench_parameters, 'reward_change_threshold', 0.30
        )
        reward_change_confirmations = getattr(
            bench_parameters, 'reward_change_confirmations', 3
        )
        if enable_reward_change_monitor:
            Print.info('Starting reward change monitors...')
            Print.info(
                'Reward change detector: '
                f'window={reward_change_window_size}, lag={reward_change_lag}, '
                f'threshold={reward_change_threshold}, '
                f'confirmations={reward_change_confirmations}'
            )
            for i, address in enumerate(primary_addresses):
                host = Committee.ip(address)
                cmd = CommandMaker.run_reward_change_monitor(
                    node_index=i,
                    repo_name=self.settings.repo_name,
                    metrics_dir=f'{self.home}/metrics-{i}',
                    python_bin=CommandMaker.agent_venv_python(),
                    window_size=reward_change_window_size,
                    lag=reward_change_lag,
                    threshold=reward_change_threshold,
                    confirmations=reward_change_confirmations,
                )
                log_file = join(
                    PathMaker.logs_path(), f'reward_change_monitor-{i}.log'
                )
                self._background_run(host, cmd, log_file)
        else:
            Print.info('Reward change monitors disabled; skipping startup.')

        # Fix socket permissions to allow metrics_collector to connect.
        Print.info('Fixing socket permissions for metrics collection...')
        for i, address in enumerate(primary_addresses):
            host = Committee.ip(address)
            socket_path = f'/tmp/autopilot_core_{i}.sock'
            cmd = f'chmod 666 {socket_path} 2>/dev/null || true'
            try:
                c = Connection(host, user=self.settings.username, connect_kwargs=self.connect)
                c.run(cmd, hide=True)
            except Exception as e:
                Print.warn(f'Failed to set socket permissions for node {i}: {e}')

        # Start clients only after all primaries/workers are ready.
        Print.info('Booting clients...')
        for i, addresses in enumerate(workers_addresses):
            for (id, address) in addresses:
                host = Committee.ip(address)
                cmd = CommandMaker.run_client(
                    address,
                    bench_parameters.tx_size,
                    rate_share,
                    all_worker_nodes,
                    node_id=i,
                    hotspot_config=hotspot_config,
                )
                log_file = PathMaker.client_log_file(i, id)
                self._background_run(host, cmd, log_file)
        
        # Wait for all transactions to be processed.
        duration = bench_parameters.duration
        for i in progress_bar(range(20), prefix=f'Running benchmark ({duration} sec):'):
            sleep(ceil(duration / 20))
        self.kill(hosts=hosts, delete_logs=False)

    def _simulate_partition(self, bench_parameters, committee, faults):
        partition_ips = []
        for i, address in enumerate(committee.primary_addresses(faults)):
            if i < bench_parameters.partition_nodes:
                print(i, address)
                cmd = []
                #cmd = ['sudo tc qdisc del dev ens4 root']
                cmd.append('sudo tc qdisc add dev ens4 root handle 1: htb')
                cmd.append('sudo tc class add dev ens4 parent 1: classid 1:1 htb rate 10gibps')
                idx = 2
                for j, addr in enumerate(committee.primary_addresses(faults)):
                    if i == j:
                        continue
                    cmd.append('sudo tc class add dev ens4 parent 1:1 classid 1:' + str(idx) + ' htb rate 10gibps')
                    cmd.append('sudo tc qdisc add dev ens4 handle ' + str(idx) + ': parent 1:' 
                            + str(idx) + ' netem delay 5000ms')
                    cmd.append('sudo tc filter add dev ens4 pref ' + str(idx) + ' protocol ip u32 match ip dst ' + 
                            Committee.ip(addr) + ' flowid 1:' + str(idx))
                    idx = idx + 1
                ip = [Committee.ip(address)]
                g = Group(*ip, user=self.settings.username, connect_kwargs=self.connect)
                g.run(' && '.join(cmd), hide=True) 
        

         
        #hosts = committee.ips()
        #cmd = ['sudo iptables -A OUTPUT -d ' + ip + ' -j DROP' for ip in partition_ips]
        #cmd = ['sudo tc qdisc add dev ens4 root netem delay 5000ms']
        
        #g = Group(*partition_ips, user='neilgiridharan', connect_kwargs=self.connect)
        #g.run(' && '.join(cmd), hide=True) 
        
        #for i, address in enumerate(committee.primary_addresses(faults)):
        
        #host = Committee.ip(address)
        #for partition_ip in partition_ips:
        #cmd = 'sudo iptables -A OUTPUT -d ' + partition_ip + '-j DROP'
        
        ##log_file = PathMaker.primary_log_file(i)
        #self._background_run(host, cmd, log_file)
    
    def _delete_partition(self, bench_parameters, committee, faults):
        partition_ips = []
        for i, address in enumerate(committee.primary_addresses(faults)):
            if i < bench_parameters.partition_nodes:
                partition_ips = [Committee.ip(address)]
                cmd = ['sudo tc qdisc del dev ens4 root']
                g = Group(*partition_ips, user=self.settings.username, connect_kwargs=self.connect)
                g.run(' && '.join(cmd), hide=True) 

    def _logs(self, committee, faults):
        # Delete local logs (if any).
        cmd = CommandMaker.clean_logs()
        subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

        # Download log files.
        workers_addresses = committee.workers_addresses(faults)
        progress = progress_bar(workers_addresses, prefix='Downloading workers logs:')
        for i, addresses in enumerate(progress):
            for id, address in addresses:
                host = Committee.ip(address)
                c = Connection(host, user=self.settings.username, connect_kwargs=self.connect)
                c.get(
                    PathMaker.client_log_file(i, id), 
                    local=PathMaker.client_log_file(i, id)
                )
                c.get(
                    PathMaker.worker_log_file(i, id), 
                    local=PathMaker.worker_log_file(i, id)
                )

        primary_addresses = committee.primary_addresses(faults)
        progress = progress_bar(primary_addresses, prefix='Downloading primaries logs:')
        for i, address in enumerate(progress):
            host = Committee.ip(address)
            c = Connection(host, user=self.settings.username, connect_kwargs=self.connect)
            c.get(
                PathMaker.primary_log_file(i), 
                local=PathMaker.primary_log_file(i)
            )

        # Parse logs and return the parser.
        Print.info('Parsing logs and computing performance...')
        return LogParser.process(PathMaker.logs_path(), faults=faults)

    def run(self, bench_parameters_dict, node_parameters_dict, debug=False):
        assert isinstance(debug, bool)
        Print.heading('Starting remote benchmark')
        try:
            bench_parameters = BenchParameters(bench_parameters_dict)
            node_parameters = NodeParameters(node_parameters_dict)
        except ConfigError as e:
            raise BenchError('Invalid nodes or bench parameters', e)

        # Select which hosts to use.
        selected_hosts, node_regions = self._select_hosts(bench_parameters)
        if not selected_hosts:
            Print.warn('There are not enough instances available')
            return

        # Resolve asynchrony regions -> explicit node ids BEFORE writing parameters file.
        # This ensures primaries receive asynchrony_node_ids_per_window via .parameters.json.
        asynchrony_regions = node_parameters.json.get('asynchrony_regions', [])
        asynchrony_nodes = node_parameters.json.get('asynchrony_nodes', [])
        raw_egress_penalty = node_parameters.json.get('egress_penalty', 0)

        region_based_asynchrony = node_parameters.json.get('simulate_asynchrony', False)
        region_based_hotspot = getattr(bench_parameters, 'enable_hotspot', False)

        if asynchrony_regions and node_regions and region_based_asynchrony:
            Print.info('Asynchrony selection (region-based, pre-config):')
            Print.info(f'  Node-to-region mapping: {list(enumerate(node_regions))}')
            node_ids_per_window = []
            for w, regions in enumerate(asynchrony_regions):
                n_per_region = asynchrony_nodes[w] if w < len(asynchrony_nodes) else 0
                if isinstance(regions, str):
                    regions = [regions]
                regions = regions or []
                ids = []
                for r in regions:
                    nodes_in_r = [i for i in range(len(node_regions))
                                  if node_regions[i].lower() == str(r).lower()]
                    picked = nodes_in_r[:n_per_region] if n_per_region > 0 else []
                    ids.extend(picked)
                    Print.info(f'  Window {w+1}: region \"{r}\" has nodes {nodes_in_r}, pick first {n_per_region} => {picked}')
                node_ids_per_window.append(ids)
                Print.info(f'  Window {w+1}: asynchrony node ids = {ids}')
            node_parameters.json['asynchrony_node_ids_per_window'] = node_ids_per_window
        elif asynchrony_regions and not region_based_asynchrony:
            Print.info(
                'Asynchrony selection: region-based disabled '
                '(requires simulate_asynchrony=true and enable_hotspot=true); '
                'will ignore asynchrony_regions and use affected_nodes.'
            )
        elif asynchrony_regions and not node_regions:
            Print.info('Asynchrony selection: asynchrony_regions specified but node_regions empty, will ignore regions and use affected_nodes.')

        # Resolve per-node egress penalty from asynchrony_regions using per-node mapping.
        # New required format:
        #   asynchrony_regions = [[region_a, region_b], ...]
        #   egress_penalty   = [[[...], [...]], ...]
        #                      ↑ per-region list of per-node penalties
        egress_penalty_per_node = [0] * len(node_regions)

        for w, regions in enumerate(asynchrony_regions):
            region_list = [regions] if isinstance(regions, str) else (regions or [])
            n_per_region = asynchrony_nodes[w] if w < len(asynchrony_nodes) else 0

            region_penalties = raw_egress_penalty[w]

            if not isinstance(region_penalties, list):
                raise BenchError(
                    'Invalid egress_penalty configuration',
                    ValueError(f'egress_penalty[{w}] must be a list')
                )

            if len(region_penalties) != len(region_list):
                raise BenchError(
                    'Invalid egress_penalty configuration',
                    ValueError(
                        f'egress_penalty[{w}] length ({len(region_penalties)}) '
                        f'must equal asynchrony_regions[{w}] length ({len(region_list)})'
                    )
                )

            node_penalties = [0] * len(node_regions)

            for r_idx, region in enumerate(region_list):
                nodes_in_r = [
                    i for i in range(len(node_regions))
                    if str(node_regions[i]).lower() == str(region).lower()
                ]

                n_per_region = asynchrony_nodes[w] if w < len(asynchrony_nodes) else 0
                picked = nodes_in_r[:n_per_region] if n_per_region > 0 else []

                penalties_for_region = region_penalties[r_idx]

                if not isinstance(penalties_for_region, list):
                    raise BenchError(
                        'Invalid egress_penalty format',
                        ValueError(
                            f'egress_penalty[{w}][{r_idx}] must be a list (per-node)'
                        )
                    )

                if len(penalties_for_region) != len(picked):
                    raise BenchError(
                        'Mismatch between nodes and penalties',
                        ValueError(
                            f'Window {w}, region {region}: picked {len(picked)} nodes but '
                            f'{len(penalties_for_region)} penalties provided'
                        )
                    )

                for nid, delay in zip(picked, penalties_for_region):
                    try:
                        egress_penalty_per_node[nid] = int(delay)
                    except (TypeError, ValueError):
                        raise BenchError(
                            'Invalid egress_penalty value',
                            ValueError(
                                f'egress_penalty[{w}][{r_idx}] contains non-integer value {delay}'
                            )
                        )

                Print.info(
                    f'  Window {w+1}: region "{region}" picked nodes {picked} '
                    f'with delays {penalties_for_region}'
                )


        node_parameters.json['egress_penalty_per_node'] = egress_penalty_per_node

        # 兼容 Rust
        node_parameters.json['egress_penalty'] = 0

        Print.info(f'Egress penalty per node (resolved): {egress_penalty_per_node}')

        # Update nodes.
        print(selected_hosts)
        try:
            self._update(selected_hosts, bench_parameters.collocate)
            self._install_agent_dependencies(selected_hosts, bench_parameters.collocate)
        except (GroupException, ExecutionError) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to update nodes', e)
        except subprocess.SubprocessError as e:
            raise BenchError('Failed to install Agent dependencies', e)

        # Upload all configuration files.
        try:
            committee = self._config(
                selected_hosts, node_parameters, bench_parameters
            )
        except (subprocess.SubprocessError, GroupException) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to configure nodes', e)

        # Run benchmarks.
        for n in bench_parameters.nodes:
            committee_copy = deepcopy(committee)
            committee_copy.remove_nodes(committee.size() - n)
            run_node_regions = node_regions[:n] if node_regions else []

            for r in bench_parameters.rate:
                Print.heading(f'\nRunning {n} nodes (input rate: {r:,} tx/s)')

                # Run the benchmark.
                for i in range(bench_parameters.runs):
                    Print.heading(f'Run {i+1}/{bench_parameters.runs}')
                    try:
                        self._run_single(
                            r, committee_copy, bench_parameters, node_parameters, debug,
                            node_regions=run_node_regions
                        )

                        faults = bench_parameters.faults
                        logger = self._logs(committee_copy, faults)
                        result_file = PathMaker.result_file(
                            faults,
                            n, 
                            bench_parameters.workers,
                            bench_parameters.collocate,
                            r, 
                            bench_parameters.tx_size, 
                        )
                        result_config = {}
                        result_config.update(node_parameters.json)
                        result_config.update(node_parameters_dict)
                        result_config.update(bench_parameters_dict)
                        with open(result_file, 'a') as f:
                            f.write(logger.result(extra_config=result_config))
                    except (subprocess.SubprocessError, GroupException, ParseError) as e:
                        self.kill(hosts=selected_hosts)
                        if isinstance(e, GroupException):
                            e = FabricError(e)
                        Print.error(BenchError('Benchmark failed', e))
                        continue
