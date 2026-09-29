# Copyright(C) Facebook, Inc. and its affiliates.
import sys
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec_compat(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec_compat

from fabric import task, Connection
import time
import numpy as np

from benchmark.local import LocalBench
from benchmark.logs import ParseError, LogParser
from benchmark.utils import Print, BenchError
from benchmark.plot import Ploter, PlotError
from benchmark.gcp_instance import InstanceManager
from benchmark.remote import Bench
from fabric.transfer import Transfer
from paramiko import RSAKey, SSHException
from invoke.exceptions import UnexpectedExit
import os
from invoke import Responder

@task
def local(ctx, debug=False, enable_accelerator=False, accelerator_period=100):
    ''' Run benchmarks on localhost '''
    bench_params = {
        'faults': 0, 
        'nodes': 4,
        'workers': 1,
        'rate': 50000,
        'tx_size': 512,
        'duration': 60,

        # CMAB: set a checkpoint path to resume RL, or None to train from scratch.
        'cmab_resume_from': None,
        # RL algorithm: "cmab", "xgboost", "gp_bo", "kernel_ucb", or "dqn"
        'rl_algo': 'cmab',
        # Optional standard CMAB RF Cut x FPT features (12 extra columns).
        'enable_cmab_cut_fpt_cross_feature': False,
        # Online DQN; set rl_algo='dqn'. Only node0 trains, all primaries receive actions.
        'dqn_action_port': 19100,
        'dqn_learning_rate': 1e-3,
        'dqn_gamma': 0.90,
        'dqn_batch_size': 32,
        'dqn_learning_starts': 32,
        'dqn_replay_capacity': 2000,
        'dqn_target_update_interval': 20,
        'dqn_epsilon_start': 1.0,
        'dqn_epsilon_end': 0.05,
        'dqn_epsilon_decay_steps': 200,
        'dqn_seed': 0,
        # Set cmab_resume_from to a .pt checkpoint; use finetune for offline -> online.
        'dqn_checkpoint_load_mode': 'resume',
        'rl_max_training_iterations': 2500,
        # Optional node0 CMAB transition export for offline DQN training.
        'enable_cmab_transition_export': False,
        'cmab_transition_export_dir': '/tmp/autopilot_offline_data',
        'cmab_environment_label': 'gcp',

        # CMAB-RF only: "numeric" preserves the legacy raw parameter vector;
        # "one_hot" assigns one indicator feature to each complete arm.
        'cmab_action_encoding': 'numeric',
        # Pair this seed with the matching one_hot run; change only between reps.
        'cmab_seed': 0,
        'rl_warmup_iterations': 5,
        # rf_ts: global RF. factorized / combined: dedicated reward models.
        'cmab_policy': 'rf_ts',
        'enable_accelerator': bool(enable_accelerator),
        'accelerator_period': int(accelerator_period),

        # Unused
        'simulate_partition': False,
        'partition_start': 5,
        'partition_duration': 5,
        'partition_nodes': 1,
        
        'enable_hotspot': False,
        'hotspot_windows':[[0, 1500]],
        'hotspot_nodes': [1],
    }
    node_params = {
        'timeout_delay': 4_000,  # ms
        'header_size': 32,  # bytes
        'max_header_delay': 400,  # ms
        'gc_depth': 40,  # rounds
        'sync_retry_delay': 1_000,  # ms
        'sync_retry_nodes': 4,  # number of nodes
        'batch_size': 100_000,  # bytes
        'max_batch_delay': 400,  # ms
        'use_optimistic_tips': False,
        'use_parallel_proposals': True,
        'k': 1,
        'epoch_slots': 65,
        'window_size': 10,
        'applied_begin': 30,
        'use_fast_path': True,
        'fast_path_timeout': 0,
        'use_ride_share': False,
        'car_timeout': 2000,
        'cut_condition_type': 1,

        'simulate_asynchrony': False,
        'asynchrony_type': [6],

        'asynchrony_start': [0], #s
        'asynchrony_duration': [1500], #s
        'affected_nodes': [2],
        'egress_penalty': 200, #ms

        'use_fast_sync': True,
        'use_exponential_timeouts': True,

        # Ablation: aggregation strategy for global state.
        # "normal"  -> original behaviour (max for growth_rates, median for reward/fpr).
        # "mean"    -> arithmetic mean for all three metrics.
        'aggregation_strategy': 'mean',

        # Ablation: data-pollution simulation.
        # List the 0-based node indices that act as polluters.
        'data_pollution_node_ids': [0],
        # Probability [0.0, 1.0] that a polluter reports fake metrics.
        'data_pollution_prob': 1.0,
        # "random_scale" keeps the old random up/down scaling behaviour.
        # "mean_equalize" makes polluted metrics converge to a narrow target band.
        'data_pollution_strategy': 'mean_equalize',
    }
    try:
        ret = LocalBench(bench_params, node_params).run(debug)
        print(ret.result())
    except BenchError as e:
        Print.error(e)


@task
def create(ctx, nodes=3):
    ''' Create a testbed'''
    try:
        InstanceManager.make().create_instances(nodes)
    except BenchError as e:
        Print.error(e)


@task
def destroy(ctx):
    ''' Destroy the testbed '''
    try:
        InstanceManager.make().terminate_instances()
    except BenchError as e:
        Print.error(e)


@task
def start(ctx, max=4):
    ''' Start at most `max` machines per data center '''
    try:
        InstanceManager.make().start_instances(max)
    except BenchError as e:
        Print.error(e)


@task
def stop(ctx):
    ''' Stop all machines '''
    try:
        InstanceManager.make().stop_instances()
    except BenchError as e:
        Print.error(e)


@task
def info(ctx):
    ''' Display connect information about all the available machines '''
    try:
        InstanceManager.make().print_info()
    except BenchError as e:
        Print.error(e)


@task
def install(ctx):
    ''' Install the codebase on all machines '''
    try:
        Bench(ctx).install()
    except BenchError as e:
        Print.error(e)

from fabric import task


@task
def remote(
    ctx,
    debug=False,
    cmab_seed=0,
    cmab_action_encoding='numeric',
    duration=3600,
    rl_algo='cmab',
    cmab_policy='rf_ts',
    cmab_start_pos=0,
    start_controller=True,
):
    ''' Run benchmarks on GCP.

    rl_algo: cmab, xgboost, gp_bo, kernel_ucb, dqn.
    cmab_policy applies when rl_algo is cmab:
      rf_ts, random, default, round_robin, factorized, combined.
    Omit the controller with --no-start-controller.
    '''
    encoding = str(cmab_action_encoding).lower()
    if encoding not in ('numeric', 'one_hot'):
        raise ValueError('cmab_action_encoding must be "numeric" or "one_hot"')
    algo = str(rl_algo).lower()
    if algo not in ('cmab', 'xgboost', 'gp_bo', 'kernel_ucb', 'dqn'):
        raise ValueError(
            'rl_algo must be one of cmab, xgboost, gp_bo, kernel_ucb, dqn'
        )
    policy = str(cmab_policy).lower()
    if policy not in (
        'rf_ts', 'random', 'default', 'round_robin', 'factorized', 'combined'
    ):
        raise ValueError(
            'cmab_policy must be one of rf_ts, random, default, '
            'round_robin, factorized, combined'
        )
    if not isinstance(start_controller, bool):
        raise ValueError('start_controller must be a boolean')
    zone = InstanceManager.make().settings.regions[0]
    try:
        start_pos = int(cmab_start_pos)
    except (TypeError, ValueError) as exc:
        raise ValueError('cmab_start_pos must be an integer >= 0') from exc
    if start_pos < 0:
        raise ValueError('cmab_start_pos must be an integer >= 0')
    resume_from = None
    applied_begin = 42
    bench_params = {
        'faults': 0,
        'nodes': [4],
        'workers': 1,
        'collocate': True,
        'rate': [40_000],
        'tx_size': 512,
        # Long enough for 200 valid (apply-ok) epochs; the sweep runner
        # kills earlier once that count is reached.
        'duration': int(duration),
        'runs': 1,

        # CMAB: set a checkpoint path to resume RL, or None to train from scratch.
        'cmab_resume_from': resume_from,
        # rl_algo: cmab | xgboost | gp_bo | kernel_ucb | dqn
        # cmab_policy (only forwarded when rl_algo is cmab):
        #   rf_ts        global random forest, pick the highest predicted arm
        #   random       uniform arm each epoch; the forest is not used to choose
        #   default      one fixed arm (batch 500000, header 32, cut 3, timeout 200, k 1)
        #   round_robin  cycle the shuffled catalog; no model update; see cmab_start_pos
        #   factorized   per-parameter residual forests, then pick the best arm
        #   combined     global forest plus a residual forest on cut and timeout
        'rl_algo': algo,
        # Optional standard CMAB RF Cut x FPT features (12 extra columns).
        'enable_cmab_cut_fpt_cross_feature': False,
        # Online DQN; set rl_algo='dqn'. Only node0 trains, all primaries receive actions.
        'dqn_action_port': 19100,
        'dqn_learning_rate': 1e-3,
        'dqn_gamma': 0.90,
        'dqn_batch_size': 32,
        'dqn_learning_starts': 32,
        'dqn_replay_capacity': 2000,
        'dqn_target_update_interval': 20,
        'dqn_epsilon_start': 1.0,
        'dqn_epsilon_end': 0.05,
        'dqn_epsilon_decay_steps': 200,
        'dqn_seed': 0,
        # Set cmab_resume_from to a .pt checkpoint; use finetune for offline -> online.
        'dqn_checkpoint_load_mode': 'resume',
        'rl_max_training_iterations': 2500,
        # Optional node0 CMAB transition export for offline DQN training.
        'enable_cmab_transition_export': False,
        'cmab_transition_export_dir': '/tmp/autopilot_offline_data',
        'cmab_environment_label': 'gcp',

        # CMAB-RF action representation. Use "numeric" for the existing
        # baseline and "one_hot" for the encoding comparison experiment.
        'cmab_action_encoding': encoding,
        # Pair numeric/one_hot with the same seed; change only between reps.
        'cmab_seed': int(cmab_seed),
        'rl_warmup_iterations': 5,
        'cmab_policy': policy,
        'cmab_start_pos': start_pos,
        'enable_accelerator': False,
        'accelerator_period': 20,

        # Unused
        'simulate_partition': False,
        'partition_start': 5,
        'partition_duration': 5,
        'partition_nodes': 2,
        
        'enable_hotspot': False,
        'hotspot_windows': [[0, 3000]],
        'hotspot_regions': [[zone]],
        'hotspot_nodes': [[2]],
        'hotspot_region_rates': [[[0.99, 0.99]]], 
    }
    node_params = {
        'timeout_delay': 5_000,  # ms
        'header_size': 32,  # bytes
        'max_header_delay': 5000,  # ms
        'gc_depth': 50,  # rounds
        'sync_retry_delay': 5000,  # ms
        'sync_retry_nodes': 3,  # number of nodes
        'batch_size': 100_000,  # bytes
        'max_batch_delay': 5000,  # ms
        'use_optimistic_tips': True,
        'use_parallel_proposals': True,
        'k': 4,
        'epoch_slots': 30,
        'window_size': 12,
        # Apply on first commit at/after this position (Rust uses >=).
        # Keep well below epoch_slots so k-parallel slot skips still land.
        'applied_begin': 28,
        'use_fast_path': True,
        'fast_path_timeout': 50,
        'use_ride_share': False,
        'car_timeout': 2000,
        'cut_condition_type': 2,

        'simulate_asynchrony': False,
        'asynchrony_type': [6],

        'asynchrony_start': [0],  # s
        'asynchrony_duration': [120],  # s
        'affected_nodes': [0],
        'asynchrony_nodes': [0],
        'asynchrony_regions': [[ ]],
        'egress_penalty': [[ ]],

        'use_fast_sync': True,
        'use_exponential_timeouts': False,

        'aggregation_strategy': 'normal',
        'data_pollution_node_ids': [],
        'data_pollution_prob': 1.0,
        'data_pollution_strategy': 'random_scale',
    }
    try:
        Bench(ctx).run(
            bench_params, node_params, debug, start_controller=start_controller
        )
    except BenchError as e:
        Print.error(e)


@task
def plot(ctx):
    ''' Plot performance using the logs generated by "fab remote" '''
    plot_params = {
        'faults': [0],
        'nodes': [4],
        'workers': [1, 4, 7, 10],
        'collocate': True,
        'tx_size': 512,
        'max_latency': [2_000, 2_500]
    }
    try:
        Ploter.plot(plot_params)
    except PlotError as e:
        Print.error(BenchError('Failed to plot performance', e))


@task
def kill(ctx):
    ''' Stop execution on all machines '''
    try:
        Bench(ctx).kill()
    except BenchError as e:
        Print.error(e)


@task
def logs(ctx):
    ''' Print a summary of the logs '''
    try:
        print(LogParser.process('./logs', faults='?').result())
    except ParseError as e:
        Print.error(BenchError('Failed to parse logs', e))

def _parse_committee_from_logs(log_dir='../../logs'):
    """Parse committee information from metrics log files

    Returns:
        Dict mapping node names to IP addresses
    """
    import glob
    import json
    import os

    node_mapping = {}  # name -> ip

    # First try to read from metrics log files (new format)
    metrics_files = glob.glob(f'{log_dir}/metrics-*.log')
    if metrics_files:
        # Use the first metrics file
        metrics_file = metrics_files[0]
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('event_type') == 'committee':
                            details = event.get('details', {})
                            pk = details.get('pk')
                            consensus_addr = details.get('consensus_addr')
                            if pk and consensus_addr:
                                ip = consensus_addr.split(':')[0]  # Extract IP from "ip:port"
                                node_mapping[pk] = ip
                    except json.JSONDecodeError:
                        continue
            if node_mapping:
                print(f"Parsed committee from metrics log: {len(node_mapping)} nodes")
                return node_mapping
        except Exception as e:
            print(f"Warning: Failed to parse committee from metrics log {metrics_file}: {e}")

    # Fallback: try to read from primary log files (old format)
    primary_logs = glob.glob(f'{log_dir}/primary-*.log')
    if not primary_logs:
        print(f"Warning: No primary log files found in {log_dir}")
        return node_mapping

    # Use the first available primary log
    log_file = primary_logs[0]

    try:
        import re
        with open(log_file, 'r') as f:
            for line in f:
                # Look for Authority lines in config section
                match = re.search(r"Authority ([A-Za-z0-9+/=]+)=: stake=(\d+), consensus=([^,]+)", line)
                if match:
                    pk, stake, consensus_addr = match.groups()
                    ip = consensus_addr.split(':')[0]  # Extract IP from "ip:port"
                    node_mapping[pk] = ip
    except Exception as e:
        print(f"Warning: Failed to parse committee from primary log {log_file}: {e}")

    return node_mapping


def _parse_current_node_from_logs(log_dir='../../logs'):
    """Parse current node information from primary log files

    Returns:
        Tuple of (node_name, node_ip) or (None, None) if not found
    """
    import glob
    import re
    import os

    # Find primary log files
    primary_logs = glob.glob(f'{log_dir}/primary-*.log')
    if not primary_logs:
        print(f"Warning: No primary log files found in {log_dir}")
        return None, None

    # Check each primary log file for current node info
    for log_file in primary_logs:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Try to extract node index from store_path in the log line
                    match = re.search(r'extracting node index from store_path: \.db-(\d+)', line)
                    if match:
                        node_index = int(match.group(1))
                        print(f"Found node index {node_index} from store_path in log")

                        # Now get the committee to find the corresponding public key
                        committee = _parse_committee_from_logs(log_dir)
                        if committee:
                            # Find the public key at this index position
                            # Since committee is a dict, we need to get the key at the specific index
                            sorted_keys = sorted(committee.keys())
                            if node_index < len(sorted_keys):
                                node_name = sorted_keys[node_index]
                                node_ip = committee[node_name]
                                print(f"Mapped node index {node_index} to public key {node_name}")
                                return node_name, node_ip

        except Exception as e:
            print(f"Warning: Failed to parse current node from log {log_file}: {e}")

    print(f"Warning: Could not find current node information in any primary log file")
    return None, None


def _get_nodes_from_fab_info():
    """Get node metadata (name, zone, ip) from the GCP InstanceManager."""
    manager = InstanceManager.make()
    ids_by_region, ips_by_region = manager._get(['STAGING', 'RUNNING'])

    nodes = []
    for region in sorted(ips_by_region.keys()):
        region_ips = ips_by_region.get(region, [])
        region_names = ids_by_region.get(region, [])
        for idx, ip in enumerate(region_ips):
            name = region_names[idx] if idx < len(region_names) else f"{region}-node-{idx}"
            nodes.append({"name": name, "region": region, "ip": ip})
    return nodes


def _detect_current_node_from_fab_info(nodes):
    """Best-effort detect current node by matching local IP to fab-info IPs."""
    import socket
    import subprocess

    local_ips = set()
    try:
        local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
        if output:
            local_ips.update(output.split())
    except Exception:
        pass

    local_ips = {ip for ip in local_ips if ip and not ip.startswith("127.")}
    if not local_ips:
        return None

    for node in nodes:
        if node["ip"] in local_ips:
            return node

    # fab may run on a control VM whose address is not in the instance list.
    preferred = sorted(
        local_ips,
        key=lambda ip: (0 if ip.startswith('10.') else 1, ip),
    )[0]
    return {
        "name": socket.gethostname().split('.')[0],
        "region": "local",
        "ip": preferred,
    }


def _ssh_connect_settings():
    """Load SSH key/username from settings.json."""
    settings = InstanceManager.make().settings
    password = getattr(settings, 'ssh_key_password', None) or os.environ.get('SSH_KEY_PASSWORD')
    if password:
        pkey = RSAKey.from_private_key_file(settings.key_path, password=password)
    else:
        pkey = RSAKey.from_private_key_file(settings.key_path)
    return settings.username, pkey


def _rtt_mean_median(samples):
    """Arithmetic mean and median of per-packet RTTs, in milliseconds."""
    if not samples:
        return np.nan, np.nan
    values = np.asarray(samples, dtype=float)
    return float(np.mean(values)), float(np.median(values))


def _format_rtt(mean_ms, median_ms):
    if not np.isfinite(mean_ms) or not np.isfinite(median_ms):
        return "Failed"
    return f"mean {mean_ms:.2f} ms, median {median_ms:.2f} ms"


def _ping_rtt_stats(src_node, dst_ip, username, connect_kwargs, repeat=5, quiet=False):
    """ICMP RTT mean and median (ms) from src_node to dst_ip.

    Parses each `time=` sample from `ping` and summarizes those samples.
    Same-host pairs return (0.0, 0.0). Failures return (nan, nan).
    """
    import re

    if src_node["ip"] == dst_ip:
        return 0.0, 0.0
    try:
        conn = Connection(
            host=src_node["ip"],
            user=username,
            connect_kwargs=connect_kwargs,
        )
        result = conn.run(
            f"ping -c {repeat} -W 2 {dst_ip}",
            hide=True,
            warn=True,
            timeout=max(30, repeat * 3),
        )
        conn.close()
        samples = [
            float(match)
            for match in re.findall(r"time[=<]([\d.]+)\s*ms", result.stdout or "")
        ]
        if not samples:
            if not quiet:
                print(f"[Error] ping {src_node['ip']} → {dst_ip}: no rtt samples")
            return np.nan, np.nan
        return _rtt_mean_median(samples)
    except Exception as e:
        if not quiet:
            print(f"[Error] ping {src_node['ip']} → {dst_ip}: {e}")
        return np.nan, np.nan


def collect_latency_matrices(repeat=5, quiet=False):
    """
    Full node-to-node ICMP RTT matrices in milliseconds.

    Returns (mean_matrix, median_matrix). Each has shape (n, n): diagonal is
    0.0, failed pings are nan. One ping run supplies both summaries.
    """
    username, pkey = _ssh_connect_settings()
    connect_kwargs = {"pkey": pkey}
    node_records = _get_nodes_from_fab_info()
    if not node_records:
        raise RuntimeError("Failed to read node info from fab/InstanceManager")

    n = len(node_records)
    mean_matrix = np.full((n, n), np.nan, dtype=float)
    median_matrix = np.full((n, n), np.nan, dtype=float)
    for i, src_node in enumerate(node_records):
        for j, dst_node in enumerate(node_records):
            if i == j:
                mean_matrix[i, j] = 0.0
                median_matrix[i, j] = 0.0
                continue
            mean_ms, median_ms = _ping_rtt_stats(
                src_node,
                dst_node["ip"],
                username,
                connect_kwargs,
                repeat=repeat,
                quiet=quiet,
            )
            mean_matrix[i, j] = mean_ms
            median_matrix[i, j] = median_ms
    return mean_matrix, median_matrix


def collect_latency_matrix(repeat=5, quiet=False):
    """Mean ICMP RTT matrix. Accelerator callers keep using this mean matrix."""
    mean_matrix, _median_matrix = collect_latency_matrices(repeat=repeat, quiet=quiet)
    return mean_matrix


def publish_accelerator_hint(hint: dict, dest_path: str) -> None:
    """Write the accelerator hint locally, then copy it to every other node."""
    import json
    import shlex
    from pathlib import Path

    payload = json.dumps(hint, separators=(",", ":"))
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(dest)

    username, pkey = _ssh_connect_settings()
    connect_kwargs = {"pkey": pkey}
    nodes = _get_nodes_from_fab_info()
    current = _detect_current_node_from_fab_info(nodes)
    current_ip = current["ip"] if current else None
    quoted = shlex.quote(payload)
    quoted_path = shlex.quote(str(dest_path))
    quoted_tmp = shlex.quote(str(dest_path) + ".tmp")
    cmd = f"printf '%s\\n' {quoted} > {quoted_tmp} && mv {quoted_tmp} {quoted_path}"
    for node in nodes:
        if node["ip"] == current_ip:
            continue
        try:
            conn = Connection(
                host=node["ip"],
                user=username,
                connect_kwargs=connect_kwargs,
            )
            conn.run(cmd, hide=True, warn=True, timeout=20)
            conn.close()
        except Exception as e:
            print(f"[Error] publish accelerator hint to {node.get('name', node['ip'])}: {e}")

@task
def latency(ctx, cross_region=False, source_node=None, full_matrix=True):
    """
    Measure ICMP ping latency between nodes.

    SSHes into the source node and runs `ping` from there to each destination.
    Each probe reports both the mean and the median of the per-packet RTTs.

    Args:
        cross_region: Whether to also measure cross-region latency (default: False)
        source_node: If specified, measure latency only from this node to all others.
                    If None, auto-detect current node.
        full_matrix: If True, measure full node-to-node latency matrix instead of single source.
    """
    import json
    import re
    import time
    import numpy as np
    from fabric import Connection
    from paramiko import SSHException
    from benchmark.utils import Print

    try:
        username, pkey = _ssh_connect_settings()
        ctx.connect_kwargs.pkey = pkey
        connect_kwargs = ctx.connect_kwargs
    except (IOError, SSHException) as e:
        Print.error(f"Failed to load SSH key: {e}")
        return

    # Read node metadata from fab info source.
    node_records = _get_nodes_from_fab_info()
    if not node_records:
        error_msg = "Failed to read node info from fab/InstanceManager"
        Print.error(BenchError(error_msg, Exception(error_msg)))
        return

    print(f"Total nodes from fab info: {len(node_records)}")

    source_record = None
    if source_node is None:
        source_record = _detect_current_node_from_fab_info(node_records)
        if source_record is None:
            error_msg = "Could not determine current node from fab info; please pass --source-node"
            Print.error(BenchError(error_msg, Exception(error_msg)))
            return
        source_node = source_record["name"]
        print(
            f"Auto-detected current node from fab info: {source_record['name']} "
            f"({source_record['region']}, {source_record['ip']})"
        )
    else:
        print(f"Using specified source node: {source_node}")
        for node in node_records:
            if node["name"] == source_node:
                source_record = node
                break

    # Single source mode (default behavior)
    if source_record is None:
        error_msg = f"Source node {source_node} not found in committee"
        Print.error(BenchError(error_msg, Exception(error_msg)))
        return

    current_node_ip = source_record["ip"]
    print(f"Source node: {source_record['name']} ({source_record['region']})")
    print(f"Source node IP: {current_node_ip}")

    # Get all target nodes (excluding self)
    target_nodes = [node for node in node_records if node["name"] != source_node]
    print(f"Will measure latency to {len(target_nodes)} other nodes")

    def ping_latency(src_node, dst_ip, repeat=5):
        return _ping_rtt_stats(src_node, dst_ip, username, connect_kwargs, repeat=repeat)

    if full_matrix:
        # Full matrix mode: measure all node pairs (fallback for compatibility)
        print("Warning: Full matrix mode requested - this will measure all node pairs and may be slow")
        print("For normal operation, consider using single-source mode (default)")

        region_to_nodes = {}
        for node in node_records:
            region_to_nodes.setdefault(node["region"], []).append(node)

        region_nodes = []
        all_nodes = list(node_records)
        for region, nodes in region_to_nodes.items():
            if len(nodes) < 2:
                Print.warn(f"[Skip] Region {region} has <2 nodes.")
            region_nodes.append((region, nodes))

        m = len(region_nodes)
        region_matrix = np.zeros((m, m))
        region_names = [r[0] for r in region_nodes]

        node_names = [node["name"] for node in all_nodes]

        print("=== Measuring Full Node-to-Node Latency Matrix (ICMP ping) ===")
        full_latency_matrix, full_latency_median_matrix = collect_latency_matrices()
        name_to_idx = {node["name"]: idx for idx, node in enumerate(all_nodes)}
        for i, src_node in enumerate(all_nodes):
            for j, dst_node in enumerate(all_nodes):
                if i == j:
                    continue
                print(
                    f"  {src_node['name']} → {dst_node['name']}: "
                    f"{_format_rtt(full_latency_matrix[i][j], full_latency_median_matrix[i][j])}"
                )

        # Region summaries and statistics for full matrix mode...
        print("\n=== Calculating Region Summaries ===")
        for i, (region, nodes) in enumerate(region_nodes):
            region_node_names = {node["name"] for node in nodes}
            region_indices = [idx for idx, node in enumerate(all_nodes) if node["name"] in region_node_names]
            region_means = []
            region_medians = []
            for j in region_indices:
                for k in region_indices:
                    if j == k:
                        continue
                    if not np.isnan(full_latency_matrix[j][k]):
                        region_means.append(full_latency_matrix[j][k])
                    if not np.isnan(full_latency_median_matrix[j][k]):
                        region_medians.append(full_latency_median_matrix[j][k])

            if region_means:
                avg_latency = np.mean(region_means)
                region_matrix[i][i] = avg_latency
                median_latency = np.median(region_medians) if region_medians else np.nan
                print(f"  [Region] {region}: {_format_rtt(avg_latency, median_latency)}")
            else:
                region_matrix[i][i] = np.nan

        if cross_region:
            print("\n=== Measuring Cross-region Latency ===")
            for i in range(m):
                region_i, nodes_i = region_nodes[i]
                for j in range(m):
                    if i == j:
                        continue
                    region_j, nodes_j = region_nodes[j]

                    node_i = nodes_i[0]
                    node_j = nodes_j[0]
                    src_idx = name_to_idx[node_i["name"]]
                    dst_idx = name_to_idx[node_j["name"]]
                    mean_ms = full_latency_matrix[src_idx][dst_idx]
                    median_ms = full_latency_median_matrix[src_idx][dst_idx]
                    region_matrix[i][j] = mean_ms
                    print(f"[Cross] {region_i} → {region_j}: {_format_rtt(mean_ms, median_ms)}")

        print(f"\n=== Full Node Latency Matrix, mean (ms) ===")
        print("Nodes:", node_names)
        print(full_latency_matrix)
        print(f"\n=== Full Node Latency Matrix, median (ms) ===")
        print(full_latency_median_matrix)

        def full_matrix_stats(matrix):
            off_diagonal = [matrix[i][j] for i in range(len(matrix))
                          for j in range(len(matrix)) if i != j and not np.isnan(matrix[i][j])]

            if not off_diagonal:
                return None

            return {
                "mean": float(np.mean(off_diagonal)),
                "median": float(np.median(off_diagonal)),
                "std": float(np.std(off_diagonal)),
                "min": float(np.min(off_diagonal)),
                "max": float(np.max(off_diagonal)),
            }

        def print_matrix_stats(label, matrix):
            stats = full_matrix_stats(matrix)
            print(f"\n=== Full Matrix Statistics ({label}) ===")
            if stats is None:
                print("No valid measurements")
                return
            print(f"Mean Latency: {stats['mean']:.2f} ms")
            print(f"Median Latency: {stats['median']:.2f} ms")
            print(f"Std Deviation: {stats['std']:.2f} ms")
            print(f"Min Latency: {stats['min']:.2f} ms")
            print(f"Max Latency: {stats['max']:.2f} ms")

        print_matrix_stats("per-link mean", full_latency_matrix)
        print_matrix_stats("per-link median", full_latency_median_matrix)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mean_path = f"latency_full_matrix_{timestamp}.npy"
        median_path = f"latency_full_matrix_median_{timestamp}.npy"
        np.save(mean_path, full_latency_matrix)
        np.save(median_path, full_latency_median_matrix)
        print(f"\nResults saved to:")
        print(f"  - {mean_path}")
        print(f"  - {median_path}")

    # Default behavior: measure latency vector from source node to all others.
    # Reuse the full matrix when this source was already probed.
    print("=== Measuring Latency from Current Node (ICMP ping) ===")
    source_idx = next(
        (i for i, node in enumerate(node_records) if node["name"] == source_record["name"]),
        None,
    )
    node_index = {node["name"]: i for i, node in enumerate(node_records)}
    latency_means = []
    latency_medians = []
    measured_count = 0

    for target in target_nodes:
        target_name = target["name"]
        target_ip = target["ip"]
        target_region = target["region"]
        if full_matrix and source_idx is not None and target_name in node_index:
            dst_idx = node_index[target_name]
            mean_ms = full_latency_matrix[source_idx][dst_idx]
            median_ms = full_latency_median_matrix[source_idx][dst_idx]
        else:
            mean_ms, median_ms = ping_latency(source_record, target_ip)
        latency_means.append(mean_ms)
        latency_medians.append(median_ms)

        print(
            f"  {source_node} ({source_record['region']}) → {target_name} ({target_region}): "
            f"{_format_rtt(mean_ms, median_ms)}"
        )
        if np.isfinite(mean_ms) and np.isfinite(median_ms):
            measured_count += 1

    print(f"\nSuccessfully measured latency to {measured_count}/{len(target_nodes)} nodes")

    def print_vector_stats(label, values):
        valid = [lat for lat in values if not np.isnan(lat)]
        print(f"\n=== Latency Statistics ({label}) ===")
        if not valid:
            print("No valid measurements")
            return
        print(f"Mean Latency: {np.mean(valid):.2f} ms")
        print(f"Median Latency: {np.median(valid):.2f} ms")
        print(f"Std Deviation: {np.std(valid):.2f} ms")
        print(f"Min Latency: {np.min(valid):.2f} ms")
        print(f"Max Latency: {np.max(valid):.2f} ms")
        print(f"Success Rate: {len(valid)}/{len(values)}")

    print_vector_stats("per-link mean", latency_means)
    print_vector_stats("per-link median", latency_medians)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    def _json_latency(values):
        return [float(x) if not np.isnan(x) else None for x in values]

    # Output the latency vector as JSON for easy parsing.
    # `latencies` stays the per-link mean so existing parsers keep working.
    import json
    output_data = {
        'source_node': source_node,
        'source_region': source_record['region'],
        'source_ip': current_node_ip,
        'target_nodes': [{'name': node['name'], 'region': node['region'], 'ip': node['ip']} for node in target_nodes],
        'latencies': _json_latency(latency_means),
        'latencies_mean': _json_latency(latency_means),
        'latencies_median': _json_latency(latency_medians),
        'timestamp': timestamp
    }

    # Print JSON output that can be captured by the calling process
    print("LATENCY_VECTOR_JSON_START")
    print(json.dumps(output_data))
    print("LATENCY_VECTOR_JSON_END")

    # Also save to file for debugging
    mean_vector_path = f"latency_vector_{timestamp}.npy"
    median_vector_path = f"latency_vector_median_{timestamp}.npy"
    np.save(mean_vector_path, np.array(latency_means))
    np.save(median_vector_path, np.array(latency_medians))
    print(
        f"\nResults also saved to: {mean_vector_path}, {median_vector_path}",
        file=sys.stderr,
    )

    return


@task
def bandwidth(ctx, source_node=None, duration=8, parallel=4):
    """
    Measure bandwidth between nodes using iperf3.

    Args:
        source_node: If specified, measure bandwidth from this node to all others.
                    If None, auto-detect current node.
        duration: Duration of each iperf3 test in seconds (default: 8)
        parallel: Number of parallel streams for iperf3 (default: 4)
    """
    import time
    import numpy as np
    import json
    import os
    from fabric import Connection
    from paramiko import RSAKey, SSHException
    from invoke.exceptions import UnexpectedExit
    from benchmark.utils import Print

    try:
        # GCP: ctx.connect_kwargs.pkey = RSAKey.from_private_key_file("/home/ccclr0302/.ssh/gcp_rsa")
        username, pkey = _ssh_connect_settings()
        ctx.connect_kwargs.pkey = pkey
        connect_kwargs = ctx.connect_kwargs
    except (IOError, SSHException) as e:
        Print.error(f"Failed to load SSH key: {e}")
        return

    # Determine source node
    if source_node is None:
        # Auto-detect current node from primary logs
        source_node, current_node_ip = _parse_current_node_from_logs()
        if source_node is None:
            error_msg = "Could not determine current node from primary logs and no source_node specified"
            Print.error(BenchError(error_msg, Exception(error_msg)))
            return
        print(f"Auto-detected current node: {source_node} at {current_node_ip} (from primary log)")
    else:
        print(f"Using specified source node: {source_node}")
        current_node_ip = None

    # Read committee info from logs
    node_mapping = _parse_committee_from_logs()
    if not node_mapping:
        error_msg = "Failed to parse committee info from logs"
        Print.error(BenchError(error_msg, Exception(error_msg)))
        return

    print(f"Total nodes in committee: {len(node_mapping)}")

    if source_node not in node_mapping:
        error_msg = f"Source node {source_node} not found in committee"
        Print.error(BenchError(error_msg, Exception(error_msg)))
        return

    # For specified source node, get IP from committee mapping
    if current_node_ip is None:
        current_node_ip = node_mapping[source_node]
    print(f"Source node IP: {current_node_ip}")

    # Get all target nodes (excluding self)
    target_nodes = [(name, ip) for name, ip in node_mapping.items() if name != source_node]
    print(f"Will measure bandwidth to {len(target_nodes)} other nodes")

    def measure_iperf3_bandwidth(src_conn, dst_ip, duration=8, parallel=4):
        """Measure bandwidth from source connection to destination IP using iperf3"""
        dst_conn = None
        try:
            # Check if iperf3 is available on both nodes
            try:
                src_conn.run("which iperf3", hide=True)
            except:
                print(f"[Error] iperf3 not available on source node {src_conn.host}")
                return np.nan

            # Start iperf3 server on destination
            # GCP: dst_conn = Connection(host=dst_ip, user="ccclr0302", connect_kwargs=connect_kwargs)
            dst_conn = Connection(host=dst_ip, user=username, connect_kwargs=connect_kwargs)
            try:
                dst_conn.run("which iperf3", hide=True)
            except:
                print(f"[Error] iperf3 not available on destination node {dst_ip}")
                dst_conn.close()
                return np.nan

            # Kill any existing iperf3 processes
            dst_conn.run("pkill -f iperf3", hide=True, warn=True)
            src_conn.run("pkill -f iperf3", hide=True, warn=True)

            # Start iperf3 server on destination (in background)
            dst_conn.run("iperf3 -s -D -1", hide=True, warn=True)  # Start server in daemon mode
            time.sleep(3)  # Wait for server to start

            # Verify server is running
            try:
                dst_conn.run("pgrep -f 'iperf3 -s'", hide=True, timeout=5)
            except:
                print(f"[Error] iperf3 server failed to start on {dst_ip}")
                return np.nan

            # Run iperf3 client from source
            cmd = f"iperf3 -c {dst_ip} -t {duration} -P {parallel} -J --connect-timeout 5000"
            result = src_conn.run(cmd, hide=True, timeout=duration + 20)

            # Check if command failed (Fabric uses 'exited' attribute, not 'returncode')
            if hasattr(result, 'exited') and result.exited != 0:
                print(f"[Error] iperf3 client failed: {getattr(result, 'stderr', 'Unknown error')}")
                return np.nan
            elif hasattr(result, 'returncode') and result.returncode != 0:
                print(f"[Error] iperf3 client failed: {getattr(result, 'stderr', 'Unknown error')}")
                return np.nan

            output = result.stdout

            # Parse JSON output
            data = json.loads(output)
            bps = None
            try:
                bps = data["end"]["sum_received"]["bits_per_second"]
            except Exception:
                try:
                    bps = data["end"]["sum_sent"]["bits_per_second"]
                except Exception:
                    bps = None

            if bps is None:
                print(f"[Error] Could not parse bandwidth from iperf3 output")
                return np.nan

            bandwidth_mbps = float(bps) / 1e6

            # Validate result (reasonable bandwidth range)
            if bandwidth_mbps < 1.0 or bandwidth_mbps > 100000.0:  # 1Mbps to 100Gbps
                print(f"[Warning] Unusual bandwidth measurement: {bandwidth_mbps:.2f} Mbps")
                return np.nan

            return bandwidth_mbps

        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse iperf3 JSON output: {e}")
            return np.nan
        except Exception as e:
            print(f"[Error] iperf3 measurement failed {src_conn.host} → {dst_ip}: {e}")
            # Try fallback: estimate bandwidth based on latency (rough approximation)
            try:
                # Use ping to get basic connectivity and latency
                ping_result = src_conn.run(f"ping -c 3 -W 2 {dst_ip}", hide=True, timeout=10)
                # Check if ping succeeded
                if (hasattr(ping_result, 'exited') and ping_result.exited == 0) or \
                   (hasattr(ping_result, 'returncode') and ping_result.returncode == 0):
                    # Very rough estimation: assume 100Mbps for reachable hosts
                    # This is just a placeholder - real bandwidth estimation is complex
                    print(f"[Fallback] Using estimated bandwidth for reachable host: {dst_ip}")
                    return 100.0  # 100 Mbps default
            except:
                pass
            return np.nan
        finally:
            # Clean up servers
            try:
                if dst_conn:
                    dst_conn.run("pkill -f iperf3", hide=True, warn=True)
                    dst_conn.close()
            except:
                pass

    print("=== Measuring Bandwidth from Current Node ===")
    bandwidth_vector = []
    measured_count = 0

    # Connect to source node for running client
    # GCP: src_conn = Connection(host=current_node_ip, user="ccclr0302", connect_kwargs=connect_kwargs)
    src_conn = Connection(host=current_node_ip, user=username, connect_kwargs=connect_kwargs)

    for target_name, target_ip in target_nodes:
        bandwidth = measure_iperf3_bandwidth(src_conn, target_ip, duration=duration, parallel=parallel)
        bandwidth_vector.append(bandwidth if not np.isnan(bandwidth) else np.nan)

        if not np.isnan(bandwidth):
            print(".2f")
            measured_count += 1
        else:
            print(f"  {source_node} → {target_name}: Failed")

    src_conn.close()

    print(f"\nSuccessfully measured bandwidth to {measured_count}/{len(target_nodes)} nodes")

    # Calculate statistics
    valid_bandwidths = [bw for bw in bandwidth_vector if not np.isnan(bw)]

    if valid_bandwidths:
        mean_bw = np.mean(valid_bandwidths)
        std_bw = np.std(valid_bandwidths)
        min_bw = np.min(valid_bandwidths)
        max_bw = np.max(valid_bandwidths)

        print("\n=== Bandwidth Statistics ===")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"Success Rate: {len(valid_bandwidths)}/{len(bandwidth_vector)}")
    else:
        print("\n=== No Valid Measurements ===")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Output the bandwidth vector as JSON for easy parsing
    output_data = {
        'source_node': source_node,
        'source_ip': current_node_ip,
        'target_nodes': [{'name': name, 'ip': ip} for name, ip in target_nodes],
        'bandwidths': [float(x) if not np.isnan(x) else None for x in bandwidth_vector],
        'timestamp': timestamp
    }

    # Print JSON output that can be captured by the calling process
    print("BANDWIDTH_VECTOR_JSON_START")
    print(json.dumps(output_data))
    print("BANDWIDTH_VECTOR_JSON_END")

    # Also save to file for debugging
    np.save(f"bandwidth_vector_{timestamp}.npy", np.array(bandwidth_vector))
    print(f"\nResults also saved to: bandwidth_vector_{timestamp}.npy", file=sys.stderr)

    return
