# Copyright(C) Facebook, Inc. and its affiliates.
from json import dump, load
from collections import OrderedDict
import math


class ConfigError(Exception):
    pass


class Key:
    def __init__(self, name, secret):
        self.name = name
        self.secret = secret

    @classmethod
    def from_file(cls, filename):
        assert isinstance(filename, str)
        with open(filename, 'r') as f:
            data = load(f)
        return cls(data['name'], data['secret'])


class Committee:
    ''' The committee looks as follows:
        "authorities: {
            "name": {
                "stake": 1,
                "consensus: {
                    "consensus_to_consensus": x.x.x.x:x,
                },
                "primary: {
                    "primary_to_primary": x.x.x.x:x,
                    "worker_to_primary": x.x.x.x:x,
                },
                "workers": {
                    "0": {
                        "primary_to_worker": x.x.x.x:x,
                        "worker_to_worker": x.x.x.x:x,
                        "transactions": x.x.x.x:x
                    },
                    ...
                }
            },
            ...
        }
    '''

    def __init__(self, addresses, base_port):
        ''' The `addresses` field looks as follows:
            { 
                "name": ["host", "host", ...],
                ...
            }
        '''
        assert isinstance(addresses, OrderedDict)
        assert all(isinstance(x, str) for x in addresses.keys())
        assert all(
            isinstance(x, list) and len(x) > 1 for x in addresses.values()
        )
        assert all(
            isinstance(x, str) for y in addresses.values() for x in y
        )
        assert len({len(x) for x in addresses.values()}) == 1
        assert isinstance(base_port, int) and base_port > 1024

        port = base_port
        self.json = {'authorities': OrderedDict()}

        for name, hosts in addresses.items():
            host = hosts.pop(0)
            consensus_addr = {
                'consensus_to_consensus': f'{host}:{port}',
            }
            port += 1

            primary_addr = {
                'primary_to_primary': f'{host}:{port}',
                'worker_to_primary': f'{host}:{port + 1}'
            }
            port += 2

            workers_addr = OrderedDict()
            for j, host in enumerate(hosts):
                workers_addr[j] = {
                    'primary_to_worker': f'{host}:{port}',
                    'transactions': f'{host}:{port + 1}',
                    'worker_to_worker': f'{host}:{port + 2}',
                }
                port += 3

            self.json['authorities'][name] = {
                'stake': 1,
                'consensus': consensus_addr,
                'primary': primary_addr,
                'workers': workers_addr
            }

    def primary_addresses(self, faults=0):
        ''' Returns an ordered list of primaries' addresses. '''
        assert faults < self.size()
        addresses = []
        good_nodes = self.size() - faults
        for authority in list(self.json['authorities'].values())[:good_nodes]:
            addresses += [authority['primary']['primary_to_primary']]
        return addresses

    def workers_addresses(self, faults=0):
        ''' Returns an ordered list of list of workers' addresses. '''
        assert faults < self.size()
        addresses = []
        good_nodes = self.size() - faults
        for authority in list(self.json['authorities'].values())[:good_nodes]:
            authority_addresses = []
            for id, worker in authority['workers'].items():
                authority_addresses += [(id, worker['transactions'])]
            addresses.append(authority_addresses)
        return addresses

    def ips(self, name=None):
        ''' Returns all the ips associated with an authority (in any order). '''
        if name is None:
            names = list(self.json['authorities'].keys())
        else:
            names = [name]

        ips = set()
        for name in names:
            addresses = self.json['authorities'][name]['consensus']
            ips.add(self.ip(addresses['consensus_to_consensus']))

            addresses = self.json['authorities'][name]['primary']
            ips.add(self.ip(addresses['primary_to_primary']))
            ips.add(self.ip(addresses['worker_to_primary']))

            for worker in self.json['authorities'][name]['workers'].values():
                ips.add(self.ip(worker['primary_to_worker']))
                ips.add(self.ip(worker['worker_to_worker']))
                ips.add(self.ip(worker['transactions']))

        return list(ips)

    def remove_nodes(self, nodes):
        ''' remove the `nodes` last nodes from the committee. '''
        assert nodes < self.size()
        for _ in range(nodes):
            self.json['authorities'].popitem()

    def size(self):
        ''' Returns the number of authorities. '''
        return len(self.json['authorities'])

    def workers(self):
        ''' Returns the total number of workers (all authorities altogether). '''
        return sum(len(x['workers']) for x in self.json['authorities'].values())

    def print(self, filename):
        assert isinstance(filename, str)
        with open(filename, 'w') as f:
            dump(self.json, f, indent=4, sort_keys=True)

    @staticmethod
    def ip(address):
        assert isinstance(address, str)
        return address.split(':')[0]


class LocalCommittee(Committee):
    def __init__(self, names, port, workers):
        assert isinstance(names, list)
        assert all(isinstance(x, str) for x in names)
        assert isinstance(port, int)
        assert isinstance(workers, int) and workers > 0
        addresses = OrderedDict((x, ['127.0.0.1']*(1+workers)) for x in names)
        super().__init__(addresses, port)


class NodeParameters:
    def __init__(self, json):
        required_ints = [
            'timeout_delay', 'header_size', 'max_header_delay', 'gc_depth',
            'sync_retry_delay', 'sync_retry_nodes', 'batch_size', 'max_batch_delay','cut_condition_type'
        ]
        optional_bools = [
            'use_optimistic_tips', 'use_parallel_proposals', 'use_fast_path',
            'use_ride_share', 'simulate_asynchrony', 'use_fast_sync', 'use_exponential_timeouts'
        ]
        optional_ints = [
            'k', 'fast_path_timeout', 'car_timeout', 'egress_penalty',
            'epoch_slots', 'window_size'
        ]
        optional_lists = [
            'asynchrony_type', 'asynchrony_start', 'asynchrony_duration', 'affected_nodes',
            # Optional: explicit node ids per async window (region-based selection in remote.py).
            'asynchrony_node_ids_per_window',
            # Optional: resolved per-node egress penalties generated in remote.py.
            'egress_penalty_per_node'
        ]
        hotspot_info =[
            'node_id', 'hotspot-windows', 'hotspot-nodes', 'hotspot-rates'
        ]
        # for key in required_ints:
        #     if key not in json or not isinstance(json[key], int):
        #         raise ConfigError(f'Malformed parameters: missing or invalid key {key}')
        # for key in optional_bools:
        #     if key in json and not isinstance(json[key], bool):
        #         raise ConfigError(f'Invalid type for {key}, should be bool')
        # for key in optional_ints:
        #     if key in json and not isinstance(json[key], int):
        #         raise ConfigError(f'Invalid type for {key}, should be int')
        # for key in optional_lists:
        #     if key in json and not isinstance(json[key], list):
        #         raise ConfigError(f'Invalid type for {key}, should be list')
        # for key in hotspot_info:
        #     if key in json and not isinstance(json[key], list):
        #         raise ConfigError(f'Invalid type for {key}, should be list')
        self.json = json

    def print(self, filename):
        assert isinstance(filename, str)
        with open(filename, 'w') as f:
            dump(self.json, f, indent=4, sort_keys=True)


class BenchParameters:
    def __init__(self, json):
        try:
            print(json)
            self.faults = int(json['faults'])

            nodes = json['nodes']
            nodes = nodes if isinstance(nodes, list) else [nodes]
            if not nodes or any(x <= 1 for x in nodes):
                raise ConfigError('Missing or invalid number of nodes')
            self.nodes = [int(x) for x in nodes]

            rate = json['rate']
            rate = rate if isinstance(rate, list) else [rate]
            if not rate:
                raise ConfigError('Missing input rate')
            self.rate = [int(x) for x in rate]

            self.workers = int(json['workers'])

            if 'collocate' in json:
                self.collocate = bool(json['collocate'])
            else:
                self.collocate = True

            self.tx_size = int(json['tx_size'])

            self.duration = int(json['duration'])

            self.runs = int(json['runs']) if 'runs' in json else 1

            new_result_file_per_run = json.get(
                'new_result_file_per_run', False
            )
            if not isinstance(new_result_file_per_run, bool):
                raise ConfigError('new_result_file_per_run must be true or false')
            self.new_result_file_per_run = new_result_file_per_run

            # Experiment-level RL switch. Metrics collection and change
            # detection are independent and continue running when disabled.
            enable_rl = json.get('enable_rl', True)
            if not isinstance(enable_rl, bool):
                raise ConfigError('enable_rl must be true or false')
            self.enable_rl = enable_rl

            # Optional CMAB checkpoint path passed to RL controller (--resume-from).
            # None / empty / false => start training from scratch.
            resume_from = json.get('cmab_resume_from', None)
            if resume_from in (None, '', False):
                self.cmab_resume_from = None
            else:
                if not isinstance(resume_from, str):
                    raise ConfigError('cmab_resume_from must be a string path or null')
                self.cmab_resume_from = resume_from

            # CloudLab checkpoint switch. When checkpoint_path is set, it is a
            # path on the node running `fab remote` (normally node0); Fabric
            # copies that file to every controller node before startup.
            #
            # Defaulting to bool(cmab_resume_from) keeps older configurations
            # working. An explicit false always means a cold start.
            enable_checkpoint = json.get(
                'enable_checkpoint', bool(self.cmab_resume_from)
            )
            if not isinstance(enable_checkpoint, bool):
                raise ConfigError('enable_checkpoint must be true or false')
            self.enable_checkpoint = enable_checkpoint

            checkpoint_path = json.get('checkpoint_path', None)
            if checkpoint_path in (None, '', False):
                self.checkpoint_path = None
            else:
                if not isinstance(checkpoint_path, str):
                    raise ConfigError('checkpoint_path must be a string path or null')
                self.checkpoint_path = checkpoint_path

            if (
                self.enable_rl
                and self.enable_checkpoint
                and self.checkpoint_path is None
                and self.cmab_resume_from is None
            ):
                raise ConfigError(
                    'enable_checkpoint=true requires checkpoint_path '
                    'or legacy cmab_resume_from'
                )

            # RL algorithm selected by the experiment configuration.
            rl_algo = json.get('rl_algo', 'cmab')
            if rl_algo in (None, ''):
                rl_algo = 'cmab'
            supported_algorithms = (
                'cmab',
                'gp_bo',
                'kernel_ucb',
                'dqn',
                'coverage_round_robin',
            )
            if (
                not isinstance(rl_algo, str)
                or rl_algo.lower() not in supported_algorithms
            ):
                raise ConfigError(
                    'rl_algo must be "cmab", "gp_bo", "kernel_ucb", '
                    '"dqn", or "coverage_round_robin"'
                )
            self.rl_algo = rl_algo.lower()
            cmab_action_encoding = json.get(
                'cmab_action_encoding', 'numeric'
            )
            if (
                not isinstance(cmab_action_encoding, str)
                or cmab_action_encoding.lower() not in ('numeric', 'one_hot')
            ):
                raise ConfigError(
                    'cmab_action_encoding must be "numeric" or "one_hot"'
                )
            self.cmab_action_encoding = cmab_action_encoding.lower()
            enable_cmab_pairwise_residual_rf = json.get(
                'enable_cmab_pairwise_residual_rf', False
            )
            if not isinstance(enable_cmab_pairwise_residual_rf, bool):
                raise ConfigError(
                    'enable_cmab_pairwise_residual_rf must be true or false'
                )
            self.enable_cmab_pairwise_residual_rf = (
                enable_cmab_pairwise_residual_rf
            )
            if self.rl_algo == 'coverage_round_robin' and self.enable_checkpoint:
                raise ConfigError(
                    'coverage_round_robin is a data collector and cannot load '
                    'a policy checkpoint; set enable_checkpoint=false'
                )

            # Unified warmup passed to controller/trainer:
            # cmab -> skip N policy updates; gp_bo -> N cold-start samples before GP fit.
            warmup = json.get('rl_warmup_iterations', 5)
            if warmup in (None, ''):
                warmup = 5
            try:
                warmup = int(warmup)
            except (TypeError, ValueError) as e:
                raise ConfigError('rl_warmup_iterations must be an integer >= 0') from e
            if warmup < 0:
                raise ConfigError('rl_warmup_iterations must be an integer >= 0')
            self.rl_warmup_iterations = warmup

            enable_cmab_protocol_rules = json.get(
                'enable_cmab_protocol_rules', False
            )
            if not isinstance(enable_cmab_protocol_rules, bool):
                raise ConfigError(
                    'enable_cmab_protocol_rules must be true or false'
                )
            self.enable_cmab_protocol_rules = enable_cmab_protocol_rules

            enable_cmab_transition_export = json.get(
                'enable_cmab_transition_export', False
            )
            if not isinstance(enable_cmab_transition_export, bool):
                raise ConfigError(
                    'enable_cmab_transition_export must be true or false'
                )
            self.enable_cmab_transition_export = enable_cmab_transition_export

            transition_export_dir = json.get(
                'cmab_transition_export_dir', '/local/autopilot_offline_data'
            )
            if (
                not isinstance(transition_export_dir, str)
                or not transition_export_dir.strip()
            ):
                raise ConfigError(
                    'cmab_transition_export_dir must be a non-empty path'
                )
            self.cmab_transition_export_dir = transition_export_dir

            environment_label = json.get('cmab_environment_label', 'unlabeled')
            if (
                not isinstance(environment_label, str)
                or not environment_label.strip()
            ):
                raise ConfigError(
                    'cmab_environment_label must be a non-empty string'
                )
            self.cmab_environment_label = environment_label.strip()
            if (
                self.enable_rl
                and self.rl_algo == 'coverage_round_robin'
                and not self.enable_cmab_transition_export
            ):
                raise ConfigError(
                    'coverage_round_robin requires '
                    'enable_cmab_transition_export=true'
                )

            # Maximum number of iterations processed by this RL trainer run.
            # None means that training continues until the experiment shuts it down.
            max_training_iterations = json.get('rl_max_training_iterations', 200)
            if max_training_iterations in (None, ''):
                self.rl_max_training_iterations = None
            else:
                if isinstance(max_training_iterations, bool):
                    raise ConfigError(
                        'rl_max_training_iterations must be a positive integer or null'
                    )
                try:
                    max_training_iterations = int(max_training_iterations)
                except (TypeError, ValueError) as e:
                    raise ConfigError(
                        'rl_max_training_iterations must be a positive integer or null'
                    ) from e
                if max_training_iterations <= 0:
                    raise ConfigError(
                        'rl_max_training_iterations must be a positive integer or null'
                    )
                self.rl_max_training_iterations = max_training_iterations

            def positive_int(name, default):
                value = json.get(name, default)
                if isinstance(value, bool):
                    raise ConfigError(f'{name} must be a positive integer')
                try:
                    value = int(value)
                except (TypeError, ValueError) as e:
                    raise ConfigError(f'{name} must be a positive integer') from e
                if value <= 0:
                    raise ConfigError(f'{name} must be a positive integer')
                return value

            def positive_float(name, default, allow_zero=False):
                value = json.get(name, default)
                if isinstance(value, bool):
                    raise ConfigError(f'{name} must be a finite number')
                try:
                    value = float(value)
                except (TypeError, ValueError) as e:
                    raise ConfigError(f'{name} must be a finite number') from e
                lower_bound_valid = value >= 0 if allow_zero else value > 0
                if not lower_bound_valid or not math.isfinite(value):
                    qualifier = 'non-negative' if allow_zero else 'positive'
                    raise ConfigError(f'{name} must be a finite {qualifier} number')
                return value

            def non_negative_int(name, default):
                value = json.get(name, default)
                if isinstance(value, bool):
                    raise ConfigError(f'{name} must be a non-negative integer')
                try:
                    value = int(value)
                except (TypeError, ValueError) as e:
                    raise ConfigError(
                        f'{name} must be a non-negative integer'
                    ) from e
                if value < 0:
                    raise ConfigError(f'{name} must be a non-negative integer')
                return value

            # Continuous-timeout KernelUCB controls. They are parsed for every
            # run so switching rl_algo in fabfile is the only required action.
            self.kernel_ucb_alpha = positive_float(
                'kernel_ucb_alpha', 1.0, allow_zero=True
            )
            self.kernel_ucb_regularization = positive_float(
                'kernel_ucb_regularization', 0.1
            )
            self.kernel_ucb_length_scale = positive_float(
                'kernel_ucb_length_scale', 1.0
            )
            self.kernel_ucb_timeout_min = positive_float(
                'kernel_ucb_timeout_min', 1.0
            )
            self.kernel_ucb_timeout_max = positive_float(
                'kernel_ucb_timeout_max', 300.0
            )
            if self.kernel_ucb_timeout_min > self.kernel_ucb_timeout_max:
                raise ConfigError(
                    'kernel_ucb_timeout_min cannot exceed kernel_ucb_timeout_max'
                )
            self.kernel_ucb_optimizer_restarts = positive_int(
                'kernel_ucb_optimizer_restarts', 5
            )
            self.kernel_ucb_replay_window = positive_int(
                'kernel_ucb_replay_window', 200
            )

            # Centralized node0 DQN controls.  Only node0 owns the neural
            # network/replay buffer; every primary runs a TCP action receiver.
            self.dqn_training_node = non_negative_int('dqn_training_node', 0)
            if self.dqn_training_node != 0:
                raise ConfigError('centralized DQN currently requires dqn_training_node=0')
            self.dqn_action_port = positive_int('dqn_action_port', 19100)
            if self.dqn_action_port > 65535:
                raise ConfigError('dqn_action_port must be <= 65535')
            self.dqn_action_timeout = positive_float('dqn_action_timeout', 2.0)
            self.dqn_action_retries = non_negative_int('dqn_action_retries', 2)
            self.dqn_learning_rate = positive_float('dqn_learning_rate', 1e-3)
            self.dqn_gamma = positive_float(
                'dqn_gamma', 0.90, allow_zero=True
            )
            if self.dqn_gamma > 1:
                raise ConfigError('dqn_gamma must be between 0 and 1')
            self.dqn_replay_capacity = positive_int(
                'dqn_replay_capacity', 2000
            )
            self.dqn_batch_size = positive_int('dqn_batch_size', 32)
            self.dqn_learning_starts = positive_int('dqn_learning_starts', 32)
            if self.dqn_replay_capacity < self.dqn_batch_size:
                raise ConfigError(
                    'dqn_replay_capacity must be at least dqn_batch_size'
                )
            if self.dqn_learning_starts < self.dqn_batch_size:
                raise ConfigError(
                    'dqn_learning_starts must be at least dqn_batch_size'
                )
            self.dqn_target_update_interval = positive_int(
                'dqn_target_update_interval', 20
            )
            self.dqn_epsilon_start = positive_float(
                'dqn_epsilon_start', 1.0, allow_zero=True
            )
            self.dqn_epsilon_end = positive_float(
                'dqn_epsilon_end', 0.05, allow_zero=True
            )
            if not (
                0 <= self.dqn_epsilon_end <= self.dqn_epsilon_start <= 1
            ):
                raise ConfigError(
                    'DQN epsilon values must satisfy 0 <= end <= start <= 1'
                )
            self.dqn_epsilon_decay_steps = positive_int(
                'dqn_epsilon_decay_steps', 200
            )
            self.dqn_gradient_updates = positive_int(
                'dqn_gradient_updates', 1
            )
            self.dqn_gradient_clip = positive_float(
                'dqn_gradient_clip', 10.0
            )
            self.dqn_hidden_dim = positive_int('dqn_hidden_dim', 64)
            self.cmab_seed = non_negative_int('cmab_seed', 0)
            self.dqn_seed = non_negative_int('dqn_seed', 0)
            self.coverage_seed = non_negative_int('coverage_seed', 0)
            checkpoint_load_mode = json.get(
                'dqn_checkpoint_load_mode', 'resume'
            )
            if checkpoint_load_mode not in ('resume', 'finetune'):
                raise ConfigError(
                    'dqn_checkpoint_load_mode must be "resume" or "finetune"'
                )
            self.dqn_checkpoint_load_mode = checkpoint_load_mode

            enable_reward_change_monitor = json.get(
                'enable_reward_change_monitor', True
            )
            if not isinstance(enable_reward_change_monitor, bool):
                raise ConfigError(
                    'enable_reward_change_monitor must be true or false'
                )
            self.enable_reward_change_monitor = enable_reward_change_monitor

            self.reward_change_window_size = positive_int(
                'reward_change_window_size', 8
            )
            self.reward_change_lag = positive_int('reward_change_lag', 3)
            self.reward_change_confirmations = positive_int(
                'reward_change_confirmations', 3
            )

            threshold = json.get('reward_change_threshold', 0.30)
            if isinstance(threshold, bool):
                raise ConfigError(
                    'reward_change_threshold must be a finite non-negative number'
                )
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as e:
                raise ConfigError(
                    'reward_change_threshold must be a finite non-negative number'
                ) from e
            if threshold < 0 or not math.isfinite(threshold):
                raise ConfigError(
                    'reward_change_threshold must be a finite non-negative number'
                )
            self.reward_change_threshold = threshold

            # Optional phase-one A/B experience matching. The monitor only
            # reports the nearest checkpoint pool; it never mutates CMAB data.
            enable_experience_matching = json.get(
                'enable_experience_matching', False
            )
            if not isinstance(enable_experience_matching, bool):
                raise ConfigError('enable_experience_matching must be true or false')
            # The reward-change monitor is the parent process for experience
            # matching.  Its master switch therefore disables matching too,
            # so one fabfile edit is enough for clean algorithm comparisons.
            self.enable_experience_matching = (
                enable_experience_matching and self.enable_reward_change_monitor
            )

            def optional_path(name):
                value = json.get(name, None)
                if value in (None, '', False):
                    return None
                if not isinstance(value, str):
                    raise ConfigError(f'{name} must be a string path or null')
                return value

            self.experience_checkpoint_a = optional_path('experience_checkpoint_a')
            self.experience_checkpoint_b = optional_path('experience_checkpoint_b')
            if self.enable_experience_matching and (
                self.experience_checkpoint_a is None
                or self.experience_checkpoint_b is None
            ):
                raise ConfigError(
                    'enable_experience_matching=true requires '
                    'experience_checkpoint_a and experience_checkpoint_b'
                )
            self.experience_pool_size = positive_int('experience_pool_size', 200)
            self.experience_match_reward_count = positive_int(
                'experience_match_reward_count', 3
            )

            self.simulate_partition = bool(json['simulate_partition'])

            self.partition_nodes = int(json['partition_nodes'])
            self.partition_start = int(json['partition_start'])
            self.partition_duration = int(json['partition_duration'])
            
            # New hotspot parameters
            self.enable_hotspot = bool(json.get('enable_hotspot'))
            
            if self.enable_hotspot:
                # Hotspot time windows in format [[start1, end1], [start2, end2], ...]
                self.hotspot_windows = json.get('hotspot_windows')
                if not isinstance(self.hotspot_windows, list):
                    raise ConfigError('hotspot_windows must be a list of [start, end] pairs')
                
                # Validate window format
                for window in self.hotspot_windows:
                    if not isinstance(window, list) or len(window) != 2:
                        raise ConfigError('Each hotspot window must be [start, end] pair')
                    if not all(isinstance(x, int) and x >= 0 for x in window):
                        raise ConfigError('Hotspot window times must be non-negative integers')
                    if window[0] >= window[1]:
                        raise ConfigError('Hotspot window start must be less than end')
                
                # Optional: restrict hotspot nodes to specific regions (per window).
                self.hotspot_regions = json.get('hotspot_regions', [])
                if self.hotspot_regions:
                    if not isinstance(self.hotspot_regions, list):
                        raise ConfigError('hotspot_regions must be a list')
                    if len(self.hotspot_regions) != len(self.hotspot_windows):
                        raise ConfigError('hotspot_regions length must match hotspot_windows length')
                    normalized = []
                    for regions in self.hotspot_regions:
                        if isinstance(regions, str):
                            region_list = [regions.strip().lower()]
                        elif isinstance(regions, list):
                            region_list = [r.strip().lower() for r in regions if isinstance(r, str) and r.strip()]
                        else:
                            raise ConfigError('hotspot_regions item must be string or list of strings')
                        if not region_list:
                            raise ConfigError('hotspot_regions item must contain at least one region')
                        normalized.append(region_list)
                    self.hotspot_regions = normalized
                else:
                    self.hotspot_regions = []

                # Number of hotspot nodes for each window (and optionally per region).
                # Without hotspot_regions: [1, 2] -> one count per window.
                # With hotspot_regions: [[1, 1, 2]] -> per-window per-region counts
                # (aligned with hotspot_regions).
                self.hotspot_nodes = json.get('hotspot_nodes')
                if not isinstance(self.hotspot_nodes, list):
                    raise ConfigError('hotspot_nodes must be a list')
                if len(self.hotspot_nodes) != len(self.hotspot_windows):
                    raise ConfigError('hotspot_nodes length must match hotspot_windows length')

                if self.hotspot_regions:
                    normalized_nodes = []
                    for w, counts in enumerate(self.hotspot_nodes):
                        if isinstance(counts, int):
                            if counts <= 0:
                                raise ConfigError('hotspot_nodes must be positive integers')
                            counts = [counts] * len(self.hotspot_regions[w])
                        elif isinstance(counts, list):
                            if len(counts) != len(self.hotspot_regions[w]):
                                raise ConfigError(
                                    'hotspot_nodes[window] length must match hotspot_regions[window] length'
                                )
                            if not all(isinstance(x, int) and x > 0 for x in counts):
                                raise ConfigError('hotspot_nodes must be positive integers')
                        else:
                            raise ConfigError('hotspot_nodes item must be int or list of ints')
                        normalized_nodes.append(counts)
                    self.hotspot_nodes = normalized_nodes
                elif not all(isinstance(x, int) and x > 0 for x in self.hotspot_nodes):
                    raise ConfigError('hotspot_nodes must be positive integers')

                # Required: per-window per-region hotspot rates.
                # Aligned with egress_penalty nesting:
                #   asynchrony_regions = [['utah']]
                #   asynchrony_nodes   = [2]
                #   egress_penalty     = [[[200, 300]]]   # window -> region -> per-node
                #
                # Hotspot equivalents:
                #   hotspot_regions      = [['utah']]
                #   hotspot_nodes        = [[3]]
                #   hotspot_region_rates = [[[0.5, 0.5, 0.3]]]  # window -> region -> per-node
                #
                # Backward compatible scalar-per-region form is still accepted:
                #   hotspot_region_rates = [[0.9, 0.6]]
                self.hotspot_region_rates = json.get('hotspot_region_rates', [])
                if not isinstance(self.hotspot_region_rates, list):
                    raise ConfigError('hotspot_region_rates must be a list')
                if len(self.hotspot_region_rates) != len(self.hotspot_windows):
                    raise ConfigError('hotspot_region_rates length must match hotspot_windows length')
                if not self.hotspot_regions:
                    raise ConfigError('hotspot_region_rates requires hotspot_regions')

                normalized_region_rates = []
                for w, rates in enumerate(self.hotspot_region_rates):
                    if not isinstance(rates, list):
                        raise ConfigError('Each hotspot_region_rates item must be a list')
                    if len(rates) != len(self.hotspot_regions[w]):
                        raise ConfigError(
                            'hotspot_region_rates[window] length must match hotspot_regions[window] length'
                        )

                    region_node_counts = self.hotspot_nodes[w]
                    if isinstance(region_node_counts, int):
                        region_node_counts = [region_node_counts] * len(self.hotspot_regions[w])

                    normalized_window = []
                    for r_idx, rate_or_list in enumerate(rates):
                        n_pick = region_node_counts[r_idx] if r_idx < len(region_node_counts) else 0
                        if isinstance(rate_or_list, (int, float)):
                            if rate_or_list < 0:
                                raise ConfigError(
                                    'hotspot_region_rates values must be non-negative numbers'
                                )
                            # Scalar rate: broadcast to all picked nodes in this region.
                            normalized_window.append([float(rate_or_list)] * n_pick)
                        elif isinstance(rate_or_list, list):
                            if len(rate_or_list) != n_pick:
                                raise ConfigError(
                                    'hotspot_region_rates[window][region] length must match '
                                    f'hotspot_nodes[window][region] ({n_pick})'
                                )
                            if not all(
                                isinstance(x, (int, float)) and x >= 0 for x in rate_or_list
                            ):
                                raise ConfigError(
                                    'hotspot_region_rates values must be non-negative numbers'
                                )
                            normalized_window.append([float(x) for x in rate_or_list])
                        else:
                            raise ConfigError(
                                'hotspot_region_rates[window][region] must be a number '
                                'or a list of per-node rates'
                            )
                    normalized_region_rates.append(normalized_window)
                self.hotspot_region_rates = normalized_region_rates
            else:
                self.hotspot_windows = []
                self.hotspot_nodes = []
                self.hotspot_regions = []
                self.hotspot_region_rates = []
            
        except KeyError as e:
            raise ConfigError(f'Malformed bench parameters: missing key {e}')

        except ValueError:
            raise ConfigError('Invalid parameters type')

        if min(self.nodes) <= self.faults:
            raise ConfigError('There should be more nodes than faults')

        # Validate hotspot parameters against total nodes
        if self.enable_hotspot:
            max_hotspot_nodes = 0
            for counts in self.hotspot_nodes:
                if isinstance(counts, int):
                    max_hotspot_nodes = max(max_hotspot_nodes, counts)
                elif isinstance(counts, list):
                    max_hotspot_nodes = max(max_hotspot_nodes, sum(counts))
            total_client_nodes = sum(self.nodes)  # Total number of client nodes
            if max_hotspot_nodes > total_client_nodes:
                raise ConfigError(f'Maximum hotspot nodes ({max_hotspot_nodes}) exceeds total client nodes ({total_client_nodes})')
            


class PlotParameters:
    def __init__(self, json):
        try:
            faults = json['faults']
            faults = faults if isinstance(faults, list) else [faults]
            self.faults = [int(x) for x in faults] if faults else [0]

            nodes = json['nodes']
            nodes = nodes if isinstance(nodes, list) else [nodes]
            if not nodes:
                raise ConfigError('Missing number of nodes')
            self.nodes = [int(x) for x in nodes]

            workers = json['workers']
            workers = workers if isinstance(workers, list) else [workers]
            if not workers:
                raise ConfigError('Missing number of workers')
            self.workers = [int(x) for x in workers]

            if 'collocate' in json:
                self.collocate = bool(json['collocate'])
            else:
                self.collocate = True

            self.tx_size = int(json['tx_size'])

            max_lat = json['max_latency']
            max_lat = max_lat if isinstance(max_lat, list) else [max_lat]
            if not max_lat:
                raise ConfigError('Missing max latency')
            self.max_latency = [int(x) for x in max_lat]

        except KeyError as e:
            raise ConfigError(f'Malformed bench parameters: missing key {e}')

        except ValueError:
            raise ConfigError('Invalid parameters type')

        if len(self.nodes) > 1 and len(self.workers) > 1:
            raise ConfigError(
                'Either the "nodes" or the "workers can be a list (not both)'
            )

    def scalability(self):
        return len(self.workers) > 1
