# Copyright(C) Facebook, Inc. and its affiliates.
from json import dump, load
from collections import OrderedDict


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

            # Optional CMAB checkpoint path passed to RL controller (--resume-from).
            # None / empty / false => start training from scratch.
            resume_from = json.get('cmab_resume_from', None)
            if resume_from in (None, '', False):
                self.cmab_resume_from = None
            else:
                if not isinstance(resume_from, str):
                    raise ConfigError('cmab_resume_from must be a string path or null')
                self.cmab_resume_from = resume_from

            # RL algorithm: "cmab" (RF-TS), "xgboost", "gp_bo" (GP-UCB), or "kernel_ucb".
            rl_algo = json.get('rl_algo', 'cmab')
            if rl_algo in (None, ''):
                rl_algo = 'cmab'
            if not isinstance(rl_algo, str) or rl_algo.lower() not in (
                'cmab',
                'xgboost',
                'gp_bo',
                'kernel_ucb',
            ):
                raise ConfigError(
                    'rl_algo must be "cmab", "xgboost", "gp_bo", or "kernel_ucb"'
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

            # RF/XGBoost random_state. Pair numeric vs one_hot with the same
            # seed; change it only between experiment repetitions.
            try:
                cmab_seed = int(json.get('cmab_seed', 0) or 0)
            except (TypeError, ValueError) as e:
                raise ConfigError('cmab_seed must be an integer >= 0') from e
            if cmab_seed < 0:
                raise ConfigError('cmab_seed must be an integer >= 0')
            self.cmab_seed = cmab_seed

            # Unified warmup passed to controller/trainer:
            # cmab -> skip N policy updates; gp_bo/kernel_ucb -> N cold-start samples before fit.
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

            enable_acc = json.get('enable_accelerator', False)
            if enable_acc in (None, '', False, 0, '0', 'false', 'False'):
                self.enable_accelerator = False
            else:
                self.enable_accelerator = bool(enable_acc)

            cmab_policy = json.get('cmab_policy', 'rf_ts')
            if cmab_policy in (None, ''):
                cmab_policy = 'rf_ts'
            if not isinstance(cmab_policy, str):
                raise ConfigError('cmab_policy must be a string')
            cmab_policy = cmab_policy.lower()
            if cmab_policy not in (
                'rf_ts', 'random', 'default', 'round_robin',
                'factorized', 'combined',
            ):
                raise ConfigError(
                    'cmab_policy must be one of rf_ts, random, default, '
                    'round_robin, factorized, combined'
                )
            self.cmab_policy = cmab_policy

            try:
                cmab_start_pos = int(json.get('cmab_start_pos', 0) or 0)
            except (TypeError, ValueError) as e:
                raise ConfigError('cmab_start_pos must be an integer >= 0') from e
            if cmab_start_pos < 0:
                raise ConfigError('cmab_start_pos must be an integer >= 0')
            self.cmab_start_pos = cmab_start_pos

            # Probe every N consensus epochs on node 0; all nodes apply at detect_epoch+5.
            acc_period = json.get('accelerator_period', 100)
            if acc_period in (None, ''):
                acc_period = 100
            try:
                acc_period = int(acc_period)
            except (TypeError, ValueError) as e:
                raise ConfigError('accelerator_period must be an integer >= 1') from e
            if acc_period < 1:
                raise ConfigError('accelerator_period must be an integer >= 1')
            self.accelerator_period = acc_period

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
