import json
import time
import numpy as np
from pathlib import Path
from typing import Mapping, Optional


DQN_STATE_SCHEMA = "lane_growth_norm_0_20_plus_global_fpr_v1"


def build_dqn_state(data: Mapping) -> np.ndarray:
    """Build the shared raw DQN/CMAB state from one global-state record.

    Lane growth values are sorted by lane id and mapped into ``[0, 20]``.
    ``DQNPolicy`` performs the final ``[0, 1]`` scaling when a transition is
    inserted, so persisted offline datasets must keep this raw representation.
    """

    growth_rates = data.get("state_4_lane_vector", {}).get("growth_rates", {})
    lane_values = []
    if isinstance(growth_rates, dict):
        for _, value in sorted(growth_rates.items()):
            try:
                lane_values.append(float(value))
            except (TypeError, ValueError):
                continue

    growth_norm = [
        max(0.0, min(20.0, (value - 2.0) / (100.0 - 2.0) * 20.0))
        for value in lane_values
    ]
    try:
        fast_path_ratio = float(data.get("global_fast_path_ratio", 0.0))
    except (TypeError, ValueError):
        fast_path_ratio = 0.0

    return np.asarray([*growth_norm, fast_path_ratio], dtype=np.float32)

def parse_metrics_with_context(json_path):
    """
    Parse metrics JSON file, separating context (static) from dynamic state

    This implements the contextual policy approach where:
    policy = π(action | dynamic_state ; system_config)

    Args:
        json_path: metrics JSON file path

    Returns:
        tuple: (context, dynamic_state, reward, action)
        - context: system configuration (hardware + network) - relatively static
        - dynamic_state: workload + lane vector + fast path - changes over time
        - reward: scalar reward value
        - action: current action dict (from current_action field in metrics file)
    """
    data = None
    for attempt in range(5):
        try:
            with open(json_path, "r") as f:
                raw = f.read()
            if not raw.strip():
                raise json.JSONDecodeError("Empty file", raw, 0)
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.1)

    # # === CONTEXT (System Configuration - Static) ===
    context = []

    # # Hardware metrics - COMMENTED OUT
    # hw = data["state_1_hardware"]
    # context.extend([
    #     hw["cpu_cores"],
    #     hw["memory_gb"],
    #     hw["network_bandwidth_mbps"],
    #     hw["workers_per_node"],
    # ])

    # # Network latency vector - COMMENTED OUT
    # context.extend(data["state_2_network"]["latency_vector"])

    # # === DYNAMIC STATE (Changes over time) ===
    dynamic_state = []

    # # Workload metrics - COMMENTED OUT
    # wl = data["state_3_workload"]
    # dynamic_state.extend([wl["tx_size_bytes"], wl["tx_arrival_rate_tps"]])

    # Lane growth rates - ONLY ACTIVE STATE
    growth_rates_dict = data["state_4_lane_vector"]["growth_rates"]
    growth = list(growth_rates_dict.values())
    # Normalize growth rates from ~[2, 100] to ~[0, 20]
    growth_min = 2.0
    growth_max = 100.0
    growth_scale = 20.0
    growth_norm = []
    for value in growth:
        normalized = (float(value) - growth_min) / (growth_max - growth_min) * growth_scale
        # Clamp to keep within expected range
        normalized = max(0.0, min(growth_scale, normalized))
        growth_norm.append(normalized)
    dynamic_state.extend(growth_norm)

    # Fast path ratio - COMMENTED OUT
    dynamic_state.append(data["state_5_fast_path_ratio"])

    # Convert to numpy arrays
    context = np.array(context, dtype=np.float32)
    dynamic_state = np.array(dynamic_state, dtype=np.float32)

    # === REWARD ===
    metrics = data["end_to_end_metrics"]
    latency_ms = metrics["end_to_end_latency_ms"]

    # Reward function focused only on latency optimization
    # Use logarithmic scaling to handle wide range and ensure convergence

    # Base latency reward: higher for lower latency
    # Use 1/(1+latency) to bound the reward between 0 and 1
    latency_reward = 1.0 / (1.0 + latency_ms / 100.0)  # Scale latency by 100ms

    # Add logarithmic component for better gradient properties
    # This gives higher reward for very low latency while being smooth
    log_reward = np.log(1.0 + 1.0 / (1.0 + latency_ms))

    # Combine both components with appropriate weights
    if latency_ms == 0.0:
        reward = 0
    else:
        reward = 1000 * (1.0 / (latency_ms + 1.0))

    # === ACTION ===
    # Parse current_action from metrics file
    action = None
    if "current_action" in data:
        action = data["current_action"].copy()  # Return as dict

    return context, dynamic_state, reward, action


def get_state_dim(metrics_dir: str, sample_file: Optional[str] = None, check_consistency: bool = True, include_context: bool = True) -> int:
    """
    Calculate state dimension by parsing metrics files

    Args:
        metrics_dir: metrics file directory
        sample_file: optional sample file path, if None then automatically find the latest file
        check_consistency: whether to check dimension consistency across multiple files
        include_context: whether to include context dimensions (for training) or just dynamic state (for observation)

    Returns:
        int: state dimension

    Raises:
        ValueError: if dimension inconsistency is found
    """
    metrics_path = Path(metrics_dir)

    # If sample file is provided, use it
    if sample_file:
        json_path = Path(sample_file)
        if include_context:
            # For training: return full state dimension (context + dynamic)
            context, dynamic_state, _ = parse_metrics_with_context(str(json_path))
            return len(context) + len(dynamic_state)
        else:
            # For observation: return only dynamic state dimension
            _, dynamic_state, _ = parse_metrics_with_context(str(json_path))
            return len(dynamic_state)

    # Otherwise find metrics files
    json_files = sorted(metrics_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metrics JSON files found in directory {metrics_dir}")

    # Parse the latest file to get state dimension
    latest_file = json_files[-1]
    if include_context:
        # For training: return full state dimension
        context, dynamic_state, _ = parse_metrics_with_context(str(latest_file))
        state_dim = len(context) + len(dynamic_state)
    else:
        # For observation: return only dynamic state dimension
        _, dynamic_state, _ = parse_metrics_with_context(str(latest_file))
        state_dim = len(dynamic_state)

    # If consistency check is enabled, validate dimensions across multiple files
    if check_consistency and len(json_files) > 1:
        # Check dimensions of recent files
        check_files = json_files[-min(5, len(json_files)):]
        dims = []
        for f in check_files:
            try:
                if include_context:
                    c, d, _ = parse_metrics_with_context(str(f))
                    dims.append(len(c) + len(d))
                else:
                    _, d, _ = parse_metrics_with_context(str(f))
                    dims.append(len(d))
            except Exception:
                continue

                if dims:
                    unique_dims = set(dims)
                    if len(unique_dims) > 1:
                        import warnings
                        warnings.warn(
                            f"Dimension inconsistency found! File dimensions: {unique_dims}."
                            f"Using latest file dimension: {state_dim}."
                            f"Environment will automatically handle dimension mismatches (padding or truncation)."
                        )

    return state_dim
