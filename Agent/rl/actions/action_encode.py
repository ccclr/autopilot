import numpy as np
from typing import Dict, List, Optional


class ActionCodec:
    """
    Encode/Decode PPO actions <-> Autobahn parameters

    Action prediction flow:
    1. Define legal value sets for each parameter (e.g., batch_size_values = [100000, 300000, 500000])
    2. PPO predicts indices (e.g., batch_size_idx = 0/1/2) using MultiDiscrete action space
    3. System maps indices to actual values (e.g., batch_size = batch_size_values[batch_size_idx])
    """

    def __init__(self, policy: str = "rf_ts"):
        """
        Initialize action encoder, define legal value sets and action dimensions
        
        For each parameter:
        - Define legal value set (e.g., batch_size_values = [100000, 300000, 500000])
        - PPO will predict an index (0, 1, or 2) for batch_size
        - System maps index to value: batch_size = batch_size_values[index]
        """
        # Step 1: Define legal value sets for each parameter.
        # policy == "default" means fixed one-arm configuration.
        if policy == "default":
            self.batch_size_values = [500000]
            self.header_size_values = [32]
            self.cut_condition_type_values = [3]
            self.fast_path_timeout_ms_values = [200]
            self.parallel_proposals_values = [1]
            # Inclusive continuous bounds used by GP-BO mixed search.
            self.fast_path_timeout_ms_bounds = (200, 200)
            # self.use_optimistic_tips_values = [True]
        else:
            self.batch_size_values = [100000, 500000]  # 100KB to 800KB
            self.header_size_values = [32, 64]  # 32 to 256 bytes
            self.cut_condition_type_values = [2, 3, 4]  # 1 to 2f+1 (reasonable cut conditions)
            # Discrete grid kept for CMAB; GP-BO uses continuous bounds below.
            self.fast_path_timeout_ms_values = [0, 100, 200, 300]
            self.fast_path_timeout_ms_bounds = (0, 300)  # continuous ms interval for GP-BO
            self.parallel_proposals_values = [1, 4]  # 1 to 3 parallel proposals
            # self.use_optimistic_tips_values = [True, False]  # True or False

        # self.batch_size_values = [100000]
        # self.header_size_values = [32]
        # self.cut_condition_type_values = [3]
        # self.fast_path_timeout_ms_values = [0]
        # self.parallel_proposals_values = [3]
        # self.use_optimistic_tips_values = [True]
        
        # Store all value sets in a dictionary for easy access
        self._action_values = {
            "batch_size": self.batch_size_values,
            "header_size": self.header_size_values,
            "cut_condition_type": self.cut_condition_type_values,
            "fast_path_timeout_ms": self.fast_path_timeout_ms_values,
            "parallel_proposals": self.parallel_proposals_values,
            # "use_optimistic_tips": self.use_optimistic_tips_values,
        }
        
        # Step 2: Define action dimensions (number of choices for each parameter)
        # PPO will predict indices in range [0, dim-1] for each dimension
        self.action_dims = [
            len(self.batch_size_values),  # 3 choices: indices 0, 1, 2
            len(self.header_size_values),  # 3 choices: indices 0, 1, 2
            len(self.cut_condition_type_values),  # 2 choices: indices 0, 1
            len(self.fast_path_timeout_ms_values),  # 3 choices: indices 0, 1, 2
            len(self.parallel_proposals_values),  # 2 choices: indices 0, 1
            # len(self.use_optimistic_tips_values),  # 2 choices: indices 0, 1
        ]

        self.action_space = np.array(self.action_dims)

    def decode(self, action: np.ndarray) -> Dict:
        """
        Step 3: Map PPO-predicted indices to actual parameter values
        
        This method performs the mapping: index -> actual_value
        Example: batch_size_idx=0 -> batch_size=100000, batch_size_idx=1 -> batch_size=300000, etc.

        Args:
            action: np.ndarray, MultiDiscrete action vector containing indices [batch_size_idx, header_size_idx, ...]
                   Each index should be in range [0, len(values)-1] for the corresponding parameter

        Returns:
            dict: Autobahn parameter dictionary with actual values
                  {
                      "batch_size": 100000 or 300000 or 500000,
                      "header_size": 32 or 128 or 256,
                      ...
                  }
        """
        if len(action) != len(self.action_dims):
            raise ValueError(f"Action dimension mismatch: expected {len(self.action_dims)}, got {len(action)}")
        
        # Ensure action is integer type (indices must be integers)
        action = np.asarray(action, dtype=np.int32)
        
        # Validate indices are within valid range
        for i, (dim, idx) in enumerate(zip(self.action_dims, action)):
            if idx < 0 or idx >= dim:
                raise ValueError(f"Action {i} index {idx} out of range [0, {dim-1}]")
        
        # Map indices to actual values
        return {
            "batch_size": self.batch_size_values[action[0]],  # action[0] is batch_size_idx
            "header_size": self.header_size_values[action[1]],  # action[1] is header_size_idx
            "cut_condition_type": self.cut_condition_type_values[action[2]],  # action[1] is cut_condition_type_idx
            "fast_path_timeout_ms": self.fast_path_timeout_ms_values[action[3]],  # action[3] is fast_path_timeout_ms_idx
            "parallel_proposals": self.parallel_proposals_values[action[4]],  # action[4] is parallel_proposals_idx
            # "use_optimistic_tips": self.use_optimistic_tips_values[action[4]],  # action[5] is use_optimistic_tips_idx
        }
    
    def encode(self, params: Dict) -> Optional[np.ndarray]:
        """
        Reverse mapping: Convert actual parameter values back to PPO action indices
        
        This is the inverse of decode(): actual_value -> index
        Used for encoding existing parameters into action format (e.g., for initialization)

        Args:
            params: Autobahn parameter dictionary with actual values
                   {
                       "batch_size": 100000 or 300000 or 500000,
                       "header_size": 32 or 128 or 256,
                       ...
                   }

        Returns:
            np.ndarray: PPO action vector containing indices [batch_size_idx, header_size_idx, ...]
        """
        # Define mapping from parameter names to action indices
        param_to_action = {
            "batch_size": (0, "batch_size"),
            "header_size": (1, "header_size"),
            "cut_condition_type": (2, "cut_condition_type"),
            "fast_path_timeout": (3, "fast_path_timeout_ms"),
            "k": (4, "parallel_proposals"),
            # "use_optimistic_tips": (4, "use_optimistic_tips"),
        }
        
        # Initialize action vector (all indices start at 0)
        result_action = np.zeros(len(self.action_dims), dtype=np.int32)
        
        for param_name, (idx, action_key) in param_to_action.items():
            if param_name in params:
                value = params[param_name]
                if action_key in self._action_values:
                    try:
                        # Find the index of the value in the legal value set
                        action_idx = self._action_values[action_key].index(value)
                        result_action[idx] = action_idx
                    except ValueError:
                        # If value is not in the legal set, find the closest value's index
                        values = self._action_values[action_key]
                        closest_idx = min(range(len(values)), key=lambda i: abs(values[i] - value))
                        result_action[idx] = closest_idx
        
        return result_action