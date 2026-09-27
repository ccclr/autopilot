from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .action_features import ActionFeatureEncoder

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Action-conditioned critic returning one Q value per input pair."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.network(state_action)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNPolicy:
    """Action-conditioned DQN with replay and a hard-updated target network."""

    CHECKPOINT_VERSION = 2
    Q_ARCHITECTURE = "state_action_q_v1"

    def __init__(
        self,
        *,
        state_dim: int,
        arms: Sequence[str],
        learning_rate: float = 1e-3,
        gamma: float = 0.90,
        replay_capacity: int = 2000,
        batch_size: int = 32,
        learning_starts: int = 32,
        target_update_interval: int = 20,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200,
        gradient_clip: float = 10.0,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not arms:
            raise ValueError("DQN requires at least one action")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if replay_capacity <= 0 or batch_size <= 0 or learning_starts <= 0:
            raise ValueError("replay, batch, and learning_starts must be positive")
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if target_update_interval <= 0 or epsilon_decay_steps <= 0:
            raise ValueError("target update and epsilon decay intervals must be positive")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if gradient_clip <= 0 or hidden_dim <= 0:
            raise ValueError("gradient_clip and hidden_dim must be positive")

        self.state_dim = int(state_dim)
        self.arms = tuple(str(arm) for arm in arms)
        self.action_count = len(self.arms)
        self.action_encoder = ActionFeatureEncoder(self.arms)
        self.action_feature_dim = self.action_encoder.feature_dim
        self.network_input_dim = self.state_dim + self.action_feature_dim
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.learning_starts = max(int(learning_starts), self.batch_size)
        self.target_update_interval = int(target_update_interval)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.gradient_clip = float(gradient_clip)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # One CPU thread keeps this small network predictable and avoids
        # competing with the protocol process on node0.
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.online_network = QNetwork(self.network_input_dim, self.hidden_dim).to(
            self.device
        )
        self.target_network = QNetwork(self.network_input_dim, self.hidden_dim).to(
            self.device
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self._action_features = torch.from_numpy(
            self.action_encoder.features.copy()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer: deque[Transition] = deque(maxlen=self.replay_capacity)
        self.rng = np.random.default_rng(self.seed)
        self.decision_steps = 0
        self.gradient_steps = 0
        self.transitions_seen = 0
        logger.info(
            "DQN_POLICY_INIT architecture=%s state_dim=%d action_feature_dim=%d "
            "input_dim=%d actions=%d action_features=%s",
            self.Q_ARCHITECTURE,
            self.state_dim,
            self.action_feature_dim,
            self.network_input_dim,
            self.action_count,
            self.action_encoder.feature_names,
        )

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.decision_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def select_action(self, state: np.ndarray) -> tuple[int, dict[str, float | str]]:
        state_array = self._prepare_state(state)
        epsilon = self.epsilon
        probe = float(self.rng.random())
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
            q_values = (
                self._q_values_for_all_actions(self.online_network, state_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        if probe < epsilon:
            action_id = int(self.rng.integers(self.action_count))
            mode = "explore"
        else:
            max_value = float(np.max(q_values))
            candidates = np.flatnonzero(np.isclose(q_values, max_value))
            action_id = int(candidates[0])
            mode = "greedy"

        self.decision_steps += 1
        top_indices = np.argsort(q_values)[::-1][: min(5, self.action_count)]
        logger.info(
            "DQN_SELECTED step=%d mode=%s epsilon=%.6f probe=%.6f action_id=%d "
            "arm=%s q=%.6f top=%s",
            self.decision_steps,
            mode,
            epsilon,
            probe,
            action_id,
            self.arms[action_id],
            float(q_values[action_id]),
            [(int(index), float(q_values[index])) for index in top_indices],
        )
        return action_id, {
            "mode": mode,
            "epsilon": epsilon,
            "q_value": float(q_values[action_id]),
        }

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        transition = self.prepare_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.replay_buffer.append(transition)
        self.transitions_seen += 1
        logger.info(
            "DQN_REPLAY_ADD transitions=%d active=%d action=%d reward=%.6f",
            self.transitions_seen,
            len(self.replay_buffer),
            action,
            reward,
        )

    def prepare_transition(
        self,
        *,
        state: np.ndarray | Iterable[float],
        action: int,
        reward: float,
        next_state: np.ndarray | Iterable[float],
        done: bool = False,
    ) -> Transition:
        """Validate and normalize one transition without inserting it in replay."""

        if action < 0 or action >= self.action_count:
            raise ValueError(f"action index out of range: {action}")
        return Transition(
            state=self._prepare_state(state),
            action=int(action),
            reward=float(reward),
            next_state=self._prepare_state(next_state),
            done=bool(done),
        )

    def train(self, updates: int = 1) -> list[float]:
        if updates <= 0:
            return []
        if len(self.replay_buffer) < self.learning_starts:
            logger.info(
                "DQN_LEARNING_WAIT replay=%d learning_starts=%d",
                len(self.replay_buffer),
                self.learning_starts,
            )
            return []

        losses = []
        for _ in range(updates):
            losses.append(self._train_one_batch())
        return losses

    def _train_one_batch(self) -> float:
        indices = self.rng.choice(
            len(self.replay_buffer), size=self.batch_size, replace=False
        )
        buffer_list = list(self.replay_buffer)
        batch = [buffer_list[int(index)] for index in indices]
        return self.train_batch(batch)

    def train_batch(self, batch: Sequence[Transition]) -> float:
        """Apply one update from an explicit batch, used by offline training."""

        if not batch:
            raise ValueError("DQN training batch cannot be empty")

        states = torch.from_numpy(np.stack([item.state for item in batch])).to(
            self.device
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.from_numpy(
            np.stack([item.next_state for item in batch])
        ).to(self.device)
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self._q_values_for_actions(
            self.online_network, states, actions
        )
        with torch.no_grad():
            # Vanilla DQN target.  Double-DQN is deliberately not enabled so
            # this remains a clear, conventional DQN baseline.
            next_q = self._q_values_for_all_actions(
                self.target_network, next_states
            ).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=self.gradient_clip
        )
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
            logger.info("DQN_TARGET_UPDATED gradient_step=%d", self.gradient_steps)

        value = float(loss.item())
        logger.info(
            "DQN_UPDATED gradient_step=%d loss=%.8f q_mean=%.6f target_mean=%.6f",
            self.gradient_steps,
            value,
            float(predicted.detach().mean().item()),
            float(targets.mean().item()),
        )
        return value

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        replay = [
            {
                "state": item.state,
                "action": item.action,
                "reward": item.reward,
                "next_state": item.next_state,
                "done": item.done,
            }
            for item in self.replay_buffer
        ]
        torch.save(
            {
                "algo": "dqn",
                "version": self.CHECKPOINT_VERSION,
                "q_architecture": self.Q_ARCHITECTURE,
                "state_dim": self.state_dim,
                "arms": self.arms,
                "hidden_dim": self.hidden_dim,
                "action_feature_dim": self.action_feature_dim,
                "action_feature_schema": self.action_encoder.schema_dict(),
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay": replay,
                "decision_steps": self.decision_steps,
                "gradient_steps": self.gradient_steps,
                "transitions_seen": self.transitions_seen,
                "numpy_rng_state": self.rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "config": self.config_dict(),
            },
            temporary,
        )
        temporary.replace(destination)

    def load(self, path: str | Path, mode: str = "resume") -> None:
        """Load a checkpoint for exact resume or weights-only fine-tuning."""

        if mode not in ("resume", "finetune"):
            raise ValueError("DQN checkpoint load mode must be 'resume' or 'finetune'")
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:  # PyTorch versions before the weights_only argument.
            checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("algo") != "dqn":
            raise ValueError(f"incompatible checkpoint algo={checkpoint.get('algo')!r}")
        if (
            int(checkpoint.get("version", -1)) != self.CHECKPOINT_VERSION
            or checkpoint.get("q_architecture") != self.Q_ARCHITECTURE
        ):
            raise ValueError(
                "DQN checkpoint architecture is incompatible: expected "
                f"{self.Q_ARCHITECTURE!r} version {self.CHECKPOINT_VERSION}; "
                "legacy 72-output checkpoints cannot restore this network"
            )
        if int(checkpoint.get("state_dim", -1)) != self.state_dim:
            raise ValueError("DQN checkpoint state dimension does not match")
        if tuple(checkpoint.get("arms", ())) != self.arms:
            raise ValueError("DQN checkpoint action catalog does not match")
        if int(checkpoint.get("action_feature_dim", -1)) != self.action_feature_dim:
            raise ValueError("DQN checkpoint action feature dimension does not match")
        if (
            checkpoint.get("action_feature_schema")
            != self.action_encoder.schema_dict()
        ):
            raise ValueError("DQN checkpoint action feature schema does not match")

        self.online_network.load_state_dict(checkpoint["online_network"])
        if mode == "finetune":
            # The constructor owns the new optimizer and learning rate.  Do not
            # restore offline epsilon, replay, optimizer momentum, or RNG state.
            self.target_network.load_state_dict(self.online_network.state_dict())
            self.target_network.eval()
            self.replay_buffer.clear()
            self.decision_steps = 0
            self.gradient_steps = 0
            self.transitions_seen = 0
            logger.info(
                "DQN_CHECKPOINT_FINETUNE_LOADED path=%s lr=%s",
                path,
                self.learning_rate,
            )
            return

        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_network.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # ``optimizer.load_state_dict`` restores the checkpoint learning rate;
        # this is intentional only for exact resume mode.
        self.replay_buffer.clear()
        for item in checkpoint.get("replay", []):
            self.replay_buffer.append(
                Transition(
                    state=np.asarray(item["state"], dtype=np.float32),
                    action=int(item["action"]),
                    reward=float(item["reward"]),
                    next_state=np.asarray(item["next_state"], dtype=np.float32),
                    done=bool(item["done"]),
                )
            )
        self.decision_steps = int(checkpoint.get("decision_steps", 0))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        self.transitions_seen = int(
            checkpoint.get("transitions_seen", len(self.replay_buffer))
        )
        if checkpoint.get("numpy_rng_state"):
            self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        logger.info(
            "DQN_CHECKPOINT_LOADED path=%s replay=%d decisions=%d gradients=%d",
            path,
            len(self.replay_buffer),
            self.decision_steps,
            self.gradient_steps,
        )

    def config_dict(self) -> dict[str, float | int | str]:
        return {
            "q_architecture": self.Q_ARCHITECTURE,
            "state_dim": self.state_dim,
            "action_feature_dim": self.action_feature_dim,
            "network_input_dim": self.network_input_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_update_interval": self.target_update_interval,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "gradient_clip": self.gradient_clip,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
        }

    def _q_values_for_actions(
        self,
        network: QNetwork,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q(s, a) for one selected action per state."""

        action_features = self._action_features.index_select(0, actions)
        state_actions = torch.cat((states, action_features), dim=1)
        return network(state_actions).squeeze(1)

    def _q_values_for_all_actions(
        self,
        network: QNetwork,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate every catalog action for each state in one batched call."""

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                "state tensor must have shape "
                f"(batch, {self.state_dim}), got {tuple(states.shape)}"
            )
        batch_size = states.shape[0]
        expanded_states = states.unsqueeze(1).expand(
            batch_size, self.action_count, self.state_dim
        )
        expanded_actions = self._action_features.unsqueeze(0).expand(
            batch_size, self.action_count, self.action_feature_dim
        )
        state_actions = torch.cat((expanded_states, expanded_actions), dim=2)
        q_values = network(
            state_actions.reshape(batch_size * self.action_count, -1)
        )
        return q_values.reshape(batch_size, self.action_count)

    def _prepare_state(self, state: np.ndarray | Iterable[float]) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        if values.size != self.state_dim:
            raise ValueError(
                f"state dimension changed: expected {self.state_dim}, got {values.size}"
            )
        values = np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=0.0)
        # Current state layout: normalized lane growth values in [0,20], then
        # global fast-path ratio in [0,1].  Scale both into [0,1].
        if values.size > 1:
            values[:-1] = np.clip(values[:-1] / 20.0, 0.0, 1.0)
            values[-1] = np.clip(values[-1], 0.0, 1.0)
        else:
            values[0] = np.clip(values[0], 0.0, 1.0)
        return values
