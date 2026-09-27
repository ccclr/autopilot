from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from cmab import ArmCatalog
from controllers.action_transport import ActionBroadcaster
from actions.state_encode import build_dqn_state
from .policy import DQNPolicy

logger = logging.getLogger(__name__)


class DQNTrainer:
    """Central node0 loop using actual consecutive global-state transitions."""

    def __init__(
        self,
        *,
        metrics_dir: str,
        checkpoint_dir: str,
        arm_catalog: ArmCatalog,
        broadcaster: ActionBroadcaster,
        policy_kwargs: dict,
        metrics_timeout: int = 300,
        gradient_updates_per_transition: int = 1,
        resume_from: str | None = None,
        checkpoint_load_mode: str = "resume",
        checkpoint_prefix: str = "dqn_checkpoint",
    ) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.arm_catalog = arm_catalog
        self.arms = tuple(arm_catalog.list_arms())
        self.broadcaster = broadcaster
        self.policy_kwargs = dict(policy_kwargs)
        self.metrics_timeout = int(metrics_timeout)
        self.gradient_updates_per_transition = int(
            gradient_updates_per_transition
        )
        self.resume_from = resume_from
        if checkpoint_load_mode not in ("resume", "finetune"):
            raise ValueError(
                "DQN checkpoint load mode must be 'resume' or 'finetune'"
            )
        self.checkpoint_load_mode = checkpoint_load_mode
        self.checkpoint_prefix = checkpoint_prefix
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reward_history: deque[float] = deque(maxlen=20)
        self.policy: DQNPolicy | None = None
        self.training_active = True

    def run(self, num_iterations: Optional[int], checkpoint_freq: int) -> None:
        if checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")
        logger.info(
            "DQN_TRAINER_START endpoints=%s iterations=%s updates_per_transition=%d",
            [endpoint.format() for endpoint in self.broadcaster.endpoints],
            num_iterations,
            self.gradient_updates_per_transition,
        )
        last_metrics = self._get_latest_metrics_file()
        if last_metrics is None:
            last_metrics = self._wait_for_first_available_metrics_file(
                self.metrics_timeout
            )

        iteration = 0
        stored_since_start = 0
        while self.training_active:
            if num_iterations is not None and iteration >= num_iterations:
                logger.info("Reached maximum DQN iterations, stopping.")
                break

            state = self._build_state(last_metrics)
            if self.policy is None:
                self.policy = DQNPolicy(
                    state_dim=len(state), arms=self.arms, **self.policy_kwargs
                )
                if self.resume_from:
                    self.policy.load(
                        self.resume_from,
                        mode=self.checkpoint_load_mode,
                    )

            current_epoch = self._get_epoch(last_metrics)
            if current_epoch is None:
                logger.warning("Cannot determine epoch from %s", last_metrics)
                next_metrics = self._wait_for_new_metrics_file(
                    last_metrics, self.metrics_timeout
                )
                if next_metrics is not None:
                    last_metrics = next_metrics
                continue

            action_id, selection = self.policy.select_action(state)
            arm = self.arms[action_id]
            params = self.arm_catalog.decode_arm(arm)
            decision_id = (
                f"dqn-epoch-{current_epoch}-decision-{self.policy.decision_steps}"
                f"-action-{action_id}"
            )
            result = self.broadcaster.broadcast(
                decision_id=decision_id,
                signal_epoch=current_epoch,
                action_id=action_id,
                arm=arm,
                params=params,
            )
            logger.info(
                "DQN_DECISION epoch=%d decision=%s action_id=%d arm=%s params=%s "
                "mode=%s epsilon=%.6f broadcast_success=%s",
                current_epoch,
                decision_id,
                action_id,
                arm,
                params,
                selection["mode"],
                selection["epsilon"],
                result.success,
            )

            next_metrics = self._wait_for_new_metrics_file(
                last_metrics, self.metrics_timeout
            )
            if next_metrics is None:
                logger.warning(
                    "Timeout waiting for the next DQN global state; stopping "
                    "instead of issuing a second decision for the same epoch"
                )
                break

            next_epoch = self._get_epoch(next_metrics)
            reward = self._extract_reward(next_metrics)
            next_state = self._build_state(next_metrics)
            contiguous = next_epoch == current_epoch + 1
            local_abandon = Path(
                f"/tmp/autopilot_rl_param_abandon_{current_epoch}.signal"
            ).exists()
            valid_reward = math.isfinite(reward) and 0 < reward <= 15
            apply_ok = self._load_json(next_metrics).get("param_apply_ok") is True
            accepted = (
                result.success
                and apply_ok
                and contiguous
                and not local_abandon
                and valid_reward
            )

            if accepted:
                self.policy.observe(
                    state, action_id, reward, next_state, done=False
                )
                losses = self.policy.train(
                    self.gradient_updates_per_transition
                )
                stored_since_start += 1
                self.reward_history.append(reward)
                logger.info(
                    "DQN_TRANSITION_STORED source_epoch=%d reward_epoch=%d "
                    "action_id=%d reward=%.6f replay=%d losses=%s",
                    current_epoch,
                    next_epoch,
                    action_id,
                    reward,
                    len(self.policy.replay_buffer),
                    losses,
                )
                if stored_since_start % checkpoint_freq == 0:
                    checkpoint = self.checkpoint_dir / (
                        f"{self.checkpoint_prefix}_{self.policy.transitions_seen}.pt"
                    )
                    self.policy.save(checkpoint)
                    logger.info("DQN_CHECKPOINT_SAVED path=%s", checkpoint)
            else:
                logger.warning(
                    "DQN_TRANSITION_DROPPED source_epoch=%s reward_epoch=%s "
                    "broadcast_success=%s failed_nodes=%s contiguous=%s "
                    "local_abandon=%s reward=%.6f valid_reward=%s param_apply_ok=%s",
                    current_epoch,
                    next_epoch,
                    result.success,
                    result.failed_nodes,
                    contiguous,
                    local_abandon,
                    reward,
                    valid_reward,
                    apply_ok,
                )

            iteration += 1
            average_reward = (
                sum(self.reward_history) / len(self.reward_history)
                if self.reward_history
                else 0.0
            )
            logger.info(
                "DQN_ITERATION iteration=%d source_epoch=%d reward_epoch=%s "
                "reward=%.6f avg_reward_20=%.6f stored=%d",
                iteration,
                current_epoch,
                next_epoch,
                reward,
                average_reward,
                stored_since_start,
            )
            last_metrics = next_metrics

    def stop(self) -> None:
        self.training_active = False

    def _get_latest_metrics_file(self) -> Path | None:
        files = list(self.metrics_dir.glob("global_state_epoch_*.json"))
        candidates = [
            (epoch, path)
            for path in files
            if (epoch := self._get_epoch(path)) is not None
        ]
        return max(candidates, default=(None, None), key=lambda item: item[0])[1]

    def _wait_for_first_available_metrics_file(self, timeout: int) -> Path:
        start = time.time()
        while time.time() - start < timeout:
            latest = self._get_latest_metrics_file()
            if latest is not None:
                return latest
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for any global_state_epoch_*.json")

    def _wait_for_new_metrics_file(
        self, last_metrics: Path, timeout: int
    ) -> Path | None:
        current_epoch = self._get_epoch(last_metrics)
        if current_epoch is None:
            return None
        start = time.time()
        while time.time() - start < timeout:
            candidates = []
            for path in self.metrics_dir.glob("global_state_epoch_*.json"):
                epoch = self._get_epoch(path)
                if epoch is not None and epoch > current_epoch:
                    candidates.append((epoch, path))
            if candidates:
                newest_epoch, newest_path = max(candidates, key=lambda item: item[0])
                if newest_epoch > current_epoch + 1:
                    logger.warning(
                        "DQN_EPOCH_GAP current=%d newest=%d",
                        current_epoch,
                        newest_epoch,
                    )
                return newest_path
            time.sleep(0.1)
        return None

    @staticmethod
    def _get_epoch(path: Path) -> int | None:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError, AttributeError):
            return None

    @staticmethod
    def _load_json(path: Path) -> dict:
        for attempt in range(5):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    raw = handle.read()
                if not raw.strip():
                    raise json.JSONDecodeError("empty file", raw, 0)
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("global state root must be an object")
                return data
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
        raise ValueError(f"failed to read {path}")

    def _build_state(self, path: Path) -> np.ndarray:
        data = self._load_json(path)
        return build_dqn_state(data)

    def _extract_reward(self, path: Path) -> float:
        data = self._load_json(path)
        try:
            return float(data.get("global_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
