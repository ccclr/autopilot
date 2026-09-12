from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .arm_catalog import ArmCatalog
from .context_builder import ContextBuilder
from actions.state_encode import build_dqn_state
from offline_dataset import AsyncTransitionDatasetWriter

logger = logging.getLogger(__name__)


class CMABTrainer:
    def __init__(
        self,
        metrics_dir: str,
        parameters_file: str,
        checkpoint_dir: str,
        policy: Any,
        context_builder: ContextBuilder,
        arm_catalog: ArmCatalog,
        metrics_timeout: int = 300,
        node_index: Optional[int] = None,
        warmup_iterations: int = 5,
        checkpoint_prefix: str = "cmab_checkpoint",
        transition_writer: Optional[AsyncTransitionDatasetWriter] = None,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.parameters_file = Path(parameters_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy = policy
        self.context_builder = context_builder
        self.arm_catalog = arm_catalog
        self.metrics_timeout = metrics_timeout

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.last_metrics_file: Optional[Path] = None
        self.training_active = True
        self.reward_history = deque(maxlen=20)
        self.arm_counts = {}
        self._param_socket: Optional[socket.socket] = None
        self.node_index = node_index
        self.warmup_iterations = max(0, warmup_iterations)
        self.checkpoint_prefix = checkpoint_prefix or "cmab_checkpoint"
        self.transition_writer = transition_writer

    def run(self, num_iterations: Optional[int], checkpoint_freq: int):
        logger.info("Initializing CMAB training loop...")
        logger.info("Policy: %s", self.policy.policy_name)
        logger.info("Iterations: %s", num_iterations)
        logger.info(
            "Warmup iterations (skip policy update): %s",
            self.warmup_iterations,
        )
        self._connect_param_socket()
        self.last_metrics_file = self._get_latest_metrics_file()
        if self.last_metrics_file is None:
            logger.info("No existing metrics file, waiting for first available global_state...")
            self.last_metrics_file = self._wait_for_first_available_metrics_file(
                timeout=self.metrics_timeout
            )

        iteration = 0
        last_arm = self._load_initial_arm_from_parameters_file()
        if last_arm is not None:
            logger.info("Bootstrapped initial arm from parameters file: %s", last_arm)
        action_by_epoch: dict[int, str] = {}
        while self.training_active:
            if num_iterations is not None and iteration >= num_iterations:
                logger.info("Reached maximum iterations, stopping.")
                break

            logger.info("FEATURIZE_START metrics=%s", self.last_metrics_file.name)
            context = self._build_context_from_global_state(self.last_metrics_file)
            logger.info("FEATURIZE_DONE context_dim=%d", len(context))
            current_epoch = self._get_epoch_from_metrics_file(self.last_metrics_file)
            shared_seed_hex = self._compute_shared_seed_hex(self.last_metrics_file)
            arm = self.policy.select_arm(context, shared_seed_hex=shared_seed_hex)
            params = self.arm_catalog.decode_arm(arm)
            self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
            if current_epoch is not None:
                # Action selected from state_t is credited to reward from state_{t+1}.
                action_by_epoch[current_epoch + 1] = arm

            self._write_parameters_to_file(params, current_epoch)
            logger.info("Applied params: %s", params)

            next_metrics = self._wait_for_new_metrics_file(self.last_metrics_file, timeout=self.metrics_timeout)
            if next_metrics is None:
                logger.warning("Timeout waiting for new metrics file, skipping update.")
                continue

            # Load once for the original CMAB reward path and optional passive
            # export. Export must not add another metrics-file read.
            next_metrics_data = self._load_json_with_retry(next_metrics)
            reward = self._extract_reward_from_data(
                next_metrics_data,
                next_metrics,
            )
            reward_epoch = self._get_epoch_from_metrics_file(next_metrics)
            if reward > 15 or reward == 0:
                logger.warning(
                    "Dropping sample due to suspicious high reward (>10): reward=%.6f metrics=%s",
                    reward,
                    next_metrics.name,
                )
                # Move forward to avoid reprocessing the same metrics file.
                self.last_metrics_file = next_metrics
                continue
            if current_epoch is not None and reward_epoch is not None and reward_epoch > current_epoch + 1:
                logger.warning(
                    "Detected non-contiguous reward epoch: current=%s, reward=%s, backfilling credited arm for skipped epochs",
                    current_epoch,
                    reward_epoch,
                )
                # state_t selects action_{t+1}. action_{current+1} is already set above.
                # For jumps, only backfill truly skipped epochs: [current+2, reward_epoch].
                for epoch in range(current_epoch + 2, reward_epoch + 1):
                    action_by_epoch.setdefault(epoch, arm)
            # Credit priority:
            # 1) abandon signal => previous epoch action
            # 2) direct epoch mapping => action selected for reward_epoch
            # 3) fallback => current selected arm
            use_arm = action_by_epoch.get(reward_epoch) if reward_epoch is not None else None
            abandon_signal = None
            abandoned = False
            if reward_epoch is not None:
                abandon_signal = Path(f"/tmp/autopilot_rl_param_abandon_{reward_epoch-1}.signal")
                if abandon_signal.exists():
                    abandoned = True
                    logger.warning(
                        "Detected abandon signal for epoch %s, using previous iteration action for update",
                        reward_epoch,
                    )
                    use_arm = last_arm

            if use_arm is None:
                # Safety fallback when previous action is unavailable.
                use_arm = arm

            in_warmup = iteration < self.warmup_iterations
            if not in_warmup:
                update_contexts = [context] if self.policy.uses_context else None
                self.policy.update(
                    [use_arm],
                    [reward],
                    update_contexts,
                    shared_seed_hex=shared_seed_hex,
                )
            else:
                logger.info(
                    "Warmup iteration %s/%s: skip policy update (reward=%.6f, arm=%s)",
                    iteration + 1,
                    self.warmup_iterations,
                    reward,
                    use_arm,
                )
            logger.info(
                "Iteration %s : context=%s arm=%s params=%s",
                iteration + 1,
                context,
                use_arm,
                self.arm_catalog.decode_arm(use_arm),
            )

            iteration += 1
            self.reward_history.append(reward)

            if reward_epoch is not None:
                abandon_signal = Path(f"/tmp/autopilot_rl_param_abandon_{reward_epoch-1}.signal")
                if not abandon_signal.exists():
                    last_arm = use_arm
            
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            top_arm = max(self.arm_counts, key=self.arm_counts.get)
            top_ratio = self.arm_counts[top_arm] / max(1, sum(self.arm_counts.values()))
            logger.info(
                "Iteration %s reward=%.6f avg_reward(20)=%.6f last_metrics=%s top_arm=%s top_ratio=%.2f",
                iteration,
                reward,
                avg_reward,
                next_metrics.name,
                top_arm,
                top_ratio,
            )

            if iteration % checkpoint_freq == 0:
                checkpoint_path = (
                    self.checkpoint_dir / f"{self.checkpoint_prefix}_{iteration}.pkl"
                )
                self.policy.save(str(checkpoint_path))
                logger.info("Saved checkpoint: %s", checkpoint_path)

            self.last_metrics_file = next_metrics
            # Export is deliberately last and fail-open: all CMAB state,
            # checkpoint work for this iteration is complete.
            self._export_transition_best_effort(
                current_epoch=current_epoch,
                reward_epoch=reward_epoch,
                context=context,
                arm=use_arm,
                reward=reward,
                next_metrics_data=next_metrics_data,
                abandoned=abandoned,
            )

        if self.transition_writer is not None:
            self._close_transition_writer()

    def stop(self):
        self.training_active = False
        if self.transition_writer is not None:
            self._close_transition_writer()
        if self._param_socket is not None:
            try:
                self._param_socket.close()
            except OSError:
                pass
            self._param_socket = None

    def _export_transition_best_effort(
        self,
        *,
        current_epoch: Optional[int],
        reward_epoch: Optional[int],
        context: np.ndarray,
        arm: str,
        reward: float,
        next_metrics_data: dict,
        abandoned: bool,
    ) -> None:
        writer = self.transition_writer
        if writer is None:
            return
        contiguous = (
            current_epoch is not None
            and reward_epoch == current_epoch + 1
        )
        valid_reward = np.isfinite(reward) and 0 < reward <= 15
        if not contiguous or abandoned or not valid_reward:
            logger.warning(
                "CMAB_OFFLINE_TRANSITION_DROPPED source_epoch=%s "
                "reward_epoch=%s contiguous=%s abandoned=%s reward=%s",
                current_epoch,
                reward_epoch,
                contiguous,
                abandoned,
                reward,
            )
            return
        try:
            next_state = self._build_context_from_data(next_metrics_data)
            writer.write(
                source_epoch=current_epoch,
                reward_epoch=reward_epoch,
                state=context,
                arm=arm,
                reward=reward,
                next_state=next_state,
                done=False,
                truncated=False,
            )
        except Exception:
            # This also protects CMAB if a future writer implementation stops
            # being asynchronous or validates records on the caller thread.
            logger.exception(
                "CMAB_OFFLINE_TRANSITION_EXPORT_DISABLED run=%s; "
                "CMAB will continue normally",
                getattr(writer, "run_id", "unknown"),
            )
            self._close_transition_writer()

    def _close_transition_writer(self) -> None:
        writer = self.transition_writer
        self.transition_writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception:
            logger.exception(
                "Failed to close CMAB transition writer; CMAB will continue"
            )

    def _get_latest_metrics_file(self) -> Optional[Path]:
        files = list(self.metrics_dir.glob("global_state_epoch_*.json"))
        if not files:
            return None

        def parse_file_key(file_path: Path):
            try:
                epoch = int(file_path.stem.split("_")[-1])
                return (epoch,)
            except (ValueError, IndexError, AttributeError):
                return (-1,)

        files.sort(key=parse_file_key, reverse=True)
        return files[0]

    def _wait_for_new_metrics_file(self, last_metrics_file: Path, timeout: int) -> Optional[Path]:
        logger.info("Waiting for newer metrics file (latest-wins)...")
        start_time = time.time()

        current_epoch = self._get_epoch_from_metrics_file(last_metrics_file)
        if current_epoch is None:
            current_epoch = self._get_max_epoch()

        while time.time() - start_time < timeout:
            newer_files = []
            for file_path in self.metrics_dir.glob("global_state_epoch_*.json"):
                epoch = self._get_epoch_from_metrics_file(file_path)
                if epoch is None:
                    continue
                if epoch > current_epoch:
                    newer_files.append((epoch, file_path))

            if newer_files:
                newer_files.sort(key=lambda item: item[0])
                newest_epoch, newest_file = newer_files[-1]
                if newest_epoch > current_epoch + 1:
                    logger.warning(
                        "Detected epoch gap: current=%s, newest=%s, skipping missing epochs",
                        current_epoch,
                        newest_epoch,
                    )
                return newest_file
            time.sleep(0.1)
        return None

    def _wait_for_epoch(self, epoch: int, timeout: int) -> str:
        start = time.time()
        pattern = f"global_state_epoch_{epoch}.json"
        while time.time() - start < timeout:
            files = list(self.metrics_dir.glob(pattern))
            if files:
                return str(files[0])
            time.sleep(0.1)
        raise TimeoutError(f"Timeout waiting for epoch {epoch}")

    def _wait_for_first_available_metrics_file(self, timeout: int) -> Path:
        start = time.time()
        while time.time() - start < timeout:
            latest = self._get_latest_metrics_file()
            if latest is not None:
                epoch = self._get_epoch_from_metrics_file(latest)
                if epoch is not None and epoch > 0:
                    logger.warning(
                        "First observed global state is epoch %s (epoch 0 missing), continuing with latest-wins",
                        epoch,
                    )
                return latest
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for any global_state_epoch_*.json")

    def _load_initial_arm_from_parameters_file(self) -> Optional[str]:
        if not self.parameters_file.exists():
            return None
        try:
            with open(self.parameters_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = ("batch_size", "header_size", "cut_condition_type", "fast_path_timeout", "k")
            if not all(k in data for k in keys):
                return None
            parts = []
            for k in keys:
                # if k == "use_optimistic_tips":
                #     parts.append(f"{k}={1 if data[k] else 0}")
                # else:
                parts.append(f"{k}={int(data[k])}")
            arm = ",".join(parts)
            self.arm_catalog.decode_arm(arm)
            return arm
        except Exception as e:
            logger.warning(
                "Failed to bootstrap arm from parameters file %s: %s",
                self.parameters_file,
                e,
            )
            return None

    def _get_max_epoch(self) -> int:
        files = list(self.metrics_dir.glob("global_state_epoch_*.json"))
        max_epoch = -1
        for file_path in files:
            try:
                epoch_str = file_path.stem.split("_")[-1]
                epoch_num = int(epoch_str)
                max_epoch = max(max_epoch, epoch_num)
            except (ValueError, IndexError, AttributeError):
                continue
        return max_epoch

    def _get_epoch_from_metrics_file(self, metrics_file: Path) -> Optional[int]:
        try:
            return int(metrics_file.stem.split("_")[-1])
        except (ValueError, IndexError, AttributeError):
            return None

    def _load_json_with_retry(self, json_path: Path) -> dict:
        data = None
        for attempt in range(5):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                if not raw.strip():
                    raise json.JSONDecodeError("Empty file", raw, 0)
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
        if data is None:
            raise ValueError(f"Failed to parse metrics file: {json_path}")
        return data

    def _build_context_from_global_state(self, metrics_path: Path) -> np.ndarray:
        data = self._load_json_with_retry(metrics_path)
        return self._build_context_from_data(data)

    def _build_context_from_data(self, data: dict) -> np.ndarray:
        dynamic_state = build_dqn_state(data)
        if self.context_builder.mode == "dynamic":
            return dynamic_state
        # Full mode historically concatenates context + dynamic; context is empty in current setup.
        return dynamic_state

    def _extract_reward_from_global_state(self, metrics_path: Path) -> float:
        data = self._load_json_with_retry(metrics_path)
        return self._extract_reward_from_data(data, metrics_path)

    def _extract_reward_from_data(self, data: dict, metrics_path: Path) -> float:
        reward = data.get("global_reward")
        if reward is None:
            logger.warning("global_reward missing in %s, defaulting to 0.0", metrics_path)
            return 0.0
        try:
            return float(reward)
        except (TypeError, ValueError):
            logger.warning("Invalid global_reward in %s: %s", metrics_path, reward)
            return 0.0

    def _compute_shared_seed_hex(self, metrics_path: Path) -> str:
        """
        seed_t = sha256(global_state_t || block_height || commit_hash)
        """
        data = self._load_json_with_retry(metrics_path)
        global_state_t = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        block_height = str(data.get("epoch", -1))
        commit_hash = str(data.get("cc_digest", ""))
        payload = f"{global_state_t}|{block_height}|{commit_hash}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _write_parameters_to_file(self, params, epoch: Optional[int]):
        import fcntl
        import tempfile
        import shutil

        params_file = self.parameters_file
        params_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                with open(params_file, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    current_params = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (FileNotFoundError, json.JSONDecodeError):
                current_params = {}

            current_params.update({
                "batch_size": params["batch_size"],
                "header_size": params["header_size"],
                "cut_condition_type": params["cut_condition_type"],
                "fast_path_timeout": params["fast_path_timeout"],
                "k": params["k"],
                # "use_optimistic_tips": bool(params["use_optimistic_tips"]),
            })

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=os.path.dirname(params_file)) as temp_file:
                json.dump(current_params, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            shutil.move(temp_file.name, params_file)
            logger.info("Updated parameters file: %s", params_file)

            self._send_param_update_socket(epoch)

        except Exception as e:
            logger.error("Failed to write parameters to file: %s", e)
            if "temp_file" in locals():
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


    def _get_param_socket_path(self) -> str:
        node_index = self.node_index
        return f"/tmp/autopilot_rl_param_{node_index}.sock"

    def _send_param_update_socket(self, epoch: Optional[int]) -> bool:
        socket_path = self._get_param_socket_path()
        payload = json.dumps({"epoch": epoch})
        for attempt in range(1, 11):
            try:
                if self._param_socket is None:
                    self._connect_param_socket()
                self._param_socket.sendall((payload + "\n").encode("utf-8"))
                logger.info("🚩 Sent RL param update via socket: %s", socket_path)
                return True
            except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                if self._param_socket is not None:
                    try:
                        self._param_socket.close()
                    except OSError:
                        pass
                    self._param_socket = None
                if attempt < 10:
                    time.sleep(min(0.2 * attempt, 2.0))
                    continue
                logger.warning("Failed to send RL param update via socket %s: %s", socket_path, e)
                return False

    def _connect_param_socket(self) -> None:
        if self._param_socket is not None:
            return
        socket_path = self._get_param_socket_path()
        max_retries = 20
        retry_delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "🔌 Attempting to connect RL param socket: %s (attempt %s/%s)",
                    socket_path,
                    attempt,
                    max_retries,
                )
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(0.5)
                client.connect(socket_path)
                self._param_socket = client
                logger.info("🔌 Connected RL param socket: %s", socket_path)
                return
            except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                try:
                    client.close()
                except OSError:
                    pass
                if attempt < max_retries:
                    logger.warning(
                        "⚠️  RL param socket connect attempt %s failed: %s, retrying in %.1fs...",
                        attempt,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay = 2.0
                    continue
                logger.warning("Failed to connect RL param socket %s after %s attempts: %s", socket_path, max_retries, e)
                return
