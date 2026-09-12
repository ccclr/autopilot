import time
import json
import threading
import subprocess
import socket
import struct
import logging
import os
import sys
import queue
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ControllerLogger:
    """Dedicated logger for controller events"""

    def __init__(self, log_dir: Optional[str] = None, node_index: Optional[int] = None):
        self.log_dir = Path(log_dir)
        self.node_index = node_index
        # Use node-indexed filename if node_index is provided
        filename = f"controller-{node_index}.log" if node_index is not None else "controller.log"
        self.controller_log_path = self.log_dir / filename
        self._buffer: list[str] = []
        self._buffer_size = 1  # Flush every event for immediate visibility
        self._ensure_log_dir()
        self._ensure_controller_log_file()

    def _ensure_log_dir(self):
        """Ensure log directory exists"""
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _ensure_controller_log_file(self):
        """Ensure controller log file exists"""
        try:
            if not self.controller_log_path.exists():
                self.controller_log_path.touch()
                print(f"Created controller log file: {self.controller_log_path}")
        except Exception as e:
            logger.warning(f"Failed to create controller log file: {e}")

    def log_event(self, event_type: str, details: Dict[str, Any], timestamp: Optional[str] = None):
        """Log a controller event"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + 'Z'

        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details
        }

        # Convert to JSON line
        json_line = json.dumps(event)
        self._buffer.append(json_line)

        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self):
        """Flush buffered events to file"""
        if not self._buffer:
            return

        try:
            self._ensure_controller_log_file()
            with open(self.controller_log_path, 'a', encoding='utf-8') as f:
                for line in self._buffer:
                    f.write(line + '\n')
            self._buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush controller log: {e}")

    def flush(self):
        """Force flush remaining events"""
        self._flush()

    def __del__(self):
        """Ensure buffer is flushed on destruction"""
        self.flush()


def get_controller_logger(node_index: Optional[int] = None, log_dir: Optional[str] = None) -> ControllerLogger:
    """Create a new controller logger instance"""
    return ControllerLogger(log_dir=log_dir, node_index=node_index)

class AutopilotController:
    """
    Autopilot system controller, responsible for applying parameters and retrieving metrics
    """

    def __init__(
        self,
        metrics_dir: str,
        parameters_file: str,
        node_index: Optional[int] = None,
        log_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        rl_algo: str = "cmab",
        warmup_iterations: int = 5,
        max_training_iterations: Optional[int] = None,
        kernel_ucb_alpha: float = 1.0,
        kernel_ucb_regularization: float = 0.1,
        kernel_ucb_length_scale: float = 1.0,
        kernel_ucb_timeout_min: float = 1.0,
        kernel_ucb_timeout_max: float = 300.0,
        kernel_ucb_optimizer_restarts: int = 5,
        kernel_ucb_replay_window: int = 200,
        dqn_action_endpoints: Optional[str] = None,
        dqn_action_timeout: float = 2.0,
        dqn_action_retries: int = 2,
        dqn_learning_rate: float = 1e-3,
        dqn_gamma: float = 0.90,
        dqn_replay_capacity: int = 2000,
        dqn_batch_size: int = 32,
        dqn_learning_starts: int = 32,
        dqn_target_update_interval: int = 20,
        dqn_epsilon_start: float = 1.0,
        dqn_epsilon_end: float = 0.05,
        dqn_epsilon_decay_steps: int = 200,
        dqn_gradient_updates: int = 1,
        dqn_gradient_clip: float = 10.0,
        dqn_hidden_dim: int = 64,
        dqn_seed: int = 0,
        dqn_checkpoint_load_mode: str = "resume",
        coverage_seed: int = 0,
        cmab_seed: int = 0,
        cmab_transition_export_dir: Optional[str] = None,
        cmab_environment_label: str = "unlabeled",
        cmab_transition_run_id: Optional[str] = None,
        enable_cmab_pairwise_residual_rf: bool = False,
    ):
        """
        Initialize controller

        Args:
            metrics_dir: metrics file directory
            parameters_file: parameter file path (.parameters.json)
            node_index: node index for logging
            log_dir: log directory
            resume_from: optional policy checkpoint path
            rl_algo: "cmab", "gp_bo", continuous-timeout "kernel_ucb",
                centralized node0 "dqn", or centralized data-collection-only
                "coverage_round_robin"
            warmup_iterations: unified warmup control passed to the training script.
                CMAB: skip policy updates for N iterations.
                GP-BO: collect N cold-start samples before first GP fit.
            max_training_iterations: maximum iterations in this trainer run, or None
                to continue until the controller is stopped.
        """
        self.metrics_dir = Path(metrics_dir)
        self.parameters_file = Path(parameters_file)
        self.node_index = node_index
        self.log_dir = Path(log_dir)
        self.resume_from = resume_from
        self.rl_algo = (rl_algo or "cmab").lower()
        self.enable_cmab_pairwise_residual_rf = bool(
            enable_cmab_pairwise_residual_rf
        )
        self.warmup_iterations = max(0, int(warmup_iterations))
        if max_training_iterations is not None and max_training_iterations <= 0:
            raise ValueError("max_training_iterations must be positive or None")
        self.max_training_iterations = max_training_iterations
        if self.rl_algo not in (
            "cmab",
            "gp_bo",
            "kernel_ucb",
            "dqn",
            "coverage_round_robin",
        ):
            raise ValueError(f"Unsupported rl_algo: {self.rl_algo}")
        self.kernel_ucb_alpha = float(kernel_ucb_alpha)
        self.kernel_ucb_regularization = float(kernel_ucb_regularization)
        self.kernel_ucb_length_scale = float(kernel_ucb_length_scale)
        self.kernel_ucb_timeout_min = float(kernel_ucb_timeout_min)
        self.kernel_ucb_timeout_max = float(kernel_ucb_timeout_max)
        self.kernel_ucb_optimizer_restarts = int(kernel_ucb_optimizer_restarts)
        self.kernel_ucb_replay_window = int(kernel_ucb_replay_window)
        self.dqn_action_endpoints = dqn_action_endpoints
        self.dqn_action_timeout = float(dqn_action_timeout)
        self.dqn_action_retries = int(dqn_action_retries)
        self.dqn_learning_rate = float(dqn_learning_rate)
        self.dqn_gamma = float(dqn_gamma)
        self.dqn_replay_capacity = int(dqn_replay_capacity)
        self.dqn_batch_size = int(dqn_batch_size)
        self.dqn_learning_starts = int(dqn_learning_starts)
        self.dqn_target_update_interval = int(dqn_target_update_interval)
        self.dqn_epsilon_start = float(dqn_epsilon_start)
        self.dqn_epsilon_end = float(dqn_epsilon_end)
        self.dqn_epsilon_decay_steps = int(dqn_epsilon_decay_steps)
        self.dqn_gradient_updates = int(dqn_gradient_updates)
        self.dqn_gradient_clip = float(dqn_gradient_clip)
        self.dqn_hidden_dim = int(dqn_hidden_dim)
        self.dqn_seed = int(dqn_seed)
        self.coverage_seed = int(coverage_seed)
        if self.coverage_seed < 0:
            raise ValueError("coverage_seed must be non-negative")
        self.cmab_seed = int(cmab_seed)
        if self.cmab_seed < 0:
            raise ValueError("cmab_seed must be non-negative")
        if dqn_checkpoint_load_mode not in ("resume", "finetune"):
            raise ValueError(
                "DQN checkpoint load mode must be 'resume' or 'finetune'"
            )
        self.dqn_checkpoint_load_mode = dqn_checkpoint_load_mode
        self.cmab_transition_export_dir = cmab_transition_export_dir
        self.cmab_environment_label = cmab_environment_label or "unlabeled"
        self.cmab_transition_run_id = cmab_transition_run_id
        if self.rl_algo in ("dqn", "coverage_round_robin"):
            if self.node_index != 0:
                raise ValueError(
                    f"Centralized {self.rl_algo} controller must run on node 0"
                )
            if not self.dqn_action_endpoints:
                raise ValueError(
                    f"{self.rl_algo} requires --dqn-action-endpoints"
                )
        if self.rl_algo == "coverage_round_robin":
            if self.resume_from:
                raise ValueError(
                    "coverage_round_robin does not load policy checkpoints"
                )
            if not self.cmab_transition_export_dir:
                raise ValueError(
                    "coverage_round_robin requires a transition export directory"
                )
            if not self.cmab_transition_run_id:
                raise ValueError("coverage_round_robin requires a transition run id")
        # Agent/rl root (parent of controllers/)
        self._rl_root = Path(__file__).resolve().parent.parent

        # Initialize controller logger
        self.logger = get_controller_logger(node_index=node_index, log_dir=log_dir)
        self.embedded_trainer = None
        self.training_process = None
        self.training_thread = None
        self.training_data_queue = None
        self.training_active = False
        self.training_initialized = False
        # Step counting for training frequency control (match env.py behavior)
        self.training_step_count = 0
        self.training_update_frequency = 1  # Update training every epoch/step

        # Ensure metrics directory exists
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self._start_continuous_training()

    def _is_training_active(self) -> bool:
        """Check if training is currently active"""
        return self.training_active and (self.training_process is not None or self.embedded_trainer is not None)

    def _set_training_inactive(self):
        """Set training as inactive"""
        self.training_active = False

    def _start_continuous_training(self):
        """Start embedded continuous RL training that runs throughout the Autopilot execution"""
        if self._is_training_active():
            logger.info("Continuous training already active")
            return

        logger.info("Starting continuous RL training (algo=%s)...", self.rl_algo)

        self._start_continuous_training_subprocess()

    def _training_script_name(self) -> str:
        if self.rl_algo == "gp_bo":
            return "train_gp_bo.py"
        if self.rl_algo == "kernel_ucb":
            return "train_kernel_ucb.py"
        if self.rl_algo == "dqn":
            return "train_dqn.py"
        if self.rl_algo == "coverage_round_robin":
            return "train_coverage_round_robin.py"
        return "train_cmab_continuous.py"

    def _checkpoint_dir(self) -> Path:
        home = Path.home()
        if self.rl_algo == "gp_bo":
            return home / "gp_bo_checkpoints"
        if self.rl_algo == "kernel_ucb":
            return home / "kernel_ucb_checkpoints"
        if self.rl_algo == "dqn":
            # CloudLab metrics live under /local/metrics-0, so keep the
            # centralized checkpoint outside the git-reset repository.  Keep
            # action-conditioned checkpoints separate from the incompatible
            # legacy 72-output-head checkpoints.
            return (
                self.metrics_dir.parent
                / "dqn_checkpoints"
                / "state_action_q_v1"
            )
        if self.rl_algo == "coverage_round_robin":
            # The common trainer interface accepts this argument, but coverage
            # collection has no model/checkpoint state and creates no folder.
            return self.metrics_dir.parent
        return home / "checkpoints"

    def _start_continuous_training_subprocess(self):
        logger.info("Starting continuous training subprocess (fallback mode)...")

        # Create training log file
        training_log_file = self.log_dir / f"continuous_training_{self.node_index}.log"
        logger.info(f"Continuous training output will be logged to: {training_log_file}")

        try:
            train_script = self._rl_root / self._training_script_name()
            checkpoint_dir = self._checkpoint_dir()
            if self.rl_algo != "coverage_round_robin":
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Start training script in continuous mode
            cmd = [
                sys.executable,
                str(train_script),
                "--node-index", str(self.node_index),
                "--metrics-dir", str(self.metrics_dir),
                "--parameters-file", str(self.parameters_file),
                "--checkpoint-dir", str(checkpoint_dir),
                "--warmup-iterations", str(self.warmup_iterations),
            ]
            if self.resume_from:
                cmd.extend(["--resume-from", str(self.resume_from)])
            if self.max_training_iterations is not None:
                cmd.extend(
                    ["--num-iterations", str(self.max_training_iterations)]
                )
            if (
                self.rl_algo == "cmab"
                and self.enable_cmab_pairwise_residual_rf
            ):
                cmd.append("--enable-pairwise-residual-rf")
            if self.rl_algo == "cmab":
                cmd.extend(
                    [
                        "--seed", str(self.cmab_seed),
                    ]
                )
            if self.rl_algo == "cmab" and self.cmab_transition_export_dir:
                cmd.extend(
                    [
                        "--transition-export-dir",
                        str(self.cmab_transition_export_dir),
                        "--environment-label",
                        str(self.cmab_environment_label),
                    ]
                )
                if self.cmab_transition_run_id:
                    cmd.extend(["--run-id", str(self.cmab_transition_run_id)])
            if self.rl_algo == "kernel_ucb":
                cmd.extend(
                    [
                        "--ucb-alpha", str(self.kernel_ucb_alpha),
                        "--regularization", str(self.kernel_ucb_regularization),
                        "--length-scale", str(self.kernel_ucb_length_scale),
                        "--timeout-min", str(self.kernel_ucb_timeout_min),
                        "--timeout-max", str(self.kernel_ucb_timeout_max),
                        "--optimizer-restarts", str(self.kernel_ucb_optimizer_restarts),
                        "--replay-window", str(self.kernel_ucb_replay_window),
                    ]
                )
            if self.rl_algo == "dqn":
                cmd.extend(
                    [
                        "--action-endpoints", str(self.dqn_action_endpoints),
                        "--action-timeout", str(self.dqn_action_timeout),
                        "--action-retries", str(self.dqn_action_retries),
                        "--learning-rate", str(self.dqn_learning_rate),
                        "--gamma", str(self.dqn_gamma),
                        "--replay-capacity", str(self.dqn_replay_capacity),
                        "--batch-size", str(self.dqn_batch_size),
                        "--learning-starts", str(self.dqn_learning_starts),
                        "--target-update-interval",
                        str(self.dqn_target_update_interval),
                        "--epsilon-start", str(self.dqn_epsilon_start),
                        "--epsilon-end", str(self.dqn_epsilon_end),
                        "--epsilon-decay-steps",
                        str(self.dqn_epsilon_decay_steps),
                        "--gradient-updates", str(self.dqn_gradient_updates),
                        "--gradient-clip", str(self.dqn_gradient_clip),
                        "--hidden-dim", str(self.dqn_hidden_dim),
                        "--seed", str(self.dqn_seed),
                        "--checkpoint-load-mode",
                        str(self.dqn_checkpoint_load_mode),
                    ]
                )
            if self.rl_algo == "coverage_round_robin":
                cmd.extend(
                    [
                        "--action-endpoints", str(self.dqn_action_endpoints),
                        "--action-timeout", str(self.dqn_action_timeout),
                        "--action-retries", str(self.dqn_action_retries),
                        "--transition-export-dir",
                        str(self.cmab_transition_export_dir),
                        "--environment-label",
                        str(self.cmab_environment_label),
                        "--seed", str(self.coverage_seed),
                    ]
                )
                if self.cmab_transition_run_id:
                    cmd.extend(["--run-id", str(self.cmab_transition_run_id)])

            logger.info(f"Starting continuous training with command: {' '.join(cmd)}")

            # Start training process (non-blocking, continuous)
            # Note: We don't use start_new_session=True to avoid zombie processes
            with open(training_log_file, 'a', buffering=1) as logfile:
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(self._rl_root),
                    # Don't use start_new_session=True - monitor thread will wait for it
                )

            logger.info(f"Continuous training subprocess started with PID: {self.training_process.pid}")
            self.training_active = True

            # Start background thread to monitor training process
            self.training_thread = threading.Thread(target=self._monitor_continuous_training, daemon=True)
            self.training_thread.start()

            logger.info("Continuous training subprocess started successfully")

        except Exception as e:
            logger.error(f"Failed to start continuous training subprocess: {e}")
            self.training_active = False

    def _monitor_continuous_training(self):
        """Monitor the continuous training process"""
        try:
            # Wait for process to complete (which should be never in continuous mode)
            return_code = self.training_process.wait()

            self._set_training_inactive()

            if return_code == 0:
                logger.info("  Continuous training completed normally")
            else:
                logger.error(f"  Continuous training failed with return code {return_code}")

        except Exception as e:
            logger.error(f"  Failed to monitor continuous training: {e}")
            self._set_training_inactive()

    def cleanup_training(self):
        """Cleanup training resources"""
        logger.info("Cleaning up controller training resources...")
        self.logger.log_event('controller_cleanup_start', {
            'has_embedded_trainer': self.embedded_trainer is not None,
            'has_training_process': self.training_process is not None
        })

        self.training_active = False

        if self.embedded_trainer:
            try:
                self.embedded_trainer.cleanup()
                logger.info("Embedded trainer cleaned up")
                self.logger.log_event('embedded_trainer_cleanup', {'status': 'success'})
            except Exception as e:
                logger.warning(f"Failed to cleanup embedded trainer: {e}")
                self.logger.log_event('embedded_trainer_cleanup', {
                    'status': 'failed',
                    'error': str(e)
                })

        if self.training_process:
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
                logger.info("Training subprocess terminated")
                self.logger.log_event('training_process_cleanup', {'status': 'success'})
            except Exception as e:
                logger.warning(f"Failed to terminate training subprocess: {e}")
                self.logger.log_event('training_process_cleanup', {
                    'status': 'failed',
                    'error': str(e)
                })
                try:
                    self.training_process.kill()
                except:
                    pass

    def __del__(self):
        """Destructor - cleanup resources"""
        self.cleanup_training()


def main():
    """Main function to run controller as standalone server"""
    import argparse
    import signal
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autopilot RL Controller Server')
    parser.add_argument('--metrics-dir', type=str, required=True,
                       help='Directory for metrics files')
    parser.add_argument('--parameters-file', type=str, required=True,
                       help='Path to parameters JSON file')
    parser.add_argument('--slots-per-epoch', type=int, default=20,
                       help='Number of slots per epoch')
    parser.add_argument('--node-index', type=int, default=None,
                       help='Node index for logging')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Log directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume RL policy from checkpoint path')
    parser.add_argument('--rl-algo', type=str, default='cmab',
                       choices=['cmab', 'gp_bo', 'kernel_ucb', 'dqn',
                                'coverage_round_robin'],
                       help=('RL algorithm: cmab, gp_bo, continuous kernel_ucb, '
                             'centralized dqn, or coverage_round_robin'))
    parser.add_argument(
        '--cmab-seed',
        type=int,
        default=0,
        help='CMAB random-forest seed (default: 0)',
    )
    parser.add_argument(
        '--enable-cmab-pairwise-residual-rf',
        action='store_true',
        help=(
            'Enable the cut_condition_type x fast_path_timeout residual RF '
            'on top of the standard CMAB Global RF'
        ),
    )
    parser.add_argument(
        '--warmup-iterations',
        type=int,
        default=5,
        help=(
            'Unified warmup: CMAB skips updates for N iters; '
            'GP-BO collects N cold-start samples before fit'
        ),
    )
    parser.add_argument(
        '--max-training-iterations',
        type=int,
        default=None,
        help='Maximum iterations in this trainer run; omit to train until stopped',
    )
    parser.add_argument('--kernel-ucb-alpha', type=float, default=1.0)
    parser.add_argument('--kernel-ucb-regularization', type=float, default=0.1)
    parser.add_argument('--kernel-ucb-length-scale', type=float, default=1.0)
    parser.add_argument('--kernel-ucb-timeout-min', type=float, default=1.0)
    parser.add_argument('--kernel-ucb-timeout-max', type=float, default=300.0)
    parser.add_argument('--kernel-ucb-optimizer-restarts', type=int, default=5)
    parser.add_argument('--kernel-ucb-replay-window', type=int, default=200)
    parser.add_argument('--dqn-action-endpoints', type=str, default=None)
    parser.add_argument('--dqn-action-timeout', type=float, default=2.0)
    parser.add_argument('--dqn-action-retries', type=int, default=2)
    parser.add_argument('--dqn-learning-rate', type=float, default=1e-3)
    parser.add_argument('--dqn-gamma', type=float, default=0.90)
    parser.add_argument('--dqn-replay-capacity', type=int, default=2000)
    parser.add_argument('--dqn-batch-size', type=int, default=32)
    parser.add_argument('--dqn-learning-starts', type=int, default=32)
    parser.add_argument('--dqn-target-update-interval', type=int, default=20)
    parser.add_argument('--dqn-epsilon-start', type=float, default=1.0)
    parser.add_argument('--dqn-epsilon-end', type=float, default=0.05)
    parser.add_argument('--dqn-epsilon-decay-steps', type=int, default=200)
    parser.add_argument('--dqn-gradient-updates', type=int, default=1)
    parser.add_argument('--dqn-gradient-clip', type=float, default=10.0)
    parser.add_argument('--dqn-hidden-dim', type=int, default=64)
    parser.add_argument('--dqn-seed', type=int, default=0)
    parser.add_argument(
        '--coverage-seed',
        type=int,
        default=0,
        help='Seed for shuffled 96-action coverage cycles',
    )
    parser.add_argument(
        '--dqn-checkpoint-load-mode',
        choices=['resume', 'finetune'],
        default='resume',
    )
    parser.add_argument('--cmab-transition-export-dir', type=str, default=None)
    parser.add_argument('--cmab-environment-label', type=str, default='unlabeled')
    parser.add_argument('--cmab-transition-run-id', type=str, default=None)

    args = parser.parse_args()

    print("🚀 Starting Autopilot RL Controller Server")
    print(f"📊 Metrics dir: {args.metrics_dir}")
    print(f"⚙️  Parameters file: {args.parameters_file}")
    print(f"🏷️  Node index: {args.node_index}")
    print(f"📝 Log dir: {args.log_dir}")
    print(f"🧠 RL algo: {args.rl_algo}")
    print(f"🎲 CMAB random-forest seed: {args.cmab_seed}")
    print(f"🧩 CMAB pairwise residual RF: {args.enable_cmab_pairwise_residual_rf}")
    print(f"🔁 Resume from: {args.resume_from}")
    print(f"🔥 Warmup iterations: {args.warmup_iterations}")
    max_iterations = (
        args.max_training_iterations
        if args.max_training_iterations is not None
        else 'continuous'
    )
    print(f"🔢 Max training iterations: {max_iterations}")
    print(f"💾 CMAB transition export: {args.cmab_transition_export_dir}")
    print(f"🌐 CMAB environment label: {args.cmab_environment_label}")
    print(f"🧪 DQN checkpoint load mode: {args.dqn_checkpoint_load_mode}")
    print(f"🎲 Coverage round-robin seed: {args.coverage_seed}")

    # Logger will be initialized by AutopilotController

    # Initialize controller
    try:
        controller = AutopilotController(
            metrics_dir=args.metrics_dir,
            parameters_file=args.parameters_file,
            node_index=args.node_index,
            log_dir=args.log_dir,
            resume_from=args.resume_from,
            rl_algo=args.rl_algo,
            cmab_seed=args.cmab_seed,
            enable_cmab_pairwise_residual_rf=(
                args.enable_cmab_pairwise_residual_rf
            ),
            warmup_iterations=args.warmup_iterations,
            max_training_iterations=args.max_training_iterations,
            kernel_ucb_alpha=args.kernel_ucb_alpha,
            kernel_ucb_regularization=args.kernel_ucb_regularization,
            kernel_ucb_length_scale=args.kernel_ucb_length_scale,
            kernel_ucb_timeout_min=args.kernel_ucb_timeout_min,
            kernel_ucb_timeout_max=args.kernel_ucb_timeout_max,
            kernel_ucb_optimizer_restarts=args.kernel_ucb_optimizer_restarts,
            kernel_ucb_replay_window=args.kernel_ucb_replay_window,
            dqn_action_endpoints=args.dqn_action_endpoints,
            dqn_action_timeout=args.dqn_action_timeout,
            dqn_action_retries=args.dqn_action_retries,
            dqn_learning_rate=args.dqn_learning_rate,
            dqn_gamma=args.dqn_gamma,
            dqn_replay_capacity=args.dqn_replay_capacity,
            dqn_batch_size=args.dqn_batch_size,
            dqn_learning_starts=args.dqn_learning_starts,
            dqn_target_update_interval=args.dqn_target_update_interval,
            dqn_epsilon_start=args.dqn_epsilon_start,
            dqn_epsilon_end=args.dqn_epsilon_end,
            dqn_epsilon_decay_steps=args.dqn_epsilon_decay_steps,
            dqn_gradient_updates=args.dqn_gradient_updates,
            dqn_gradient_clip=args.dqn_gradient_clip,
            dqn_hidden_dim=args.dqn_hidden_dim,
            dqn_seed=args.dqn_seed,
            dqn_checkpoint_load_mode=args.dqn_checkpoint_load_mode,
            coverage_seed=args.coverage_seed,
            cmab_transition_export_dir=args.cmab_transition_export_dir,
            cmab_environment_label=args.cmab_environment_label,
            cmab_transition_run_id=args.cmab_transition_run_id,
        )
        print("✅ Controller initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize controller: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Signal handler for graceful shutdown
    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n🛑 Shutting down controller server...")
        shutdown_requested = True
        controller.cleanup_training()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep server running
    print("🔄 Controller server running (press Ctrl+C to stop)...")
    try:
        while not shutdown_requested:
            time.sleep(1)  # Keep alive
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down...")
    finally:
        print("🛑 Cleaning up controller resources...")
        controller.cleanup_training()
        print("✅ Controller server stopped.")


if __name__ == "__main__":
    main()
