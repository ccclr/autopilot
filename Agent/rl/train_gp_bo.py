#!/usr/bin/env python3
"""
Continuous GP-BO Training Script for Autopilot System.

Mixed action space:
  - discrete: batch_size, header_size, cut_condition_type, k
  - continuous: fast_path_timeout_ms in ActionCodec.fast_path_timeout_ms_bounds
    (per-base UCB maximized with multi-start L-BFGS-B)

Reuses CMABTrainer for metrics wait / reward credit / parameter write.
"""

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import CMABTrainer, ContextBuilder
from gp_bo import GPBOPolicy, MixedActionSpace, MixedArmCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Continuous GP-BO Training for Autopilot")
    parser.add_argument("--metrics-dir", type=str, default=str(home / "autopilot" / "metrics"))
    parser.add_argument("--parameters-file", type=str, default=str(home / ".parameters.json"))
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/gp_bo_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--policy", type=str, default="gp_bo", choices=["gp_bo", "default"])
    parser.add_argument("--context-mode", type=str, default="dynamic", choices=["dynamic", "full"])
    parser.add_argument("--kappa", type=float, default=2.0, help="UCB exploration coefficient")
    parser.add_argument(
        "--timeout-grid-size",
        type=int,
        default=31,
        help="Multi-start count for continuous L-BFGS-B UCB max over fast_path_timeout",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help=(
            "Cold-start samples collected before first GP fit. "
            "Trainer-side update skipping is disabled for GP-BO to avoid double warmup."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument(
        "--enable-accelerator",
        action="store_true",
        help="Enable periodic latency probing to prune fast_path_timeout.",
    )
    parser.add_argument(
        "--accelerator-period",
        type=int,
        default=100,
        help="Epochs between master latency probes (apply 5 epochs later).",
    )
    parser.add_argument(
        "--accelerator-master",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="This process is the fab-remote host and runs latency probes.",
    )
    parser.add_argument("--enable-rtt-timeout-filter", action="store_true",
                        help="Keep fast_path_timeout strictly above the RTT covering grid bound.")
    args = parser.parse_args()

    warmup_iterations = max(0, int(args.warmup_iterations))
    logger.info("Starting Autopilot Continuous GP-BO Training (mixed timeout)")
    logger.info(
        "metrics_dir=%s parameters_file=%s kappa=%.3f timeout_restarts=%d warmup=%d",
        args.metrics_dir,
        args.parameters_file,
        args.kappa,
        args.timeout_grid_size,
        warmup_iterations,
    )

    codec_policy = "default" if args.policy == "default" else "rf_ts"
    codec = ActionCodec(policy=codec_policy)
    mixed_space = MixedActionSpace(codec=codec)
    arm_catalog = MixedArmCatalog(mixed_space)
    logger.info(
        "Mixed space: %d discrete bases, timeout in [%.1f, %.1f] ms",
        len(mixed_space.list_bases()),
        mixed_space.timeout_lo,
        mixed_space.timeout_hi,
    )

    # GP-BO warmup = collect N cold-start samples, then fit.
    # Keep trainer.warmup_iterations=0 so those samples are not discarded.
    policy = GPBOPolicy(
        feature_dim=5,
        policy_name="gp_bo",
        kappa=args.kappa,
        random_state=args.seed,
        mixed_space=mixed_space,
        timeout_grid_size=args.timeout_grid_size,
        min_samples_to_fit=warmup_iterations,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded GP-BO policy from %s", args.resume_from)

    context_builder = ContextBuilder(mode=args.context_mode)
    trainer = CMABTrainer(
        metrics_dir=args.metrics_dir,
        parameters_file=args.parameters_file,
        checkpoint_dir=args.checkpoint_dir,
        policy=policy,
        context_builder=context_builder,
        arm_catalog=arm_catalog,
        metrics_timeout=args.metrics_timeout,
        node_index=args.node_index,
        warmup_iterations=0,
        checkpoint_prefix="gp_bo_checkpoint",
        enable_accelerator=args.enable_accelerator,
        enable_rtt_timeout_filter=args.enable_rtt_timeout_filter,
        accelerator_period=args.accelerator_period,
        accelerator_master=args.accelerator_master,
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
