#!/usr/bin/env python3
"""
KernelUCB training for Autopilot.

Mixed action space:
  - discrete: batch_size, header_size, cut_condition_type, k
  - continuous: fast_path_timeout_ms (KernelUCB acquisition maximized
    continuously with multi-start L-BFGS-B from Uniform seeds)

Reuses CMABTrainer for metrics wait / reward credit / parameter write.
"""

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import CMABTrainer, ContextBuilder
from kernel_ucb import KernelUCBPolicy, MixedActionSpace, MixedArmCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="KernelUCB Training for Autopilot")
    parser.add_argument("--metrics-dir", type=str, default=str(home / "autopilot" / "metrics"))
    parser.add_argument("--parameters-file", type=str, default=str(home / ".parameters.json"))
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/kernel_ucb_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--policy", type=str, default="kernel_ucb", choices=["kernel_ucb", "default"])
    parser.add_argument("--context-mode", type=str, default="dynamic", choices=["dynamic", "full"])
    parser.add_argument("--beta", type=float, default=2.0, help="KernelUCB exploration coefficient")
    parser.add_argument("--kappa", type=float, default=None, help="Alias for --beta")
    parser.add_argument("--lambda-reg", type=float, default=1e-2, help="Kernel ridge regularization λ")
    parser.add_argument("--length-scale", type=float, default=0.4, help="RBF length scale on normalized features")
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=8,
        help="Random multi-starts for continuous timeout UCB maximization",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help=(
            "Cold-start selections before using KernelUCB. "
            "Also applied after --resume-from (post-resume exploration warmup)."
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

    beta = float(args.beta if args.kappa is None else args.kappa)
    warmup_iterations = max(0, int(args.warmup_iterations))
    logger.info("Starting Autopilot KernelUCB Training")
    logger.info(
        "metrics_dir=%s parameters_file=%s beta=%.3f n_restarts=%d warmup=%d",
        args.metrics_dir,
        args.parameters_file,
        beta,
        args.n_restarts,
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

    policy = KernelUCBPolicy(
        mixed_space=mixed_space,
        policy_name="kernel_ucb",
        beta=beta,
        lambda_reg=args.lambda_reg,
        length_scale=args.length_scale,
        random_state=args.seed,
        n_restarts=args.n_restarts,
        min_samples_to_fit=warmup_iterations,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded KernelUCB policy from %s", args.resume_from)
        # Checkpoint already has enough samples, so min_samples warmup is skipped.
        # Re-arm explicit cold-start warmup for post-resume exploration.
        if warmup_iterations > 0:
            policy.start_warmup(warmup_iterations)
            logger.info(
                "Post-resume warmup enabled: %d forced cold-start iteration(s)",
                warmup_iterations,
            )
    elif warmup_iterations > 0:
        # Fresh run: min_samples_to_fit already covers cold-start until fit;
        # also arm forced warmup so behavior stays consistent if min_samples is 0.
        policy.start_warmup(warmup_iterations)

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
        checkpoint_prefix="kernel_ucb_checkpoint",
        enable_accelerator=args.enable_accelerator,
        enable_rtt_timeout_filter=args.enable_rtt_timeout_filter,
        accelerator_period=args.accelerator_period,
        accelerator_master=args.accelerator_master,
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
