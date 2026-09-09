#!/usr/bin/env python3
"""
Continuous CMAB Training Script for Autopilot System.
"""

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import ArmCatalog, CMABPolicy, CMABTrainer, ContextBuilder, FactorizedCMABPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Continuous CMAB Training for Autopilot")
    parser.add_argument("--metrics-dir", type=str, default=str(home / "autopilot" / "metrics"))
    parser.add_argument("--parameters-file", type=str, default=str(home / ".parameters.json"))
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/cmab_continuous_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument(
        "--policy",
        type=str,
        default="rf_ts",
        choices=["rf_ts", "random", "default", "round_robin"],
    )
    parser.add_argument(
        "--action-encoding",
        type=str,
        default="numeric",
        choices=CMABPolicy.ACTION_ENCODINGS,
        help=(
            "CMAB-RF action features: legacy raw numeric parameters or one "
            "indicator per complete arm"
        ),
    )
    parser.add_argument("--context-mode", type=str, default="dynamic", choices=["dynamic", "full"])
    parser.add_argument("--epsilon", type=float, default=0, help="Epsilon-greedy exploration rate")
    parser.add_argument("--max-arms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Skip policy updates for the first N iterations (CMAB trainer warmup).",
    )
    parser.add_argument(
        "--enable-factorized-reward",
        action="store_true",
        help=(
            "Use the hierarchical factorized reward model instead of a "
            "single global-reward RF."
        ),
    )
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
    args = parser.parse_args()

    warmup_iterations = max(0, int(args.warmup_iterations))
    logger.info("Starting Autopilot Continuous CMAB Training")
    logger.info(
        "metrics_dir=%s parameters_file=%s warmup=%d action_encoding=%s "
        "factorized_reward=%s seed=%d",
        args.metrics_dir,
        args.parameters_file,
        warmup_iterations,
        args.action_encoding,
        args.enable_factorized_reward,
        args.seed,
    )

    codec = ActionCodec(policy=args.policy)
    arm_catalog = ArmCatalog(codec=codec, max_arms=args.max_arms, seed=args.seed)
    arms = arm_catalog.list_arms()
    feature_dim = len(arm_catalog.decode_arm(arms[0])) if arms else 0
    if args.enable_factorized_reward:
        policy = FactorizedCMABPolicy(
            arms,
            feature_dim=feature_dim,
            policy_name=args.policy,
            epsilon=args.epsilon,
            random_state=args.seed,
        )
        logger.info("Using FactorizedCMABPolicy (hierarchical main + pair RFs)")
    else:
        policy = CMABPolicy(
            arms,
            feature_dim=feature_dim,
            policy_name=args.policy,
            epsilon=args.epsilon,
            random_state=args.seed,
            action_encoding=args.action_encoding,
        )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded CMAB policy from %s", args.resume_from)

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
        warmup_iterations=warmup_iterations,
        enable_accelerator=args.enable_accelerator,
        accelerator_period=args.accelerator_period,
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()

