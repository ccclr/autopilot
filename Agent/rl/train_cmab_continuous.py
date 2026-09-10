#!/usr/bin/env python3
"""
Continuous CMAB Training Script for Autopilot System.
"""

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import (
    ArmCatalog,
    CMABPolicy,
    CMABTrainer,
    ContextBuilder,
)
from offline_dataset import AsyncTransitionDatasetWriter

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
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Maximum iterations in this run; omit to train until stopped",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--policy", type=str, default="rf_ts", choices=["rf_ts", "random", "default"])
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
    parser.add_argument(
        "--enable-pairwise-residual-rf",
        action="store_true",
        help=(
            "Add a small cut_condition_type x fast_path_timeout RF trained "
            "on pre-update Global RF residuals"
        ),
    )
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
        "--enable-protocol-rules",
        action="store_true",
        help="Enable structured initialization and Autobahn-aware CMAB filters.",
    )
    parser.add_argument("--transition-export-dir", type=str, default=None)
    parser.add_argument("--environment-label", type=str, default="unlabeled")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    warmup_iterations = max(0, int(args.warmup_iterations))
    logger.info("Starting Autopilot Continuous CMAB Training")
    logger.info(
        "metrics_dir=%s parameters_file=%s warmup=%d max_iterations=%s "
        "protocol_rules=%s action_encoding=%s pairwise_residual_rf=%s",
        args.metrics_dir,
        args.parameters_file,
        warmup_iterations,
        args.num_iterations,
        args.enable_protocol_rules,
        args.action_encoding,
        args.enable_pairwise_residual_rf,
    )

    codec = ActionCodec(policy=args.policy)
    arm_catalog = ArmCatalog(codec=codec, max_arms=args.max_arms, seed=args.seed)
    arms = arm_catalog.list_arms()
    feature_dim = len(arm_catalog.decode_arm(arms[0])) if arms else 0
    policy = CMABPolicy(
        arms,
        feature_dim=feature_dim,
        policy_name=args.policy,
        epsilon=args.epsilon,
        random_state=args.seed,
        action_encoding=args.action_encoding,
        enable_pairwise_residual_rf=args.enable_pairwise_residual_rf,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded CMAB policy from %s", args.resume_from)

    context_builder = ContextBuilder(mode=args.context_mode)
    transition_writer = None
    if args.transition_export_dir and args.node_index == 0:
        try:
            transition_writer = AsyncTransitionDatasetWriter(
                root_dir=args.transition_export_dir,
                environment=args.environment_label,
                run_id=args.run_id or "",
                arms=arms,
                node_index=args.node_index,
                metadata={
                    "policy": args.policy,
                    "seed": args.seed,
                    "protocol_rules": args.enable_protocol_rules,
                    "warmup_iterations": warmup_iterations,
                    "action_encoding": args.action_encoding,
                    "pairwise_residual_rf": args.enable_pairwise_residual_rf,
                },
            )
            logger.info(
                "CMAB offline transitions: run=%s path=%s",
                transition_writer.run_id,
                transition_writer.transition_path,
            )
        except Exception:
            # Dataset collection is observational. A missing/unwritable export
            # directory must never prevent the original CMAB trainer from running.
            logger.exception(
                "CMAB_OFFLINE_TRANSITION_EXPORT_DISABLED during setup; "
                "CMAB will continue normally"
            )
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
        enable_protocol_rules=args.enable_protocol_rules,
        transition_writer=transition_writer,
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
