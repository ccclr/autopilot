#!/usr/bin/env python3
"""
Continuous XGBoost CMAB training for Autopilot.

Uses the same discrete arm catalog and trainer loop as CMAB-RF.
"""

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import ArmCatalog, CMABTrainer, ContextBuilder
from cmab.xgboost_policy import XGBoostPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(description="Continuous XGBoost Training for Autopilot")
    parser.add_argument("--metrics-dir", type=str, default=str(home / "autopilot" / "metrics"))
    parser.add_argument("--parameters-file", type=str, default=str(home / ".parameters.json"))
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/xgboost_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--policy", type=str, default="xgboost", choices=["xgboost", "random", "default"])
    parser.add_argument(
        "--action-encoding",
        type=str,
        default="numeric",
        choices=XGBoostPolicy.ACTION_ENCODINGS,
        help="Action features: raw numeric parameters or one indicator per arm",
    )
    parser.add_argument("--context-mode", type=str, default="dynamic", choices=["dynamic", "full"])
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Bootstrap committee size (same role as RF trees in CMAB-RF TS).",
    )
    parser.add_argument("--boosting-rounds", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-arms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Skip policy updates for the first N iterations.",
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
    parser.add_argument(
        "--accelerator-master",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="This process is the fab-remote host and runs latency probes.",
    )
    args = parser.parse_args()

    warmup_iterations = max(0, int(args.warmup_iterations))
    logger.info("Starting Autopilot Continuous XGBoost Training")
    logger.info(
        "metrics_dir=%s parameters_file=%s warmup=%d action_encoding=%s n_estimators=%d seed=%d",
        args.metrics_dir,
        args.parameters_file,
        warmup_iterations,
        args.action_encoding,
        args.n_estimators,
        args.seed,
    )

    codec_policy = "default" if args.policy == "default" else "rf_ts"
    codec = ActionCodec(policy=codec_policy)
    arm_catalog = ArmCatalog(codec=codec, max_arms=args.max_arms, seed=args.seed)
    arms = arm_catalog.list_arms()
    feature_dim = len(arm_catalog.decode_arm(arms[0])) if arms else 0
    policy = XGBoostPolicy(
        arms,
        feature_dim=feature_dim,
        policy_name=args.policy,
        n_estimators=args.n_estimators,
        boosting_rounds=args.boosting_rounds,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        action_encoding=args.action_encoding,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded XGBoost policy from %s", args.resume_from)

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
        checkpoint_prefix="xgboost_checkpoint",
        enable_accelerator=args.enable_accelerator,
        accelerator_period=args.accelerator_period,
        accelerator_master=args.accelerator_master,
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
