#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import ArmCatalog
from controllers.action_transport import ActionBroadcaster
from dqn import DQNPolicy, DQNTrainer


def main() -> None:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Centralized node0 DQN baseline for Autobahn"
    )
    parser.add_argument("--metrics-dir", required=True)
    # Kept for the common controller interface. Followers own parameter files.
    parser.add_argument("--parameters-file", required=True)
    parser.add_argument("--checkpoint-dir", default=str(home / "dqn_checkpoints"))
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=("resume", "finetune"),
        default="resume",
    )
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--action-endpoints", required=True)
    parser.add_argument("--action-timeout", type=float, default=2.0)
    parser.add_argument("--action-retries", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--replay-capacity", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=20)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200)
    parser.add_argument("--gradient-updates", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    if args.node_index != 0:
        parser.error("centralized DQN trainer must run on node 0")

    codec = ActionCodec(policy="rf_ts")
    arm_catalog = ArmCatalog(codec=codec)
    broadcaster = ActionBroadcaster.from_csv(
        args.action_endpoints,
        timeout=args.action_timeout,
        retries=args.action_retries,
    )
    policy_kwargs = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "target_update_interval": args.target_update_interval,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "gradient_clip": args.gradient_clip,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
    }
    logger.info(
        "DQN_CONFIG actions=%d q_architecture=%s policy=%s endpoints=%s "
        "warmup_ignored=%d",
        len(arm_catalog.list_arms()),
        DQNPolicy.Q_ARCHITECTURE,
        policy_kwargs,
        args.action_endpoints,
        args.warmup_iterations,
    )
    trainer = DQNTrainer(
        metrics_dir=args.metrics_dir,
        checkpoint_dir=args.checkpoint_dir,
        arm_catalog=arm_catalog,
        broadcaster=broadcaster,
        policy_kwargs=policy_kwargs,
        metrics_timeout=args.metrics_timeout,
        gradient_updates_per_transition=args.gradient_updates,
        resume_from=args.resume_from,
        checkpoint_load_mode=args.checkpoint_load_mode,
    )
    trainer.run(args.num_iterations, args.checkpoint_freq)


if __name__ == "__main__":
    main()
