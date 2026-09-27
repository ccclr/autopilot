#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import ArmCatalog
from dqn import DQNPolicy
from offline_dataset import (
    BalancedTransitionSampler,
    dataset_summary,
    discover_transition_files,
    load_transition_files,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-train action-conditioned DQN from CMAB transitions"
    )
    parser.add_argument(
        "--dataset-root",
        default="/local/autopilot_offline_data",
        help="Root containing environment/run/transitions.jsonl datasets",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=None,
        help="Environment label to include; repeat for A/B/C (default: all)",
    )
    parser.add_argument("--gradient-steps", type=_positive_int, default=5000)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--target-update-interval", type=_positive_int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=_positive_int, default=100)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional prior offline DQN checkpoint for A->AB->ABC fine-tuning",
    )
    parser.add_argument(
        "--checkpoint-output",
        default=None,
        help="Output .pt path (default: /local/dqn_offline_checkpoints/...)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    paths = discover_transition_files(args.dataset_root, args.environment)
    if not paths:
        selected = args.environment or ["all environments"]
        parser.error(
            f"no transition files found under {args.dataset_root} for {selected}"
        )
    records = load_transition_files(paths)
    summary = dataset_summary(records)
    logger.info("OFFLINE_DATASET %s", json.dumps(summary, sort_keys=True))

    codec = ActionCodec(policy="rf_ts")
    arm_catalog = ArmCatalog(codec=codec)
    arms = tuple(arm_catalog.list_arms())
    arm_to_id = {arm: index for index, arm in enumerate(arms)}
    unknown_arms = sorted({record.arm for record in records}.difference(arm_to_id))
    if unknown_arms:
        raise ValueError(
            f"offline dataset contains arms outside the current catalog: {unknown_arms}"
        )

    policy = DQNPolicy(
        state_dim=int(summary["state_dim"]),
        arms=arms,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        replay_capacity=max(args.batch_size, 1),
        batch_size=args.batch_size,
        learning_starts=args.batch_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=0.20,
        epsilon_end=0.05,
        epsilon_decay_steps=120,
        gradient_clip=args.gradient_clip,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    if args.init_checkpoint:
        policy.load(args.init_checkpoint, mode="finetune")
        logger.info("OFFLINE_INIT_CHECKPOINT path=%s", args.init_checkpoint)

    sampler = BalancedTransitionSampler(records, seed=args.seed)
    losses: list[float] = []
    sampled_environments: Counter[str] = Counter()
    sampled_runs: Counter[str] = Counter()
    for step in range(1, args.gradient_steps + 1):
        sampled = sampler.sample(args.batch_size)
        batch = []
        for record in sampled:
            sampled_environments[record.environment] += 1
            sampled_runs[record.run_id] += 1
            batch.append(
                policy.prepare_transition(
                    state=record.state,
                    action=arm_to_id[record.arm],
                    reward=record.reward,
                    next_state=record.next_state,
                    # Time-limit truncation is not an MDP terminal state.
                    done=record.done,
                )
            )
        loss = policy.train_batch(batch)
        losses.append(loss)
        if step == 1 or step % args.log_interval == 0 or step == args.gradient_steps:
            window = losses[-min(args.log_interval, len(losses)) :]
            logger.info(
                "OFFLINE_STEP step=%d/%d mean_loss=%.8f sampled_env=%s",
                step,
                args.gradient_steps,
                sum(window) / len(window),
                dict(sampled_environments),
            )

    policy.transitions_seen = len(records)
    if args.checkpoint_output:
        output = Path(args.checkpoint_output).expanduser()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        labels = "-".join(args.environment or sorted(summary["environments"]))
        output = Path(
            "/local/dqn_offline_checkpoints/state_action_q_v1"
        ) / f"dqn_cmab_{labels}_{timestamp}.pt"
    policy.save(output)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(output),
        "init_checkpoint": args.init_checkpoint,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "transition_files": [str(path) for path in paths],
        "dataset_summary": summary,
        "gradient_steps": args.gradient_steps,
        "batch_size": args.batch_size,
        "balanced_sampling": "uniform environment -> run -> transition",
        "sampled_environments": dict(sampled_environments),
        "sampled_runs": dict(sampled_runs),
        "final_loss": losses[-1],
        "mean_loss_last_100": sum(losses[-100:]) / min(100, len(losses)),
        "policy_config": policy.config_dict(),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("OFFLINE_CHECKPOINT_SAVED path=%s metadata=%s", output, metadata_path)


if __name__ == "__main__":
    main()
