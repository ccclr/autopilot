from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from actions.state_encode import DQN_STATE_SCHEMA


TRANSITION_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _safe_component(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-.")
    return text or fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OfflineTransition:
    environment: str
    run_id: str
    source_epoch: int
    reward_epoch: int
    state: np.ndarray
    arm: str
    action_id: int
    reward: float
    next_state: np.ndarray
    done: bool
    truncated: bool
    source_file: str


class TransitionDatasetWriter:
    """Crash-resistant, append-only writer for one behavior-policy run."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        environment: str,
        run_id: str,
        arms: Sequence[str],
        node_index: int,
        behavior_policy: str = "cmab",
        metadata: dict | None = None,
        enable_epoch_actions: bool = False,
    ) -> None:
        if node_index != 0:
            raise ValueError("offline transition export is owned by node0")
        if not arms:
            raise ValueError("transition export requires a non-empty action catalog")

        self.environment = str(environment).strip() or "unlabeled"
        self.behavior_policy = str(behavior_policy).strip()
        if not self.behavior_policy:
            raise ValueError("behavior policy must be a non-empty string")
        requested_run_id = str(run_id).strip() or datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"
        )
        self.arms = tuple(str(arm) for arm in arms)
        self._action_ids = {arm: index for index, arm in enumerate(self.arms)}
        self._written_keys: set[tuple[int, int]] = set()

        environment_dir = Path(root_dir).expanduser() / _safe_component(
            self.environment, "unlabeled"
        )
        base_name = _safe_component(requested_run_id, "run")
        run_dir = environment_dir / base_name
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = environment_dir / f"{base_name}-attempt{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)

        self.run_id = (
            requested_run_id
            if suffix == 1
            else f"{requested_run_id}-attempt{suffix}"
        )

        self.run_dir = run_dir
        self.transition_path = run_dir / "transitions.jsonl"
        self.epoch_action_path = (
            run_dir / "epoch_actions.jsonl" if enable_epoch_actions else None
        )
        self.manifest_path = run_dir / "meta.json"
        self._handle = self.transition_path.open("x", encoding="utf-8", buffering=1)
        self._epoch_action_handle = (
            self.epoch_action_path.open("x", encoding="utf-8", buffering=1)
            if self.epoch_action_path is not None
            else None
        )
        self._written_action_epochs: set[int] = set()

        catalog_payload = json.dumps(
            self.arms, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        manifest = {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "behavior_policy": self.behavior_policy,
            "environment": self.environment,
            "run_id": self.run_id,
            "node_index": node_index,
            "state_schema": DQN_STATE_SCHEMA,
            "state_representation": "raw; DQNPolicy performs final scaling",
            "action_count": len(self.arms),
            "action_catalog_sha256": hashlib.sha256(catalog_payload).hexdigest(),
            "arms": list(self.arms),
            "metadata": metadata or {},
        }
        if self.epoch_action_path is not None:
            manifest["epoch_action_file"] = self.epoch_action_path.name
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.manifest_path)

    def write(
        self,
        *,
        source_epoch: int,
        reward_epoch: int,
        state: np.ndarray | Sequence[float],
        arm: str,
        reward: float,
        next_state: np.ndarray | Sequence[float],
        done: bool = False,
        truncated: bool = False,
    ) -> bool:
        key = (int(source_epoch), int(reward_epoch))
        if key in self._written_keys:
            return False
        if int(reward_epoch) != int(source_epoch) + 1:
            raise ValueError("offline transition epochs must be contiguous")
        if arm not in self._action_ids:
            raise ValueError(f"offline transition contains an unknown arm: {arm}")
        if not math.isfinite(float(reward)) or not 0 < float(reward) <= 15:
            raise ValueError("offline transition reward must be finite and in (0, 15]")

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if state_array.size == 0 or state_array.size != next_state_array.size:
            raise ValueError("offline state and next_state dimensions must match")
        if not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(next_state_array)
        ):
            raise ValueError("offline states must contain only finite values")

        record = {
            "schema_version": TRANSITION_SCHEMA_VERSION,
            "behavior_policy": self.behavior_policy,
            "environment": self.environment,
            "run_id": self.run_id,
            "source_epoch": int(source_epoch),
            "reward_epoch": int(reward_epoch),
            "state": state_array.tolist(),
            "arm": arm,
            "action_id": self._action_ids[arm],
            "reward": float(reward),
            "next_state": next_state_array.tolist(),
            "done": bool(done),
            "truncated": bool(truncated),
        }
        self._handle.write(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._written_keys.add(key)
        return True

    def write_epoch_action(
        self,
        *,
        source_epoch: int,
        reward_epoch: int,
        selected_arm: str,
        effective_arm: str | None,
        abandoned: bool,
    ) -> bool:
        """Record the arm credited to an observed epoch, even when abandoned.

        This is an audit record, not an offline-training transition. The
        effective arm follows CMAB's existing abandon attribution logic.
        """
        epoch = int(reward_epoch)
        if self._epoch_action_handle is None:
            return False
        if epoch in self._written_action_epochs:
            return False
        if selected_arm not in self._action_ids or (
            effective_arm is not None
            and effective_arm not in self._action_ids
        ):
            raise ValueError("epoch action contains an unknown arm")
        status = (
            "abandoned_effective_unknown" if effective_arm is None
            else "abandoned_reused_previous" if abandoned else "applied"
        )
        record = {
            "schema_version": 1,
            "environment": self.environment,
            "run_id": self.run_id,
            "source_epoch": int(source_epoch),
            "epoch": epoch,
            "selected_arm": selected_arm,
            "selected_action_id": self._action_ids[selected_arm],
            "effective_arm": effective_arm,
            "effective_action_id": (
                self._action_ids[effective_arm] if effective_arm is not None else None
            ),
            "status": status,
        }
        self._epoch_action_handle.write(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        )
        self._epoch_action_handle.flush()
        os.fsync(self._epoch_action_handle.fileno())
        self._written_action_epochs.add(epoch)
        return True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if self._epoch_action_handle is not None and not self._epoch_action_handle.closed:
            self._epoch_action_handle.close()


class AsyncTransitionDatasetWriter:
    """Best-effort transition sink that never performs disk I/O on CMAB's loop.

    Dataset creation happens before CMAB starts.  Individual records are copied
    into a bounded queue and persisted by a daemon thread.  A full queue or any
    writer failure disables/drops export work without propagating into CMAB.
    """

    _STOP = object()

    def __init__(
        self,
        *,
        root_dir: str | Path,
        environment: str,
        run_id: str,
        arms: Sequence[str],
        node_index: int,
        behavior_policy: str = "cmab",
        metadata: dict | None = None,
        enable_epoch_actions: bool = False,
        queue_capacity: int = 256,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("transition export queue capacity must be positive")
        self._writer = TransitionDatasetWriter(
            root_dir=root_dir,
            environment=environment,
            run_id=run_id,
            arms=arms,
            node_index=node_index,
            behavior_policy=behavior_policy,
            metadata=metadata,
            enable_epoch_actions=enable_epoch_actions,
        )
        self.environment = self._writer.environment
        self.run_id = self._writer.run_id
        self.run_dir = self._writer.run_dir
        self.transition_path = self._writer.transition_path
        self.epoch_action_path = self._writer.epoch_action_path
        self.manifest_path = self._writer.manifest_path
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._closed = threading.Event()
        self._failed = threading.Event()
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"cmab-transition-writer-{self.run_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def failure(self) -> Exception | None:
        return self._failure

    def write(
        self,
        *,
        source_epoch: int,
        reward_epoch: int,
        state: np.ndarray | Sequence[float],
        arm: str,
        reward: float,
        next_state: np.ndarray | Sequence[float],
        done: bool = False,
        truncated: bool = False,
    ) -> bool:
        if self._closed.is_set() or self._failed.is_set():
            return False
        # Copy mutable inputs before returning control to the CMAB loop.
        payload = {
            "kind": "transition",
            "source_epoch": int(source_epoch),
            "reward_epoch": int(reward_epoch),
            "state": np.asarray(state, dtype=np.float32).reshape(-1).copy(),
            "arm": str(arm),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32)
            .reshape(-1)
            .copy(),
            "done": bool(done),
            "truncated": bool(truncated),
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning(
                "CMAB_OFFLINE_TRANSITION_DROPPED reason=queue_full "
                "run=%s source_epoch=%s reward_epoch=%s",
                self.run_id,
                source_epoch,
                reward_epoch,
            )
            return False
        return True

    def write_epoch_action(
        self,
        *,
        source_epoch: int,
        reward_epoch: int,
        selected_arm: str,
        effective_arm: str | None,
        abandoned: bool,
    ) -> bool:
        if (
            self.epoch_action_path is None
            or self._closed.is_set()
            or self._failed.is_set()
        ):
            return False
        payload = {
            "kind": "epoch_action",
            "source_epoch": int(source_epoch),
            "reward_epoch": int(reward_epoch),
            "selected_arm": str(selected_arm),
            "effective_arm": (
                str(effective_arm) if effective_arm is not None else None
            ),
            "abandoned": bool(abandoned),
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning(
                "CMAB_EPOCH_ACTION_DROPPED reason=queue_full "
                "run=%s epoch=%s", self.run_id, reward_epoch
            )
            return False
        return True

    def _run(self) -> None:
        try:
            while True:
                payload = self._queue.get()
                try:
                    if payload is self._STOP:
                        return
                    if self._failed.is_set():
                        continue
                    assert isinstance(payload, dict)
                    kind = payload.pop("kind")
                    if kind == "epoch_action":
                        recorded = self._writer.write_epoch_action(**payload)
                        if recorded:
                            logger.info(
                                "CMAB_EPOCH_ACTION_RECORDED run=%s epoch=%d "
                                "status=%s path=%s",
                                self.run_id,
                                payload["reward_epoch"],
                                (
                                    "abandoned_effective_unknown"
                                    if payload["effective_arm"] is None
                                    else "abandoned_reused_previous"
                                    if payload["abandoned"]
                                    else "applied"
                                ),
                                self.epoch_action_path,
                            )
                    elif kind == "transition":
                        exported = self._writer.write(**payload)
                        if not exported:
                            continue
                        logger.info(
                            "CMAB_OFFLINE_TRANSITION_EXPORTED run=%s "
                            "source_epoch=%d reward_epoch=%d arm=%s reward=%.6f "
                            "path=%s",
                            self.run_id,
                            payload["source_epoch"],
                            payload["reward_epoch"],
                            payload["arm"],
                            payload["reward"],
                            self.transition_path,
                        )
                    else:
                        raise ValueError(f"unknown dataset record kind: {kind}")
                except Exception as error:
                    self._failure = error
                    self._failed.set()
                    logger.exception(
                        "CMAB_OFFLINE_TRANSITION_EXPORT_DISABLED run=%s; "
                        "CMAB will continue without dataset export",
                        self.run_id,
                    )
                finally:
                    self._queue.task_done()
        finally:
            try:
                self._writer.close()
            except Exception:
                logger.exception(
                    "Failed to close CMAB transition dataset run=%s",
                    self.run_id,
                )

    def close(self, timeout: float = 2.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            # Shutdown must not block CMAB. Drop one pending export to make room
            # for the stop marker; the JSONL remains valid and append-only.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(self._STOP)
            except (queue.Empty, queue.Full):
                logger.warning(
                    "Could not enqueue transition-writer stop marker for run=%s",
                    self.run_id,
                )
        self._thread.join(timeout=max(0.0, float(timeout)))
        if self._thread.is_alive():
            logger.warning(
                "CMAB transition writer did not stop within %.1fs for run=%s; "
                "CMAB shutdown will continue",
                timeout,
                self.run_id,
            )


def discover_transition_files(
    dataset_root: str | Path,
    environments: Iterable[str] | None = None,
) -> list[Path]:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"offline dataset root does not exist: {root}")
    allowed = {str(value) for value in environments or ()}
    paths = []
    for path in root.glob("*/*/transitions.jsonl"):
        if not allowed:
            paths.append(path)
            continue
        manifest_path = path.parent / "meta.json"
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if str(manifest.get("environment")) in allowed:
            paths.append(path)
    return sorted(paths)


def load_transition_files(paths: Iterable[str | Path]) -> list[OfflineTransition]:
    records: list[OfflineTransition] = []
    seen: set[tuple[str, str, int, int]] = set()
    state_dim: int | None = None

    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if int(data.get("schema_version", -1)) != TRANSITION_SCHEMA_VERSION:
                        raise ValueError("unsupported transition schema version")
                    source_epoch = int(data["source_epoch"])
                    reward_epoch = int(data["reward_epoch"])
                    if reward_epoch != source_epoch + 1:
                        raise ValueError("transition epochs are not contiguous")
                    run_id = str(data["run_id"])
                    environment = str(data.get("environment", "unlabeled"))
                    key = (environment, run_id, source_epoch, reward_epoch)
                    if key in seen:
                        continue
                    state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
                    next_state = np.asarray(
                        data["next_state"], dtype=np.float32
                    ).reshape(-1)
                    reward = float(data["reward"])
                    if (
                        state.size == 0
                        or state.size != next_state.size
                        or not np.all(np.isfinite(state))
                        or not np.all(np.isfinite(next_state))
                    ):
                        raise ValueError("invalid state vectors")
                    if not math.isfinite(reward) or not 0 < reward <= 15:
                        raise ValueError("invalid reward")
                    if state_dim is None:
                        state_dim = int(state.size)
                    elif state.size != state_dim:
                        raise ValueError(
                            f"state dimension changed from {state_dim} to {state.size}"
                        )
                    record = OfflineTransition(
                        environment=environment,
                        run_id=run_id,
                        source_epoch=source_epoch,
                        reward_epoch=reward_epoch,
                        state=state,
                        arm=str(data["arm"]),
                        action_id=int(data["action_id"]),
                        reward=reward,
                        next_state=next_state,
                        done=bool(data.get("done", False)),
                        truncated=bool(data.get("truncated", False)),
                        source_file=str(path),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid offline transition at {path}:{line_number}: {error}"
                    ) from error
                seen.add(key)
                records.append(record)

    if not records:
        raise ValueError("no valid offline transitions were found")
    return records


class BalancedTransitionSampler:
    """Sample environment, then run, then transition with uniform probability."""

    def __init__(self, records: Sequence[OfflineTransition], seed: int = 0) -> None:
        if not records:
            raise ValueError("balanced sampler requires at least one transition")
        grouped: dict[str, dict[str, list[OfflineTransition]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in records:
            grouped[record.environment][record.run_id].append(record)
        self._grouped = {
            environment: dict(runs) for environment, runs in grouped.items()
        }
        self.environments = tuple(sorted(self._grouped))
        self.runs = {
            environment: tuple(sorted(runs))
            for environment, runs in self._grouped.items()
        }
        self.rng = np.random.default_rng(int(seed))

    def sample(self, batch_size: int) -> list[OfflineTransition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        result = []
        for _ in range(batch_size):
            environment = self.environments[
                int(self.rng.integers(len(self.environments)))
            ]
            run_ids = self.runs[environment]
            run_id = run_ids[int(self.rng.integers(len(run_ids)))]
            transitions = self._grouped[environment][run_id]
            result.append(transitions[int(self.rng.integers(len(transitions)))])
        return result


def dataset_summary(records: Sequence[OfflineTransition]) -> dict:
    return {
        "transitions": len(records),
        "state_dim": int(records[0].state.size) if records else 0,
        "environments": dict(Counter(record.environment for record in records)),
        "runs": dict(Counter(record.run_id for record in records)),
        "actions": dict(Counter(record.arm for record in records)),
    }
