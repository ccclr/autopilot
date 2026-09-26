#!/usr/bin/env python3
"""Monitor Autopilot liveness and auto-recover stalled runs.

Triggers (either one):
  1) metrics-0 has no new files for --metrics-stall-sec (default 60s)
  2) primary-0.log keeps creating blocks but stops committing for
     --commit-stall-sec (default 60s)

Recovery:
  - kill remote node/client/controller processes
  - restart `fab remote` from the newest checkpoint under --checkpoint-dir
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CREATED_RE = re.compile(r"Created B(\d+)\(")
COMMITTED_RE = re.compile(r"Committed B(\d+)\(")
CHECKPOINT_RE = re.compile(r"(\d+)\.pkl$")


def _kill_all(settings_path: Path) -> None:
    # Lazy import: fabric lives in the system/user site-packages used by `fab`.
    from kill_remote import kill_all

    kill_all(settings_path)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("liveness_monitor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


@dataclass
class StallSignal:
    reason: str
    detail: str


class LogTailer:
    """Incrementally scan a growing (or truncated) log file."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.inode: Optional[int] = None
        self.last_create_ts = 0.0
        self.last_commit_ts = 0.0
        self.creates_in_window = 0
        self.window_start = time.time()

    def _reset_identity(self, st: os.stat_result) -> None:
        self.inode = st.st_ino
        self.offset = 0

    def poll(self, now: float, create_window_sec: float) -> None:
        if not self.path.exists():
            return
        st = self.path.stat()
        if self.inode is None or st.st_ino != self.inode or st.st_size < self.offset:
            self._reset_identity(st)

        if now - self.window_start >= create_window_sec:
            self.creates_in_window = 0
            self.window_start = now

        with self.path.open("r", errors="replace") as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line:
                    break
                if CREATED_RE.search(line):
                    self.last_create_ts = now
                    self.creates_in_window += 1
                elif COMMITTED_RE.search(line):
                    self.last_commit_ts = now
            self.offset = f.tell()


def newest_metrics_mtime(metrics_dir: Path) -> Optional[float]:
    if not metrics_dir.exists():
        return None
    newest: Optional[float] = None
    for path in metrics_dir.glob("global_state_epoch_*.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is not None:
        return newest
    # Fallback: any file in the directory.
    for path in metrics_dir.iterdir():
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    cands = list(checkpoint_dir.glob("*.pkl"))
    if not cands:
        return None

    def sort_key(p: Path):
        m = CHECKPOINT_RE.search(p.name)
        iter_num = int(m.group(1)) if m else -1
        return (p.stat().st_mtime, iter_num)

    return max(cands, key=sort_key)


def _pkill_local_fab(logger: logging.Logger) -> None:
    # Only the benchmark launcher; monitor itself is monitor_liveness.py.
    cmd = ["pkill", "-f", r"fab(ric)? .*remote"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    logger.info("Local fab remote kill: exit=%s", result.returncode)


def recover(
    *,
    logger: logging.Logger,
    settings_path: Path,
    checkpoint_dir: Path,
    benchmark_dir: Path,
    dry_run: bool,
    fab_bin: str,
) -> Optional[subprocess.Popen]:
    ckpt = latest_checkpoint(checkpoint_dir)
    if ckpt is None:
        logger.error(
            "No checkpoint found under %s; kill will proceed but resume is skipped",
            checkpoint_dir,
        )
    else:
        logger.warning("Will resume from checkpoint: %s", ckpt)

    if dry_run:
        logger.warning("DRY-RUN: skip kill/restart")
        return None

    logger.warning("Killing local fab remote (if any)...")
    _pkill_local_fab(logger)
    time.sleep(1)

    logger.warning("Killing remote autopilot processes...")
    _kill_all(settings_path)
    time.sleep(2)

    if ckpt is None:
        logger.error("Cannot restart without a checkpoint")
        return None

    env = os.environ.copy()
    env["AUTOPILOT_RESUME_FROM"] = str(ckpt)
    cmd = [fab_bin, "remote", f"--resume-from={ckpt}"]
    out_path = Path.home() / "logs" / "liveness_recovery_fab.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.warning("Restarting: %s (cwd=%s)", " ".join(cmd), benchmark_dir)
    out_f = open(out_path, "a", buffering=1)
    out_f.write(f"\n===== recovery restart {time.strftime('%Y-%m-%d %H:%M:%S')} ckpt={ckpt} =====\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(benchmark_dir),
        env=env,
        stdout=out_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.warning("Started fab remote pid=%s; fab log=%s", proc.pid, out_path)
    return proc


def evaluate(
    *,
    now: float,
    armed_at: float,
    grace_sec: float,
    metrics_dir: Path,
    metrics_stall_sec: float,
    tailer: LogTailer,
    commit_stall_sec: float,
    min_creates: int,
) -> Optional[StallSignal]:
    if now < armed_at + grace_sec:
        return None

    metrics_mtime = newest_metrics_mtime(metrics_dir)
    if metrics_mtime is None:
        # After a fresh restart metrics may be empty briefly; grace covers this.
        if now >= armed_at + grace_sec:
            return StallSignal(
                reason="metrics_missing",
                detail=f"no metrics files under {metrics_dir}",
            )
    else:
        age = now - metrics_mtime
        if age >= metrics_stall_sec:
            return StallSignal(
                reason="metrics_stall",
                detail=f"newest metrics age={age:.1f}s >= {metrics_stall_sec}s",
            )

    # Create-without-commit: creates still flowing, commits stalled.
    if (
        tailer.last_create_ts > 0
        and (now - tailer.last_create_ts) <= 15.0
        and tailer.creates_in_window >= min_creates
        and (
            tailer.last_commit_ts <= 0
            or (now - tailer.last_commit_ts) >= commit_stall_sec
        )
    ):
        commit_age = (
            now - tailer.last_commit_ts if tailer.last_commit_ts > 0 else float("inf")
        )
        return StallSignal(
            reason="create_without_commit",
            detail=(
                f"creates_in_window={tailer.creates_in_window}, "
                f"commit_age={commit_age:.1f}s, "
                f"create_age={now - tailer.last_create_ts:.1f}s"
            ),
        )
    return None


def parse_args() -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(description="Autopilot liveness monitor with auto-recovery")
    p.add_argument("--primary-log", type=Path, default=home / "logs" / "primary-0.log")
    p.add_argument("--metrics-dir", type=Path, default=home / "metrics-0")
    p.add_argument("--checkpoint-dir", type=Path, default=home / "checkpoints")
    p.add_argument(
        "--settings",
        type=Path,
        default=_script_dir() / "settings.json",
    )
    p.add_argument("--benchmark-dir", type=Path, default=_script_dir())
    p.add_argument("--fab-bin", default=os.environ.get("FAB", "fab"))
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--grace-sec", type=float, default=90.0, help="Ignore stalls after start/recovery")
    p.add_argument("--cooldown-sec", type=float, default=180.0, help="Min seconds between recoveries")
    p.add_argument("--metrics-stall-sec", type=float, default=60.0)
    p.add_argument("--commit-stall-sec", type=float, default=60.0)
    p.add_argument("--create-window-sec", type=float, default=30.0)
    p.add_argument(
        "--min-creates",
        type=int,
        default=20,
        help="Min Created-block lines in create-window to treat as create-without-commit",
    )
    p.add_argument("--max-recoveries", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="Detect only; do not kill/restart")
    p.add_argument(
        "--log-file",
        type=Path,
        default=home / "logs" / "liveness_monitor.log",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = _setup_logger(args.log_file)
    logger.info(
        "Starting liveness monitor primary=%s metrics=%s checkpoints=%s dry_run=%s",
        args.primary_log,
        args.metrics_dir,
        args.checkpoint_dir,
        args.dry_run,
    )

    tailer = LogTailer(args.primary_log)
    # Seed commit/create times from a small tail so we don't false-trigger immediately.
    if args.primary_log.exists():
        try:
            with args.primary_log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256_000))
                seed = f.read().decode("utf-8", errors="replace")
                now = time.time()
                if COMMITTED_RE.search(seed):
                    tailer.last_commit_ts = now
                if CREATED_RE.search(seed):
                    tailer.last_create_ts = now
                f.seek(0, os.SEEK_END)
                tailer.offset = f.tell()
                tailer.inode = args.primary_log.stat().st_ino
        except OSError as e:
            logger.warning("Failed to seed primary log tail: %s", e)

    armed_at = time.time()
    last_recovery_at = 0.0
    recoveries = 0
    fab_proc: Optional[subprocess.Popen] = None

    def _shutdown(signum, _frame):
        logger.info("Received signal %s, exiting", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        now = time.time()
        tailer.poll(now, args.create_window_sec)
        sig = evaluate(
            now=now,
            armed_at=armed_at,
            grace_sec=args.grace_sec,
            metrics_dir=args.metrics_dir,
            metrics_stall_sec=args.metrics_stall_sec,
            tailer=tailer,
            commit_stall_sec=args.commit_stall_sec,
            min_creates=args.min_creates,
        )

        metrics_mtime = newest_metrics_mtime(args.metrics_dir)
        metrics_age = (now - metrics_mtime) if metrics_mtime else None
        logger.info(
            "heartbeat metrics_age=%s create_win=%d commit_age=%.1f create_age=%.1f",
            f"{metrics_age:.1f}s" if metrics_age is not None else "n/a",
            tailer.creates_in_window,
            (now - tailer.last_commit_ts) if tailer.last_commit_ts > 0 else -1.0,
            (now - tailer.last_create_ts) if tailer.last_create_ts > 0 else -1.0,
        )

        if sig is not None:
            if recoveries >= args.max_recoveries:
                logger.error(
                    "Stall detected (%s: %s) but max_recoveries=%d reached; giving up",
                    sig.reason,
                    sig.detail,
                    args.max_recoveries,
                )
                return 2
            if now - last_recovery_at < args.cooldown_sec and last_recovery_at > 0:
                logger.warning(
                    "Stall detected (%s: %s) but cooldown active (%.0fs left)",
                    sig.reason,
                    sig.detail,
                    args.cooldown_sec - (now - last_recovery_at),
                )
            else:
                logger.error("STALL DETECTED reason=%s detail=%s", sig.reason, sig.detail)
                fab_proc = recover(
                    logger=logger,
                    settings_path=args.settings,
                    checkpoint_dir=args.checkpoint_dir,
                    benchmark_dir=args.benchmark_dir,
                    dry_run=args.dry_run,
                    fab_bin=args.fab_bin,
                )
                recoveries += 1
                last_recovery_at = time.time()
                armed_at = last_recovery_at
                # Reset log scan after recovery so old create spam doesn't retrigger.
                tailer = LogTailer(args.primary_log)
                if args.primary_log.exists():
                    try:
                        st = args.primary_log.stat()
                        tailer.inode = st.st_ino
                        tailer.offset = st.st_size
                        tailer.last_commit_ts = last_recovery_at
                        tailer.last_create_ts = last_recovery_at
                    except OSError:
                        pass
                logger.info(
                    "Recovery #%d done; grace=%.0fs cooldown=%.0fs",
                    recoveries,
                    args.grace_sec,
                    args.cooldown_sec,
                )

        if fab_proc is not None and fab_proc.poll() is not None:
            logger.warning("Managed fab remote exited with code %s", fab_proc.returncode)
            fab_proc = None

        time.sleep(max(1.0, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
