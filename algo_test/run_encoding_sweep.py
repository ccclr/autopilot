#!/usr/bin/env python3
"""Run CMAB numeric vs one_hot encoding comparison.

Order is paired by seed so both encodings for the same seed finish on the
same day under the same CloudLab load:

  (seed=0, numeric), (seed=0, one_hot),
  (seed=1, numeric), (seed=1, one_hot),
  (seed=2, numeric), (seed=2, one_hot)

Each cell:
  - starts from scratch (no checkpoint resume)
  - stops when metrics-0 reaches --target-epoch (default 150)
  - if metrics-0 gets no new file for --stall-sec (default 120) after the
    first file appears, kill every node and restart that cell from scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from kill_remote import kill_all

EPOCH_RE = re.compile(r"(?:global_state_epoch_|epoch_)(\d+)")
JOBS = (
    (0, "numeric"),
    (0, "one_hot"),
    (1, "numeric"),
    (1, "one_hot"),
    (2, "numeric"),
    (2, "one_hot"),
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("encoding_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _checkpoint_dir(home: Path, encoding: str) -> Path:
    if encoding == "one_hot":
        return home / "checkpoints" / "cmab_one_hot"
    return home / "checkpoints"


def _metrics_dirs(home: Path) -> list[Path]:
    return sorted(
        p for p in home.glob("metrics-*") if p.is_dir() and re.fullmatch(r"metrics-\d+", p.name)
    )


def _wipe_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def newest_metrics_mtime(metrics_dir: Path) -> Optional[float]:
    if not metrics_dir.exists():
        return None
    newest: Optional[float] = None
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


def max_epoch(metrics_dir: Path, since: float) -> Optional[int]:
    if not metrics_dir.exists():
        return None
    best: Optional[int] = None
    for path in metrics_dir.iterdir():
        if not path.is_file():
            continue
        match = EPOCH_RE.search(path.name)
        if not match:
            continue
        try:
            if path.stat().st_mtime < since:
                continue
        except OSError:
            continue
        epoch = int(match.group(1))
        if best is None or epoch > best:
            best = epoch
    return best


def _pkill_local_fab(logger: logging.Logger) -> None:
    result = subprocess.run(
        ["pkill", "-f", r"fab(ric)? .*remote"],
        check=False,
        capture_output=True,
        text=True,
    )
    logger.info("Local fab remote kill: exit=%s", result.returncode)


def kill_cluster(logger: logging.Logger, settings_path: Path) -> None:
    logger.warning("Killing local fab remote and all remote Autopilot processes")
    _pkill_local_fab(logger)
    time.sleep(1)
    try:
        kill_all(settings_path)
    except Exception as exc:
        logger.error("Remote kill failed: %s", exc)
    time.sleep(2)


def archive_run(
    *,
    home: Path,
    dest: Path,
    seed: int,
    encoding: str,
    epoch: Optional[int],
    attempt: int,
    logger: logging.Logger,
) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("logs",):
        src = home / name
        if src.exists():
            shutil.copytree(src, dest / name, dirs_exist_ok=True)
    bench_logs = _script_dir() / "logs"
    if bench_logs.exists():
        shutil.copytree(bench_logs, dest / "benchmark_logs", dirs_exist_ok=True)
    for metrics_dir in _metrics_dirs(home):
        shutil.copytree(metrics_dir, dest / metrics_dir.name, dirs_exist_ok=True)
    ckpt = _checkpoint_dir(home, encoding)
    if ckpt.exists():
        shutil.copytree(ckpt, dest / "checkpoints", dirs_exist_ok=True)
    meta = {
        "seed": seed,
        "encoding": encoding,
        "final_epoch": epoch,
        "attempt": attempt,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
    }
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("Archived %s seed=%s encoding=%s epoch=%s", dest, seed, encoding, epoch)


def prepare_fresh_attempt(home: Path, encoding: str, logger: logging.Logger) -> None:
    for metrics_dir in _metrics_dirs(home):
        _wipe_dir(metrics_dir)
    _wipe_dir(home / "metrics-0")
    ckpt = _checkpoint_dir(home, encoding)
    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    logger.info("Cleared metrics and %s for a from-scratch attempt", ckpt)


def start_fab(
    *,
    benchmark_dir: Path,
    fab_bin: str,
    seed: int,
    encoding: str,
    duration: int,
    log_path: Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        fab_bin,
        "remote",
        f"--cmab-seed={seed}",
        f"--cmab-action-encoding={encoding}",
        f"--duration={duration}",
    ]
    out_f = open(log_path, "a", buffering=1)
    out_f.write(
        f"\n===== {datetime.now().isoformat(timespec='seconds')} "
        f"seed={seed} encoding={encoding} =====\n"
    )
    return subprocess.Popen(
        cmd,
        cwd=str(benchmark_dir),
        stdout=out_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(description="CMAB numeric vs one_hot encoding sweep")
    p.add_argument("--home", type=Path, default=home)
    p.add_argument("--settings", type=Path, default=_script_dir() / "settings.json")
    p.add_argument("--benchmark-dir", type=Path, default=_script_dir())
    p.add_argument("--fab-bin", default=os.environ.get("FAB", "fab"))
    p.add_argument("--target-epoch", type=int, default=150)
    p.add_argument("--stall-sec", type=float, default=120.0)
    p.add_argument("--boot-timeout-sec", type=float, default=1200.0)
    p.add_argument("--poll-sec", type=float, default=5.0)
    p.add_argument("--max-restarts", type=int, default=8)
    p.add_argument("--duration", type=int, default=7200, help="fab safety cap in seconds")
    p.add_argument(
        "--archive-root",
        type=Path,
        default=home / "experiment" / "encoding_cmp",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=home / "experiment" / "encoding_cmp" / "sweep.log",
    )
    p.add_argument(
        "--only",
        default="",
        help="Optional subset like 0:numeric,0:one_hot",
    )
    return p.parse_args()


def selected_jobs(only: str):
    if not only.strip():
        return list(JOBS)
    jobs = []
    for item in only.split(","):
        seed_s, encoding = item.strip().split(":", 1)
        jobs.append((int(seed_s), encoding))
    return jobs


def run_one_cell(args: argparse.Namespace, logger: logging.Logger, seed: int, encoding: str) -> bool:
    metrics_dir = args.home / "metrics-0"
    dest = args.archive_root / f"seed{seed}_{encoding}"
    fab_log = args.archive_root / "fab_remote.log"

    for attempt in range(1, args.max_restarts + 1):
        logger.info(
            "=== start seed=%s encoding=%s attempt=%d/%d ===",
            seed,
            encoding,
            attempt,
            args.max_restarts,
        )
        kill_cluster(logger, args.settings)
        prepare_fresh_attempt(args.home, encoding, logger)
        started_at = time.time()
        proc = start_fab(
            benchmark_dir=args.benchmark_dir,
            fab_bin=args.fab_bin,
            seed=seed,
            encoding=encoding,
            duration=args.duration,
            log_path=fab_log,
        )
        logger.info("Started fab remote pid=%s", proc.pid)
        first_metric_at: Optional[float] = None

        while True:
            time.sleep(max(1.0, args.poll_sec))
            now = time.time()
            mtime = newest_metrics_mtime(metrics_dir)
            epoch = max_epoch(metrics_dir, started_at)
            if mtime is not None and first_metric_at is None:
                first_metric_at = now
                logger.info("First metrics-0 file after %.0fs", now - started_at)

            age = (now - mtime) if mtime is not None else None
            logger.info(
                "seed=%s encoding=%s attempt=%d epoch=%s metrics_age=%s fab_alive=%s",
                seed,
                encoding,
                attempt,
                epoch if epoch is not None else "n/a",
                f"{age:.1f}s" if age is not None else "n/a",
                proc.poll() is None,
            )
            write_state(
                args.archive_root / "sweep_state.json",
                {
                    "seed": seed,
                    "encoding": encoding,
                    "attempt": attempt,
                    "epoch": epoch,
                    "metrics_age_sec": age,
                    "fab_pid": proc.pid,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

            if epoch is not None and epoch >= args.target_epoch:
                logger.info(
                    "Reached epoch %s >= %s; stopping this cell",
                    epoch,
                    args.target_epoch,
                )
                kill_cluster(logger, args.settings)
                archive_run(
                    home=args.home,
                    dest=dest,
                    seed=seed,
                    encoding=encoding,
                    epoch=epoch,
                    attempt=attempt,
                    logger=logger,
                )
                return True

            stalled = False
            if first_metric_at is None:
                if now - started_at >= args.boot_timeout_sec:
                    stalled = True
                    reason = f"no metrics-0 after {args.boot_timeout_sec:.0f}s boot window"
                else:
                    reason = ""
            elif age is not None and age >= args.stall_sec:
                stalled = True
                reason = f"metrics-0 stale for {age:.1f}s"
            else:
                reason = ""

            if stalled:
                logger.error(
                    "Liveness failure seed=%s encoding=%s attempt=%d: %s",
                    seed,
                    encoding,
                    attempt,
                    reason,
                )
                kill_cluster(logger, args.settings)
                fail_dir = args.archive_root / "failed" / f"seed{seed}_{encoding}_attempt{attempt}"
                try:
                    archive_run(
                        home=args.home,
                        dest=fail_dir,
                        seed=seed,
                        encoding=encoding,
                        epoch=epoch,
                        attempt=attempt,
                        logger=logger,
                    )
                except Exception as exc:
                    logger.warning("Failed-attempt archive skipped: %s", exc)
                break

            if proc.poll() is not None:
                logger.error(
                    "fab remote exited early code=%s before epoch %s",
                    proc.returncode,
                    args.target_epoch,
                )
                kill_cluster(logger, args.settings)
                break

        logger.warning("Restarting seed=%s encoding=%s from scratch", seed, encoding)

    logger.error("Giving up on seed=%s encoding=%s after %s attempts", seed, encoding, args.max_restarts)
    return False


def main() -> int:
    args = parse_args()
    logger = _setup_logger(args.log_file)
    jobs = selected_jobs(args.only)
    logger.info("Encoding sweep jobs=%s target_epoch=%s stall=%ss", jobs, args.target_epoch, args.stall_sec)

    def _shutdown(signum, _frame):
        logger.info("Received signal %s; killing cluster and exiting", signum)
        kill_cluster(logger, args.settings)
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    results = []
    for seed, encoding in jobs:
        ok = run_one_cell(args, logger, seed, encoding)
        results.append({"seed": seed, "encoding": encoding, "ok": ok})
        write_state(args.archive_root / "sweep_results.json", {"results": results})
        if not ok:
            logger.error("Cell failed; continuing to the next paired job")

    kill_cluster(logger, args.settings)
    logger.info("Sweep finished: %s", results)
    return 0 if all(item["ok"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
