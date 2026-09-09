#!/usr/bin/env python3
"""Run round-robin CMAB until ~192 slot windows cover the 96-arm catalog."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
BENCH = Path(__file__).resolve().parent
KILL = BENCH / "kill_remote.py"
SYSTEM_PYTHON = "/usr/bin/python3"
METRICS_DIR = HOME / "metrics-0"
EXP_DIR = HOME / "experiment" / "factorized_reward" / "lasso_coverage"
TARGET_WINDOWS = 192
STALL_SEC = 120
BOOT_SEC = 1200
POLL_SEC = 10
MAX_ATTEMPTS = 3
FAB_DURATION = 10800
SLOT_RE = re.compile(r"epoch_(\d+)_slot_(\d+)\.json$")


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}Z] {msg}"
    print(line, flush=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "run.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def kill_cluster() -> None:
    subprocess.run([SYSTEM_PYTHON, str(KILL)], cwd=str(BENCH), check=False)
    time.sleep(3)


def archive_old_metrics() -> None:
    if not METRICS_DIR.is_dir():
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        return
    files = list(METRICS_DIR.glob("epoch_*_slot_*.json"))
    if not files:
        return
    dest = EXP_DIR / "metrics-0-before"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(METRICS_DIR, dest)
    shutil.rmtree(METRICS_DIR, ignore_errors=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"archived {len(files)} old slot files to {dest}")


def slot_stats() -> dict:
    rows = []
    newest_mtime = None
    if METRICS_DIR.is_dir():
        for path in METRICS_DIR.glob("epoch_*_slot_*.json"):
            match = SLOT_RE.search(path.name)
            if not match:
                continue
            mtime = path.stat().st_mtime
            newest_mtime = mtime if newest_mtime is None else max(newest_mtime, mtime)
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            action = data.get("current_action") or {}
            arm = (
                action.get("batch_size"),
                action.get("header_size"),
                action.get("cut_condition_type"),
                action.get("fast_path_timeout_ms", action.get("fast_path_timeout")),
                action.get("parallel_proposals", action.get("k")),
            )
            rows.append({"epoch": int(match.group(1)), "arm": arm})
    unique = {row["arm"] for row in rows if None not in row["arm"]}
    k_vals = {row["arm"][4] for row in rows if row["arm"][4] is not None}
    epochs = [row["epoch"] for row in rows]
    return {
        "n": len(rows),
        "unique": len(unique),
        "k_vals": sorted(k_vals, key=lambda value: str(value)),
        "epoch_min": min(epochs) if epochs else None,
        "epoch_max": max(epochs) if epochs else None,
        "newest_mtime": newest_mtime,
    }


def start_fab() -> subprocess.Popen:
    cmd = [
        "fab",
        "remote",
        "--cmab-policy=round_robin",
        "--cmab-seed=0",
        f"--duration={FAB_DURATION}",
    ]
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(EXP_DIR / "fab.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BENCH),
        stdout=handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    return proc


def stop_fab(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        handle = getattr(proc, "_log_handle", None)
        if handle:
            handle.close()
    kill_cluster()


def run_attempt(attempt: int) -> dict:
    log(f"attempt {attempt}: starting fab remote round_robin")
    proc = start_fab()
    t0 = time.time()
    last_n = 0
    last_progress = time.time()
    try:
        while True:
            time.sleep(POLL_SEC)
            if proc.poll() is not None:
                log(f"fab exited early rc={proc.returncode}")
                return {**slot_stats(), "status": "fab_exit"}
            stats = slot_stats()
            if stats["n"] != last_n:
                last_n = stats["n"]
                last_progress = time.time()
                log(
                    f"windows={stats['n']} unique_arms={stats['unique']} "
                    f"k={stats['k_vals']} epochs={stats['epoch_min']}..{stats['epoch_max']}"
                )
            if stats["n"] >= TARGET_WINDOWS and stats["unique"] >= 80:
                log("coverage target reached")
                return {**stats, "status": "ok"}
            now = time.time()
            if stats["n"] == 0 and now - t0 > BOOT_SEC:
                log("boot timeout: no slot files")
                return {**stats, "status": "boot_timeout"}
            if stats["n"] > 0 and now - last_progress > STALL_SEC:
                log(f"stall: no new slot file for {STALL_SEC}s")
                return {**stats, "status": "stall"}
    finally:
        stop_fab(proc)


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    kill_cluster()
    archive_old_metrics()
    last = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        last = run_attempt(attempt)
        if last.get("status") == "ok":
            break
        log(f"attempt {attempt} ended status={last.get('status')}")
        if attempt < MAX_ATTEMPTS:
            archive = EXP_DIR / f"metrics-attempt-{attempt}"
            if METRICS_DIR.is_dir():
                if archive.exists():
                    shutil.rmtree(archive, ignore_errors=True)
                shutil.copytree(METRICS_DIR, archive)
            shutil.rmtree(METRICS_DIR, ignore_errors=True)
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "meta.json").write_text(json.dumps(last, indent=2) + "\n")
    log(f"done status={last.get('status')} windows={last.get('n')} unique={last.get('unique')}")
    return 0 if last.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
