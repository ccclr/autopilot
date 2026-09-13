#!/usr/bin/env python3
"""Run factorized vs global-reward CMAB, 3 seeds each, 200 valid policy updates.

Valid epoch = TRAIN_SAMPLE in node-0 continuous_training log (apply-ok and
post-warmup). Consensus stall (>120s without a new global_state epoch) restarts
the attempt. Artifacts go to ~/experiments/<run_id>/.
"""

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
TRAIN_LOG = HOME / "logs" / "continuous_training_0.log"
METRICS_DIR = HOME / "metrics-0"
EXP_ROOT = HOME / "experiments"
TARGET_VALID = 200
STALL_SEC = 120
BOOT_SEC = 1200
POLL_SEC = 10
MAX_ATTEMPTS = 3
FAB_DURATION = 10800

TRAIN_SAMPLE_RE = re.compile(r"TRAIN_SAMPLE\s+idx=")
EPOCH_RE = re.compile(r"global_state_epoch_(\d+)\.json$")

EXPERIMENTS = [
    {"factorized": True, "seed": 0},
    {"factorized": True, "seed": 1},
    {"factorized": True, "seed": 2},
    {"factorized": False, "seed": 0},
    {"factorized": False, "seed": 1},
    {"factorized": False, "seed": 2},
]


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}Z] {msg}"
    print(line, flush=True)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    with open(EXP_ROOT / "sweep.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_id(factorized: bool, seed: int, attempt: int) -> str:
    kind = "factorized" if factorized else "global"
    return f"{kind}_seed{seed}_try{attempt}"


def kill_cluster() -> None:
    subprocess.run(
        [SYSTEM_PYTHON, str(KILL)],
        cwd=str(BENCH),
        check=False,
    )
    time.sleep(3)


def clear_local_metrics() -> None:
    if METRICS_DIR.is_dir():
        shutil.rmtree(METRICS_DIR, ignore_errors=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_LOG.exists():
        TRAIN_LOG.unlink()


def count_train_samples(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if TRAIN_SAMPLE_RE.search(line):
                n += 1
    return n


def latest_epoch(metrics_dir: Path) -> tuple[int | None, float | None]:
    newest_epoch = None
    newest_mtime = None
    if not metrics_dir.is_dir():
        return None, None
    for p in metrics_dir.glob("global_state_epoch_*.json"):
        m = EPOCH_RE.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        mtime = p.stat().st_mtime
        if newest_epoch is None or epoch > newest_epoch:
            newest_epoch = epoch
            newest_mtime = mtime
        elif epoch == newest_epoch and (newest_mtime is None or mtime > newest_mtime):
            newest_mtime = mtime
    return newest_epoch, newest_mtime


def start_fab(factorized: bool, seed: int) -> subprocess.Popen:
    policy = "factorized" if factorized else "rf_ts"
    cmd = [
        "fab",
        "remote",
        f"--cmab-seed={seed}",
        f"--cmab-policy={policy}",
        f"--duration={FAB_DURATION}",
    ]
    log_path = EXP_ROOT / "current_fab.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
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
    if proc is None:
        return
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


def archive_run(name: str, meta: dict) -> None:
    dest = EXP_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logs_src = HOME / "logs"
    if logs_src.is_dir():
        for pattern in (
            "continuous_training_*.log",
            "primary-*.log",
            "metrics_collector-*.log",
        ):
            for p in logs_src.glob(pattern):
                shutil.copy2(p, dest / p.name)
    ckpt_dirs = [
        HOME / "checkpoints" / "cmab_factorized",
        HOME / "checkpoints" / "cmab_one_hot",
        HOME / "checkpoints",
    ]
    ckpt_dest = dest / "checkpoints"
    ckpt_dest.mkdir(exist_ok=True)
    for ckpt_dir in ckpt_dirs:
        if not ckpt_dir.is_dir():
            continue
        for p in ckpt_dir.glob("cmab_checkpoint_*.pkl"):
            shutil.copy2(p, ckpt_dest / f"{ckpt_dir.name}_{p.name}")
    for i in range(4):
        host = f"10.10.1.{i + 1}"
        remote = f"logs/continuous_training_{i}.log"
        local = dest / f"continuous_training_{i}.log"
        subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"clr0302@{host}:{remote}",
                str(local),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run_attempt(factorized: bool, seed: int, attempt: int) -> tuple[bool, dict]:
    name = run_id(factorized, seed, attempt)
    log(f"START {name} factorized={factorized} seed={seed} attempt={attempt}")
    kill_cluster()
    clear_local_metrics()
    proc = start_fab(factorized, seed)
    t0 = time.time()
    first_epoch_at = None
    last_epoch = None
    last_epoch_change = None
    last_report = t0
    outcome = "unknown"
    try:
        while True:
            time.sleep(POLL_SEC)
            now = time.time()
            if proc.poll() is not None and now - t0 > 60:
                outcome = f"fab_exited_{proc.returncode}"
                log(f"FAB EXIT {name} code={proc.returncode}")
                break
            valid = count_train_samples(TRAIN_LOG)
            epoch, epoch_mtime = latest_epoch(METRICS_DIR)
            if epoch is not None:
                if first_epoch_at is None:
                    first_epoch_at = now
                    last_epoch = epoch
                    last_epoch_change = now
                    log(f"{name} first epoch={epoch}")
                elif epoch != last_epoch:
                    last_epoch = epoch
                    last_epoch_change = now
            if now - last_report >= 60:
                log(
                    f"{name} valid={valid}/{TARGET_VALID} epoch={epoch} "
                    f"elapsed={int(now - t0)}s"
                )
                last_report = now
            if valid >= TARGET_VALID:
                outcome = "success"
                log(f"SUCCESS {name} valid={valid} epoch={epoch}")
                return True, {
                    "run_id": name,
                    "factorized": factorized,
                    "seed": seed,
                    "attempt": attempt,
                    "outcome": outcome,
                    "valid_updates": valid,
                    "last_epoch": epoch,
                    "elapsed_sec": round(now - t0, 1),
                }
            if first_epoch_at is None:
                if now - t0 > BOOT_SEC:
                    outcome = "boot_timeout"
                    log(f"BOOT TIMEOUT {name}")
                    break
                continue
            stall_ref = last_epoch_change or first_epoch_at
            if now - stall_ref > STALL_SEC:
                outcome = "consensus_stall"
                log(
                    f"STALL {name} no new epoch for {int(now - stall_ref)}s "
                    f"(last_epoch={last_epoch})"
                )
                break
    finally:
        stop_fab(proc)

    meta = {
        "run_id": name,
        "factorized": factorized,
        "seed": seed,
        "attempt": attempt,
        "outcome": outcome,
        "valid_updates": count_train_samples(TRAIN_LOG),
        "last_epoch": last_epoch,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    archive_run(name, meta)
    return False, meta


def main() -> int:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    log(f"Sweep start target_valid={TARGET_VALID} stall={STALL_SEC}s")
    for spec in EXPERIMENTS:
        factorized = spec["factorized"]
        seed = spec["seed"]
        success = False
        last_meta = {}
        for attempt in range(1, MAX_ATTEMPTS + 1):
            ok, meta = run_attempt(factorized, seed, attempt)
            last_meta = meta
            if ok:
                archive_run(meta["run_id"], meta)
                success = True
                break
            log(f"RETRY {run_id(factorized, seed, attempt)} next attempt")
        last_meta["success"] = success
        summary.append(last_meta)
        (EXP_ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if not success:
            log(f"FAILED experiment factorized={factorized} seed={seed}")
    log("Sweep done")
    print(json.dumps(summary, indent=2))
    return 0 if all(x.get("success") for x in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
