#!/usr/bin/env python3
"""Sweep fast_path_timeout (0..200 step 20), 3 independent runs per point.

Each trial keeps runs=1 (equivalent to re-invoking `fab remote`), archives the SUMMARY
block to a baseline-specific results directory, then moves to the next trial/timeout.

Usage (from benchmark/):
  python3 sweep_fast_path_timeout.py
  python3 sweep_fast_path_timeout.py --timeouts 20,40,100 --trials 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean

from fabric import Config
from invoke import Context

# Ensure local package imports work when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.cloudlab_remote import CloudLabBench as Bench
from benchmark.utils import BenchError, PathMaker, Print


# Fixed workload; independent of fabfile.remote(). Do not set runs>1.
BENCH_PARAMS = {
    "faults": 0,
    "nodes": [4],
    "workers": 1,
    "collocate": True,
    "rate": [40_000],
    "tx_size": 512,
    "duration": 120,
    "epochs": None,
    "runs": 1,
    "enable_rl": False,
    "enable_checkpoint": False,
    "enable_cmab_transition_export": False,
    "new_result_file_per_run": False,  # Archive reader expects the append-only file.
    "cmab_resume_from": None,
    "rl_algo": "gp_bo",
    "rl_warmup_iterations": 5,
    "simulate_partition": False,
    "partition_start": 5,
    "partition_duration": 5,
    "partition_nodes": 2,
    "enable_hotspot": False,
    "hotspot_windows": [[]],
    "hotspot_regions": [[]],
    "hotspot_nodes": [[]],
    "hotspot_region_rates": [[]],
}

# Offline environment A: 137 transitions across 3 runs, mean reward 2.099
# for batch=100000/header=32/cut=2/k=1 (pooling recorded timeout values).
# CMAB samples are not randomized trials; validate this candidate with the sweep.
NODE_PARAMS = {
    "timeout_delay": 5_000,
    "header_size": 32,
    "max_header_delay": 5000,
    "gc_depth": 50,
    "sync_retry_delay": 5000,
    "sync_retry_nodes": 3,
    "batch_size": 100_000,
    "max_batch_delay": 5000,
    "use_optimistic_tips": True,
    "use_parallel_proposals": True,
    "k": 1,
    "epoch_slots": 32,
    "window_size": 16,
    "applied_begin": 30,
    "use_fast_path": True,
    "fast_path_timeout": 100,
    "use_ride_share": False,
    "car_timeout": 2000,
    "cut_condition_type": 2,
    "simulate_asynchrony": False,
    "asynchrony_type": [6],
    "asynchrony_start": [0],
    "asynchrony_duration": [3000],
    "affected_nodes": [],
    "asynchrony_nodes": [],
    "asynchrony_regions": [[]],
    "egress_penalty": [[]],
    "use_fast_sync": True,
    "use_exponential_timeouts": True,
    "aggregation_strategy": "normal",
    "data_pollution_node_ids": [],
    "data_pollution_prob": 1.0,
    "data_pollution_strategy": "random_scale",
}


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _make_ctx() -> Context:
    return Context(config=Config())


def _default_result_file() -> Path:
    return _script_dir() / PathMaker.result_file(
        BENCH_PARAMS["faults"],
        BENCH_PARAMS["nodes"][0],
        BENCH_PARAMS["workers"],
        BENCH_PARAMS["collocate"],
        BENCH_PARAMS["rate"][0],
        BENCH_PARAMS["tx_size"],
    )


def _extract_last_summary(text: str) -> str | None:
    """Return the last full SUMMARY block (CONFIG + RESULTS), not just the title.

    Result files look like:
      -----------------------------------------\n
       SUMMARY:\n
      -----------------------------------------\n
       + CONFIG: ...\n
       + RESULTS: ...\n
      -----------------------------------------\n
    The dashed line under the title is NOT the end of the block.
    """
    sep = "-----------------------------------------"
    marker = " SUMMARY:\n"
    idx = text.rfind(marker)
    if idx < 0:
        return None

    start = text.rfind(sep, 0, idx)
    if start < 0:
        start = idx

    # Closing separator comes after RESULTS, never the one under SUMMARY.
    results = text.find(" + RESULTS:", idx)
    search_from = results if results >= 0 else (idx + len(marker))
    end = text.find(sep, search_from)
    if end < 0:
        return text[start:].rstrip() + "\n"
    end += len(sep)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[start:end]


def _parse_fast_path_timeout(summary: str) -> int | None:
    m = re.search(r"Fast path timeout:\s*(\d+)", summary)
    return int(m.group(1)) if m else None


def _parse_e2e_latency(summary: str) -> int | None:
    m = re.search(r"End-to-end latency:\s*([\d,]+)\s*ms", summary)
    return int(m.group(1).replace(",", "")) if m else None


def _archive_is_valid(path: Path, timeout: int) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    got = _parse_fast_path_timeout(text)
    e2e = _parse_e2e_latency(text)
    has_ratio = re.search(r"avg_fast_path_ratio=", text) is not None
    return got == timeout and e2e is not None and has_ratio


def _local_metrics_home() -> Path:
    """Home that contains metrics-* directories (settings.home, else $HOME)."""
    try:
        from benchmark.cloudlab_settings import CloudLabSettings

        settings = CloudLabSettings.load(str(_script_dir() / "cloudlab_settings.json"))
        return Path(settings.home)
    except Exception:
        return Path(os.environ.get("HOME", "/users/clr0302"))


def _discover_local_metrics_dirs(home: Path) -> list[Path]:
    """All local metrics-* directories under home; node id is not hardcoded."""
    return sorted(
        p for p in home.glob("metrics-*")
        if p.is_dir() and re.fullmatch(r"metrics-\d+", p.name)
    )


def _load_ratio_from_json(path: Path, preferred_keys: tuple[str, ...]) -> float | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in preferred_keys:
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
    return None


def _avg_fast_path_ratio_local() -> dict:
    """Scan local metrics-* dirs.

    Prefer global_state*.json (global_fast_path_ratio). If none found, fall back to
    other local JSON files (state_5_fast_path_ratio / fast_path_ratio).
    """
    home = _local_metrics_home()
    metrics_dirs = _discover_local_metrics_dirs(home)
    if not metrics_dirs:
        raise RuntimeError(f"No local metrics-* directories under {home}")

    global_files: dict[str, float] = {}
    local_files: dict[str, float] = {}
    skipped: list[str] = []

    for metrics_dir in metrics_dirs:
        # Prefer global aggregation files.
        for path in sorted(metrics_dir.glob("global_state*.json")):
            ratio = _load_ratio_from_json(
                path, ("global_fast_path_ratio", "fast_path_ratio")
            )
            key = f"{metrics_dir.name}/{path.name}"
            if ratio is None:
                skipped.append(f"{key}: no global ratio")
                continue
            global_files[key] = ratio

        # Fallback candidates: epoch_*_slot_*.json etc.
        for path in sorted(metrics_dir.glob("*.json")):
            if path.name.startswith("global_state"):
                continue
            ratio = _load_ratio_from_json(
                path, ("state_5_fast_path_ratio", "fast_path_ratio", "global_fast_path_ratio")
            )
            key = f"{metrics_dir.name}/{path.name}"
            if ratio is None:
                skipped.append(f"{key}: no local ratio")
                continue
            local_files[key] = ratio

    if global_files:
        source = "global"
        per_file = global_files
    elif local_files:
        source = "local_fallback"
        per_file = local_files
    else:
        raise RuntimeError(
            f"No fast_path_ratio in local metrics under {home}; "
            f"dirs={[p.name for p in metrics_dirs]}; skipped={skipped[:10]}"
        )

    values = list(per_file.values())
    return {
        "home": str(home),
        "metrics_dirs": [str(p) for p in metrics_dirs],
        "source": source,
        "n_files_used": len(per_file),
        "n_files_skipped": len(skipped),
        "skipped": skipped[:50],
        "per_file_fast_path_ratio": per_file,
        "avg_fast_path_ratio": mean(values),
    }


def _archive_summary(
    result_path: Path,
    archive_dir: Path,
    timeout: int,
    trial: int,
    size_before: int,
) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    if not result_path.exists():
        raise RuntimeError(f"Result file missing after run: {result_path}")

    text = result_path.read_text(encoding="utf-8", errors="replace")
    # Prefer the newly appended region; fall back to last summary in whole file.
    new_text = text[size_before:] if size_before < len(text) else text
    summary = _extract_last_summary(new_text) or _extract_last_summary(text)
    if not summary:
        raise RuntimeError("Could not find SUMMARY block after run")

    got = _parse_fast_path_timeout(summary)
    if got is None:
        raise RuntimeError("Archived SUMMARY missing 'Fast path timeout'")
    if got != timeout:
        raise RuntimeError(
            f"Archived SUMMARY fast_path_timeout={got} != expected {timeout}"
        )
    if _parse_e2e_latency(summary) is None:
        raise RuntimeError("Archived SUMMARY missing 'End-to-end latency'")

    ratio_info = _avg_fast_path_ratio_local()
    avg_ratio = ratio_info["avg_fast_path_ratio"]

    out = archive_dir / f"fpt-{timeout}ms-trial{trial}.txt"
    metrics_out = archive_dir / f"fpt-{timeout}ms-trial{trial}.metrics.json"
    header = (
        f"# archived {datetime.now().isoformat(timespec='seconds')}\n"
        f"# expected_fast_path_timeout_ms={timeout}\n"
        f"# trial={trial}\n"
        f"# source={result_path}\n"
        f"# metrics_dirs={json.dumps(ratio_info['metrics_dirs'])}\n"
        f"# ratio_source={ratio_info['source']}\n"
        f"# avg_fast_path_ratio={avg_ratio:.6f}\n"
        f"# n_files_used={ratio_info['n_files_used']}\n"
        f"# n_files_skipped={ratio_info['n_files_skipped']}\n"
    )
    out.write_text(header + summary, encoding="utf-8")
    metrics_out.write_text(json.dumps(ratio_info, indent=2) + "\n", encoding="utf-8")
    Print.info(
        f"avg fast_path_ratio ({ratio_info['source']}) over "
        f"{ratio_info['n_files_used']} JSON files in "
        f"{[Path(p).name for p in ratio_info['metrics_dirs']]} = {avg_ratio:.4f}"
    )
    return out


def _run_one_trial(
    timeout: int,
    trial: int,
    archive_dir: Path,
    debug: bool,
    retries: int,
) -> Path:
    result_path = _default_result_file()
    last_err: Exception | None = None

    for attempt in range(1, retries + 2):  # initial try + retries
        size_before = result_path.stat().st_size if result_path.exists() else 0
        node_params = deepcopy(NODE_PARAMS)
        node_params["fast_path_timeout"] = timeout
        bench_params = deepcopy(BENCH_PARAMS)

        Print.heading(
            f"\n=== FPT sweep: timeout={timeout}ms trial={trial} "
            f"attempt={attempt}/{retries + 1} ==="
        )
        try:
            ctx = _make_ctx()
            Bench(ctx).run(bench_params, node_params, debug)
            archived = _archive_summary(
                result_path, archive_dir, timeout, trial, size_before
            )
            Print.info(f"Archived -> {archived}")
            return archived
        except Exception as e:
            last_err = e
            Print.warn(f"Trial failed (timeout={timeout} trial={trial} attempt={attempt}): {e}")
            traceback.print_exc()
            # Best-effort cleanup between retries.
            try:
                ctx = _make_ctx()
                Bench(ctx).kill()
            except Exception:
                pass
            time.sleep(5)

    raise RuntimeError(
        f"All attempts failed for timeout={timeout}ms trial={trial}: {last_err}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep fast_path_timeout with independent fab-remote-equivalent runs")
    parser.add_argument(
        "--timeouts",
        default=",".join(str(t) for t in range(0, 201, 20)),
        help="Comma-separated timeout list in ms",
    )
    parser.add_argument("--trials", type=int, default=3, help="Independent full runs per timeout (default 3)")
    parser.add_argument("--retries", type=int, default=1, help="Extra retries per trial on failure (default 1)")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-debug", action="store_true", help="Disable debug logging for node binaries")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override bench duration seconds (default: value in BENCH_PARAMS)",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Directory for per-trial SUMMARY archives "
             "(default: results/fpt_sweep_offline_A_b100000_h32_cut2_k1_<duration>s)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials that already have a valid archive (with E2E latency + ratio)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=20.0,
        help="Seconds to sleep after each trial before the next one (default 20)",
    )
    args = parser.parse_args()

    timeouts = [int(x.strip()) for x in args.timeouts.split(",") if x.strip()]
    if not timeouts:
        print("No timeouts given", file=sys.stderr)
        return 1
    if args.trials < 1:
        print("--trials must be >= 1", file=sys.stderr)
        return 1

    debug = False if args.no_debug else args.debug
    if args.duration is not None:
        BENCH_PARAMS["duration"] = int(args.duration)
    duration = int(BENCH_PARAMS["duration"])
    archive_dir = Path(
        args.archive_dir
        if args.archive_dir
        else (_script_dir() / "results" / f"fpt_sweep_offline_A_b100000_h32_cut2_k1_{duration}s")
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Run from benchmark/ so relative logs/results paths match fab remote.
    os.chdir(_script_dir())

    failures: list[str] = []
    Print.heading(
        f"Fast-path timeout sweep: {timeouts} x {args.trials} independent runs "
        f"(runs=1, duration={duration}s each), archive={archive_dir}"
    )

    for timeout in timeouts:
        for trial in range(1, args.trials + 1):
            out = archive_dir / f"fpt-{timeout}ms-trial{trial}.txt"
            if args.skip_existing and _archive_is_valid(out, timeout):
                Print.info(f"Skip existing valid archive: {out.name}")
                continue
            try:
                _run_one_trial(timeout, trial, archive_dir, debug, args.retries)
            except Exception as e:
                msg = f"timeout={timeout}ms trial={trial}: {e}"
                failures.append(msg)
                Print.warn(f"Skipping remaining retries; recorded failure: {msg}")
            Print.info(f"Cooling down {args.cooldown}s before next trial...")
            time.sleep(args.cooldown)

    status_path = archive_dir / "sweep_status.txt"
    status_path.write_text(
        "ok\n"
        if not failures
        else "partial_failure\n" + "\n".join(failures) + "\n",
        encoding="utf-8",
    )

    if failures:
        Print.warn(f"Sweep finished with {len(failures)} failure(s). See {status_path}")
        return 1

    Print.heading(f"Sweep complete. Archives in {archive_dir}")
    return 0


if __name__ == "__main__":
    # BenchError printing is handled inside Bench; still exit non-zero on sweep failures.
    try:
        raise SystemExit(main())
    except BenchError as e:
        Print.error(e)
        raise SystemExit(1)
