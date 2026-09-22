#!/usr/bin/env python3
"""Sweep fast_path_timeout on a frozen network setting.

Network: restored from results/fpt_sweep_90s/predicted_plateaus.json.
egress_penalty[0][0] = [0, 0, 0, 80] -> Δ = 10 / 50 / 50 / 90.
node3/0/1 beat slow0 (save 50/40/40); node2 prefers slow0 by 30 ms.

Timeout grid is confined to [0, 200] ms: interior points of predicted
path-decision plateaus ([0,10) [10,50) [50,90) [90,∞)) plus an all-fast
tail (150, 200). k=1 as in that sweep.

Usage (from benchmark/):
  python3 sweep_fast_path_timeout.py
  python3 sweep_fast_path_timeout.py --timeouts 0,5,30,70,150,200 --trials 3
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
from statistics import mean, median

from fabric import Config
from invoke import Context

# Ensure local package imports work when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.cloudlab_remote import CloudLabBench as Bench
from benchmark.utils import BenchError, PathMaker, Print


# Frozen copy of fabfile.remote() params (do not set runs>1).
# Plateau verification: freeze everything except fast_path_timeout.
# RL controller is commented out in cloudlab_remote.py; keep rl_algo unused.
BENCH_PARAMS = {
    "faults": 0,
    "nodes": [4],
    "workers": 1,
    "collocate": True,
    "rate": [40_000],
    "tx_size": 512,
    "duration": 90,
    "runs": 1,
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
    "cut_condition_type": 3,
    # VoteDelay: each node delays its own ConsensusVote by the per-node
    # penalty below (index == node index). Window covers the whole run.
    # Format is window -> region -> per-node (see cloudlab_remote.run).
    "simulate_asynchrony": True,
    "asynchrony_type": [6],
    "asynchrony_start": [0],
    "asynchrony_duration": [3000],
    "affected_nodes": [4],
    "asynchrony_nodes": [4],
    "asynchrony_regions": [["Clem"]],
    "egress_penalty": [[[0, 20, 120, 160]]],
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


def _parse_metrics_ts(ts: str) -> datetime:
    """Parse RFC3339 with nanoseconds (Python only accepts <=6 fraction digits)."""
    head, _, tz = ts.partition("+")
    if "." in head:
        base, frac = head.split(".", 1)
        head = f"{base}.{frac[:6]}"
    return datetime.fromisoformat(f"{head}+{tz}" if tz else head)


def _leader_path_stats_local(n_nodes: int = 4, skip_slots: int = 4) -> dict:
    """Per-leader (slot % n) fast/slow counts and Prepare->commit medians.

    Read from the local node's <home>/logs/metrics-*.log. Leader election
    (SemiParallelRRLeaderElector) is sorted_pks[(slot + view) % n]; keys are
    regenerated per run, so the 'committee' events (pk + consensus_addr) are
    used to map the leader back to a stable node label (10.10.1.X -> nodeX-1).
    Prepare time = first 'prepare' event for the slot; commit time =
    'fast_path'/'slow_path' event (carries the committing view).
    """
    home = _local_metrics_home()
    logs = sorted((home / "logs").glob("metrics-*.log"))
    if not logs:
        return {"error": f"no metrics-*.log under {home / 'logs'}"}
    prep: dict[int, datetime] = {}
    commit: dict[int, tuple[datetime, str, int]] = {}
    committee: dict[str, str] = {}  # pk hex -> consensus ip
    with logs[0].open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = ev.get("details", {})
            et = ev.get("event_type")
            if et == "prepare" and "slot" in d:
                t = _parse_metrics_ts(ev["timestamp"])
                s = int(d["slot"])
                prep[s] = min(prep.get(s, t), t)
            elif et in ("fast_path", "slow_path") and "slot" in d:
                commit[int(d["slot"])] = (
                    _parse_metrics_ts(ev["timestamp"]), et, int(d.get("view", 1))
                )
            elif et == "committee" and "pk" in d:
                committee[d["pk"]] = str(d.get("consensus_addr", "")).split(":")[0]

    sorted_pks = sorted(committee)  # hex sort == byte sort of PublicKey

    def leader_label(slot: int, view: int) -> str:
        if len(sorted_pks) != n_nodes:
            return f"slot_mod_{slot % n_nodes}"
        ip = committee[sorted_pks[(slot + view) % n_nodes]]
        try:
            return f"node{int(ip.rsplit('.', 1)[1]) - 1}"
        except (IndexError, ValueError):
            return ip or f"slot_mod_{slot % n_nodes}"

    per_leader: dict[str, dict] = {}
    for s, (t, et, view) in commit.items():
        if s <= skip_slots or s not in prep:
            continue
        key = leader_label(s, view)
        entry = per_leader.setdefault(
            key, {"fast": 0, "slow": 0, "fast_ms": [], "slow_ms": []}
        )
        ms = (t - prep[s]).total_seconds() * 1000.0
        if et == "fast_path":
            entry["fast"] += 1
            entry["fast_ms"].append(ms)
        else:
            entry["slow"] += 1
            entry["slow_ms"].append(ms)

    out = {
        "source": str(logs[0]),
        "leader_rule": "sorted_pks[(slot+view) % n]",
        "sorted_pk_ips": [committee[pk] for pk in sorted_pks],
        "leaders": {},
    }
    for key in sorted(per_leader):
        e = per_leader[key]
        total = e["fast"] + e["slow"]
        out["leaders"][key] = {
            "n": total,
            "fast_ratio": (e["fast"] / total) if total else None,
            "fast_median_ms": median(e["fast_ms"]) if e["fast_ms"] else None,
            "slow_median_ms": median(e["slow_ms"]) if e["slow_ms"] else None,
        }
    return out


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
    try:
        ratio_info["leader_path_stats"] = _leader_path_stats_local(
            n_nodes=BENCH_PARAMS["nodes"][0]
        )
    except Exception as e:  # best-effort diagnostic only
        ratio_info["leader_path_stats"] = {"error": str(e)}

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
    for key, st in ratio_info["leader_path_stats"].get("leaders", {}).items():
        Print.info(
            f"  {key}: n={st['n']} fast_ratio={st['fast_ratio']:.2f} "
            f"fast_med={st['fast_median_ms']} slow_med={st['slow_median_ms']}"
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
        default="0,10,30,50,70,90,110,130,150,170,180,190,200",
        help="Comma-separated timeout list in ms",
    )
    parser.add_argument("--trials", type=int, default=3, help="Independent full runs per timeout")
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
             "(default: results/fpt_sweep_<duration>s)",
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
        else (_script_dir() / "results" / f"fpt_sweep_{duration}s")
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Record predicted Δ plateaus from the deployed latency_config.
    try:
        sim_dir = str((_script_dir().parent / "simulate_latency").resolve())
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        from latency_config import (
            VOTE_DELAY_MS,
            leader_fast_path_stats,
            predicted_plateaus,
        )

        # Guard: the Δ prediction assumes the deployed VoteDelay penalties.
        penalties = list(NODE_PARAMS["egress_penalty"][0][0])
        expected = [VOTE_DELAY_MS[f"node{i}"] for i in range(len(penalties))]
        if penalties != expected:
            Print.error(
                f"egress_penalty {penalties} != latency_config.VOTE_DELAY_MS {expected}"
            )
            return 1

        pred = {
            "leaders": leader_fast_path_stats(),
            "plateaus": predicted_plateaus(),
            "vote_delay_ms": penalties,
            "timeouts_ms": timeouts,
            "k": NODE_PARAMS["k"],
            "cut_condition_type": NODE_PARAMS["cut_condition_type"],
            "rate": BENCH_PARAMS["rate"],
            "duration": duration,
        }
        pred_path = archive_dir / "predicted_plateaus.json"
        pred_path.write_text(json.dumps(pred, indent=2) + "\n", encoding="utf-8")
        Print.info(f"Wrote predicted Δ plateaus -> {pred_path}")
        for p in pred["plateaus"]:
            hi = "inf" if p["timeout_hi_ms"] is None else p["timeout_hi_ms"]
            Print.info(
                f"  [{p['timeout_lo_ms']}, {hi}): fast={p['fast_leaders']} "
                f"ratio={p['fast_ratio']:.2f}"
            )
    except Exception as e:
        Print.warn(f"Could not record predicted plateaus: {e}")

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
