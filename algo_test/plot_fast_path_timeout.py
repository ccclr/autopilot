#!/usr/bin/env python3
"""Plot fast_path_timeout vs latency from results/fpt_sweep archives.

Usage (from benchmark/):
  python3 plot_fast_path_timeout.py
  python3 plot_fast_path_timeout.py --archive-dir results/fpt_sweep
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _parse_ms(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def parse_archive(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    timeout = _parse_ms(r"Fast path timeout:\s*([\d,]+)", text)
    e2e = _parse_ms(r"End-to-end latency:\s*([\d,]+)\s*ms", text)
    consensus = _parse_ms(r"Consensus latency:\s*([\d,]+)\s*ms", text)
    e2e_tps = _parse_ms(r"End-to-end TPS:\s*([\d,]+)\s*tx/s", text)
    ratio_m = re.search(r"avg_fast_path_ratio=([0-9.]+)", text)
    avg_ratio = float(ratio_m.group(1)) if ratio_m else None
    if timeout is None or e2e is None or consensus is None:
        return None
    return {
        "file": path.name,
        "timeout_ms": int(timeout),
        "e2e_latency_ms": e2e,
        "consensus_latency_ms": consensus,
        "e2e_tps": e2e_tps,
        "avg_fast_path_ratio": avg_ratio,
    }


def _keep_closest_to_median(trials: list[dict], keep: int, key: str = "e2e_latency_ms") -> list[dict]:
    """Drop outliers by keeping the `keep` trials closest to the median of `key`."""
    if keep <= 0 or len(trials) <= keep:
        return list(trials)
    vals = [t[key] for t in trials]
    med = statistics.median(vals)
    ranked = sorted(trials, key=lambda t: (abs(t[key] - med), t[key], t["file"]))
    kept = ranked[:keep]
    # Stable order by filename for reproducible CSV "files" field.
    return sorted(kept, key=lambda t: t["file"])


def aggregate(rows: list[dict], keep_trials: int = 2) -> list[dict]:
    by_timeout: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_timeout[row["timeout_ms"]].append(row)

    summary = []
    for timeout in sorted(by_timeout):
        all_trials = by_timeout[timeout]
        trials = _keep_closest_to_median(all_trials, keep_trials)
        dropped = [t["file"] for t in all_trials if t not in trials]

        def mean_err(vals: list[float]) -> tuple[float, float]:
            mean = statistics.mean(vals)
            if len(vals) == 1:
                return mean, 0.0
            if len(vals) == 2:
                err = abs(vals[0] - vals[1]) / 2.0
            else:
                err = statistics.stdev(vals)
            return mean, err

        e2e_vals = [t["e2e_latency_ms"] for t in trials]
        cons_vals = [t["consensus_latency_ms"] for t in trials]
        tps_vals = [t["e2e_tps"] for t in trials if t["e2e_tps"] is not None]
        e2e_mean, e2e_err = mean_err(e2e_vals)
        cons_mean, cons_err = mean_err(cons_vals)
        tps_mean = statistics.mean(tps_vals) if tps_vals else None
        ratio_vals = [
            t["avg_fast_path_ratio"]
            for t in trials
            if t.get("avg_fast_path_ratio") is not None
        ]
        if ratio_vals:
            ratio_mean, ratio_err = mean_err(ratio_vals)
        else:
            ratio_mean, ratio_err = None, None

        summary.append(
            {
                "timeout_ms": timeout,
                "n_trials": len(trials),
                "n_trials_raw": len(all_trials),
                "files": ",".join(t["file"] for t in trials),
                "dropped_files": ",".join(dropped),
                "e2e_latency_mean_ms": e2e_mean,
                "e2e_latency_err_ms": e2e_err,
                "consensus_latency_mean_ms": cons_mean,
                "consensus_latency_err_ms": cons_err,
                "e2e_tps_mean": tps_mean,
                "avg_fast_path_ratio_mean": ratio_mean,
                "avg_fast_path_ratio_err": ratio_err,
            }
        )
    return summary


def write_csv(summary: list[dict], path: Path) -> None:
    fields = [
        "timeout_ms",
        "n_trials",
        "n_trials_raw",
        "files",
        "dropped_files",
        "e2e_latency_mean_ms",
        "e2e_latency_err_ms",
        "consensus_latency_mean_ms",
        "consensus_latency_err_ms",
        "e2e_tps_mean",
        "avg_fast_path_ratio_mean",
        "avg_fast_path_ratio_err",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def plot_curve(summary: list[dict], out_path: Path) -> None:
    xs = [r["timeout_ms"] for r in summary]
    e2e = [r["e2e_latency_mean_ms"] for r in summary]
    e2e_err = [r["e2e_latency_err_ms"] for r in summary]
    cons = [r["consensus_latency_mean_ms"] for r in summary]
    cons_err = [r["consensus_latency_err_ms"] for r in summary]
    ratios = [r["avg_fast_path_ratio_mean"] for r in summary]
    ratio_err = [r["avg_fast_path_ratio_err"] or 0.0 for r in summary]
    has_ratio = any(v is not None for v in ratios)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        xs,
        e2e,
        yerr=e2e_err,
        marker="o",
        linewidth=2,
        capsize=4,
        label="End-to-end latency",
        color="#1f4e79",
    )
    ax.errorbar(
        xs,
        cons,
        yerr=cons_err,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        capsize=4,
        label="Consensus latency",
        color="#c45911",
    )
    ax.set_xlabel("Fast path timeout (ms)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fast path timeout vs latency")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.5)
    ax.set_ylim(0, 800)

    if has_ratio:
        ax2 = ax.twinx()
        ax2.errorbar(
            xs,
            [v if v is not None else float("nan") for v in ratios],
            yerr=ratio_err,
            marker="^",
            linewidth=1.5,
            linestyle=":",
            capsize=3,
            label="Avg fast_path_ratio",
            color="#2e7d32",
        )
        ax2.set_ylabel("Avg fast_path_ratio")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    else:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot FPT sweep latency curve")
    parser.add_argument(
        "--archive-dir",
        default=str(_script_dir() / "results" / "fpt_sweep_90s"),
        help="Directory with fpt-*ms-trial*.txt archives",
    )
    parser.add_argument(
        "--out",
        default=str(_script_dir() / "plots" / "fast_path_timeout_latency.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output summary CSV path (default: <archive-dir>/summary.csv)",
    )
    parser.add_argument(
        "--keep-trials",
        type=int,
        default=3,
        help=(
            "Per timeout, keep this many trials closest to the E2E-latency median "
            "(drop the rest as outliers), then average. Use 0 to keep all."
        ),
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    files = sorted(archive_dir.glob("fpt-*ms-trial*.txt"))
    if not files:
        print(f"No archive files in {archive_dir}")
        return 1

    rows = []
    for path in files:
        parsed = parse_archive(path)
        if parsed is None:
            print(f"Skip unparseable: {path.name}")
            continue
        rows.append(parsed)

    if not rows:
        print("No parseable trial archives")
        return 1

    summary = aggregate(rows, keep_trials=args.keep_trials)
    csv_path = Path(args.csv) if args.csv else (archive_dir / "summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary, csv_path)

    out_path = Path(args.out)
    plot_curve(summary, out_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_path}")
    print(
        f"Aggregation: keep_trials={args.keep_trials} "
        f"(0 means keep all; else keep closest-to-median by E2E)"
    )
    print("timeout_ms  e2e_mean±err  consensus_mean±err  n(kept/raw)  dropped")
    for r in summary:
        dropped = r.get("dropped_files") or "-"
        print(
            f"{r['timeout_ms']:>5}      "
            f"{r['e2e_latency_mean_ms']:.0f}±{r['e2e_latency_err_ms']:.0f}        "
            f"{r['consensus_latency_mean_ms']:.0f}±{r['consensus_latency_err_ms']:.0f}             "
            f"{r['n_trials']}/{r['n_trials_raw']}  {dropped}"
        )

    best = min(summary, key=lambda r: r["e2e_latency_mean_ms"])
    print(
        f"Lowest E2E latency: {best['e2e_latency_mean_ms']:.0f} ms "
        f"at fast_path_timeout={best['timeout_ms']} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
