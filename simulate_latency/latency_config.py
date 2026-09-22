# latency_config.py
#
# Restored from results/fpt_sweep_90s/predicted_plateaus.json.
# Cross edges 30+30=60; node0↔node1 45+45=90; node2↔node3 35+35=70.
# VoteDelay only on node3 (80 ms). Δ = 10 / 50 / 90 (gaps 40). node2 prefers slow0.
#
#   leader  vote arrivals (ms)   T_2f+1   T_n   Δ    fast vs slow0
#   node3   60, 60, 70           60        70   10    70 vs 120  (save  50)
#   node0   60, 90, 140          90       140   50   140 vs 180  (save  40)
#   node1   60, 90, 140          90       140   50   140 vs 180  (save  40)
#   node2   60, 60, 150          60       150   90   150 vs 120  (worse 30)
#
# Plateaus: [0,10) none; [10,50) n3; [50,90) n3+n0+n1; [90,∞) all.
# Sweep interiors: 0, 5, 30, 70, 150, 200.

NODES = {
    "node0": "10.10.1.1",
    "node1": "10.10.1.2",
    "node2": "10.10.1.3",
    "node3": "10.10.1.4",
}

# One-way netem delay in milliseconds. RTT = 2 × entry.
# Cross edges 30+30=60; node0↔node1 45+45=90; node2↔node3 35+35=70.
LATENCY = {
    "node0": {
        "node1": 10,
        "node2": 10,
        "node3": 10,
    },
    "node1": {
        "node0": 10,
        "node2": 10,
        "node3": 10,
    },
    "node2": {
        "node0": 10,
        "node1": 10,
        "node3": 10,
    },
    "node3": {
        "node0": 10,
        "node1": 10,
        "node2": 10,
    },
}


# Per-voter ConsensusVote delay (ms) injected via asynchrony_type=6 (VoteDelay).
# Index order == node index. Mirrors NODE_PARAMS['egress_penalty'][0][0] in
# benchmark/sweep_fast_path_timeout.py; keep the two in sync.
# p = [0, 0, 0, 80] -> Δ = 10 / 50 / 50 / 90 (gaps 40).
VOTE_DELAY_MS = {
    "node0": 0,
    "node1": 20,
    "node2": 120,
    "node3": 160,
}


def pairwise_rtt_ms():
    """RTT[src][dst] = one-way(src→dst) + one-way(dst→src)."""
    rtt = {src: {} for src in NODES}
    for src in NODES:
        for dst in NODES:
            if src == dst:
                continue
            rtt[src][dst] = LATENCY[src][dst] + LATENCY[dst][src]
    return rtt


def leader_fast_path_stats():
    """Per-leader 2f+1 / n vote times and Δ = T_n - T_{2f+1}.

    n=4, f=1: Slow QC needs self + 2 remotes (2nd-fastest remote vote).
    Fast QC needs all 3 remotes (slowest remote vote). Confirm ≈ same as 2f+1.
    Remote vote arrival = RTT(leader, voter) + VOTE_DELAY_MS[voter].
    """
    rtt = pairwise_rtt_ms()
    stats = []
    for leader in NODES:
        remotes = sorted(
            rtt[leader][v] + VOTE_DELAY_MS.get(v, 0) for v in NODES if v != leader
        )
        t_2f1 = remotes[1]
        t_n = remotes[2]
        stats.append(
            {
                "leader": leader,
                "remote_rtts_ms": remotes,  # RTT + voter VoteDelay
                "vote_delay_ms": VOTE_DELAY_MS.get(leader, 0),
                "t_2f1_ms": t_2f1,
                "t_n_ms": t_n,
                "delta_ms": t_n - t_2f1,
                "t_confirm_ms": t_2f1,
                "fast_better_than_slow0": t_n < t_2f1 + t_2f1,
            }
        )
    stats.sort(key=lambda x: x["delta_ms"])
    return stats


def predicted_plateaus():
    """Timeout intervals on which the set of fast-path leaders is constant."""
    stats = leader_fast_path_stats()
    # Dedupe: equal Δ (e.g. leader0/leader1 under VoteDelay) flip together.
    thresholds = sorted({0, *(s["delta_ms"] for s in stats)})
    plateaus = []
    for i, lo in enumerate(thresholds):
        hi = thresholds[i + 1] if i + 1 < len(thresholds) else None
        fast = [s["leader"] for s in stats if s["delta_ms"] <= lo]
        plateaus.append(
            {
                "timeout_lo_ms": lo,
                "timeout_hi_ms": hi,
                "fast_leaders": fast,
                "fast_ratio": len(fast) / len(stats),
            }
        )
    return plateaus
