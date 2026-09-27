"""Collapse a factor once every level has enough clean samples.

The check is not on an epoch clock. A factor stays in the search until each
of its levels (for cut and timeout, each cut×timeout cell) has
``min_samples`` clean observations. Levels whose median latency is within
``margin_ms`` of the best level then collapse to that best level. A later
sample can reopen a level when the same rule, rerun on everything collected
so far, no longer collapses it.
"""

from __future__ import annotations

import logging
from statistics import median_low

logger = logging.getLogger(__name__)

FACTORS = (
    "batch_size",
    "header_size",
    "cut_condition_type",
    "fast_path_timeout",
    "k",
)
MARGINAL_FACTORS = ("batch_size", "header_size", "k")
OUTLIER_RATIO = 2.0


def parse_arm(arm: str) -> dict[str, str]:
    values = {}
    for part in str(arm).split(","):
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            values[key] = value
    return values


def latency_ms(reward: float) -> float:
    """Invert global_reward = 1000 / (latency_ms + 1)."""
    return 1000.0 / float(reward) - 1.0


def _sort_key(value: str):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def clean_latencies(samples: list[tuple[str, float]], ratio: float = OUTLIER_RATIO):
    """Drop per-arm latencies outside [center/ratio, center*ratio]."""
    grouped: dict[str, list[float]] = {}
    for arm, reward in samples:
        if reward <= 0:
            continue
        grouped.setdefault(arm, []).append(latency_ms(reward))

    cleaned: list[tuple[dict[str, str], float]] = []
    for arm, latencies in grouped.items():
        parsed = parse_arm(arm)
        if len(latencies) < 2:
            cleaned.extend((parsed, latency) for latency in latencies)
            continue
        center = median_low(latencies)
        if center <= 0:
            cleaned.extend((parsed, latency) for latency in latencies)
            continue
        low, high = center / ratio, center * ratio
        cleaned.extend(
            (parsed, latency)
            for latency in latencies
            if low <= latency <= high
        )
    return cleaned


def collapse_levels(
    groups: dict[str, list[float]],
    min_samples: int,
    margin_ms: float,
) -> set[str] | None:
    """Keep the best level and any level worse than it by more than margin_ms.

    Returns None when some level still has fewer than min_samples.
    """
    if not groups or any(len(values) < min_samples for values in groups.values()):
        return None
    medians = {level: median_low(values) for level, values in groups.items()}
    best = min(medians, key=lambda level: (medians[level], _sort_key(level)))
    best_latency = medians[best]
    keep = {best}
    for level, latency in medians.items():
        if latency - best_latency > margin_ms:
            keep.add(level)
    return keep


def _stratified_keep(
    cleaned: list[tuple[dict[str, str], float]],
    row_key: str,
    col_key: str,
    rows: set[str],
    cols: set[str],
    min_samples: int,
    margin_ms: float,
) -> set[str] | None:
    """Drop a column only when every row collapses it away.

    None means at least one row is not ready yet.
    """
    if not rows or not cols:
        return set(cols)
    needed: set[str] = set()
    for row in rows:
        groups = {col: [] for col in cols}
        for parsed, latency in cleaned:
            if parsed.get(row_key) == row and parsed.get(col_key) in groups:
                groups[parsed[col_key]].append(latency)
        keep = collapse_levels(groups, min_samples, margin_ms)
        if keep is None:
            return None
        needed |= keep
    return needed


class FactorFreezer:
    def __init__(self, min_samples: int = 20, margin_ms: float = 30.0):
        self.min_samples = max(1, int(min_samples))
        self.margin_ms = float(margin_ms)
        if self.margin_ms <= 0:
            raise ValueError("margin_ms must be positive")
        self._samples: list[tuple[str, float]] = []
        self._signature: tuple | None = None

    def observe(self, arm: str, reward: float) -> None:
        if reward <= 0:
            return
        self._samples.append((str(arm), float(reward)))

    def _cleaned(self):
        return clean_latencies(self._samples)

    def _levels(self, arms: list[str]) -> dict[str, set[str]]:
        levels = {factor: set() for factor in FACTORS}
        for arm in arms:
            parsed = parse_arm(arm)
            for factor in FACTORS:
                value = parsed.get(factor)
                if value is not None:
                    levels[factor].add(value)
        return levels

    def kept_levels(self, arms: list[str]) -> dict[str, set[str]]:
        cleaned = self._cleaned()
        levels = self._levels(arms)
        kept = {factor: set(values) for factor, values in levels.items()}

        for factor in MARGINAL_FACTORS:
            groups = {level: [] for level in levels[factor]}
            for parsed, latency in cleaned:
                level = parsed.get(factor)
                if level in groups:
                    groups[level].append(latency)
            decision = collapse_levels(groups, self.min_samples, self.margin_ms)
            if decision is not None:
                kept[factor] = decision

        timeout_decision = _stratified_keep(
            cleaned,
            "cut_condition_type",
            "fast_path_timeout",
            levels["cut_condition_type"],
            levels["fast_path_timeout"],
            self.min_samples,
            self.margin_ms,
        )
        if timeout_decision is not None:
            kept["fast_path_timeout"] = timeout_decision

        cut_decision = _stratified_keep(
            cleaned,
            "fast_path_timeout",
            "cut_condition_type",
            levels["fast_path_timeout"],
            levels["cut_condition_type"],
            self.min_samples,
            self.margin_ms,
        )
        if cut_decision is not None:
            kept["cut_condition_type"] = cut_decision
        return kept

    def filter_arms(self, arms: list[str]) -> list[str]:
        kept = self.kept_levels(arms)
        filtered = [
            arm
            for arm in arms
            if all(parse_arm(arm).get(factor) in kept[factor] for factor in FACTORS)
        ]
        if not filtered:
            logger.error("FACTOR_FREEZE refused to drop every arm; catalog unchanged")
            return list(arms)
        signature = tuple(
            (factor, tuple(sorted(kept[factor], key=_sort_key)))
            for factor in FACTORS
        )
        if signature != self._signature:
            self._signature = signature
            logger.info(
                "FACTOR_FREEZE min_samples=%d margin_ms=%.1f kept=%s arms=%d",
                self.min_samples,
                self.margin_ms,
                {factor: list(values) for factor, values in signature},
                len(filtered),
            )
        return filtered

    def cover_arm(self, arms: list[str]) -> str | None:
        """Pick an arm that fills the level furthest below min_samples.

        Returns None once every level the freeze rule needs is full, so the
        policy can exploit.
        """
        if not arms:
            return None
        cleaned = self._cleaned()
        deficits: dict[tuple, int] = {}
        levels = self._levels(arms)
        for factor in MARGINAL_FACTORS:
            for level in levels[factor]:
                count = sum(1 for parsed, _latency in cleaned if parsed.get(factor) == level)
                if count < self.min_samples:
                    deficits[(factor, level)] = self.min_samples - count
        pairs = {
            (parsed["cut_condition_type"], parsed["fast_path_timeout"])
            for parsed in (parse_arm(arm) for arm in arms)
            if "cut_condition_type" in parsed and "fast_path_timeout" in parsed
        }
        for cut, timeout in pairs:
            count = sum(
                1
                for parsed, _latency in cleaned
                if parsed.get("cut_condition_type") == cut
                and parsed.get("fast_path_timeout") == timeout
            )
            if count < self.min_samples:
                deficits[("cut_timeout", cut, timeout)] = self.min_samples - count
        if not deficits:
            return None

        best_score = -1
        chosen = None
        for arm in arms:
            parsed = parse_arm(arm)
            score = 0
            for factor in MARGINAL_FACTORS:
                score += deficits.get((factor, parsed.get(factor)), 0)
            score += deficits.get(
                (
                    "cut_timeout",
                    parsed.get("cut_condition_type"),
                    parsed.get("fast_path_timeout"),
                ),
                0,
            )
            if score > best_score:
                best_score = score
                chosen = arm
        if chosen is not None:
            logger.info(
                "FACTOR_FREEZE_COVER arm=%s open_levels=%d",
                chosen,
                len(deficits),
            )
        return chosen
