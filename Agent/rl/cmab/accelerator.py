"""
Training accelerator: prune the action space with golden rules.

The independent RTT timeout filter reverses the upper-cap rule: it keeps
catalog timeouts strictly above the same covering grid value. Either rule can
be enabled separately; both together trigger the zero-timeout fallback.

Fast-path timeout
-----------------
Fast path needs every vote. After a leader has a quorum (2f+1) it still
waits `fast_path_timeout` for the last vote.
  timeout >= Δ_i → leader i takes the fast path.
Δ_i is that leader's gap from the quorum vote to the last vote.
The RTT cap is ceil(max_i Δ_i) plus a 10 ms margin. Discrete search keeps
every catalog timeout up to the first grid value that is at least that cap.
For caps of 62 ms and 120 ms on [0, 100, 200, 300], the retained timeouts
are [0, 100] and [0, 100, 200].

The master is the replica that launched `fab remote`, not a fixed node
index. It probes the ICMP RTT full matrix and publishes a hint
{timeout_cap, detect_epoch, apply_epoch} to every node. Followers only
read the hint. The cap is applied at apply_epoch (detect_epoch +
apply_delay), e.g. probe at epoch 100, apply at 105.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TIMEOUT_KEY = "fast_path_timeout"
_BENCH_DIR = Path(__file__).resolve().parents[3] / "benchmark"
HINT_NAME = ".accelerator.json"
APPLY_DELAY_EPOCHS = 5
# Extra milliseconds above ceil(max Δ). Covers ping noise and a short RTT underestimate.
TIMEOUT_CAP_MARGIN_MS = 10
# Discrete bound is the first catalog timeout at or above the raw RTT cap.
TIMEOUT_GRID_SAFETY_FACTOR = 1


def _remaining_dims(arms: Iterable[str]) -> dict[str, list[str]]:
    dims: dict[str, set[str]] = {}
    for arm in arms:
        for part in arm.split(","):
            key, _, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if key:
                dims.setdefault(key, set()).add(value)

    def _sort_key(v: str):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    return {k: sorted(vs, key=_sort_key) for k, vs in dims.items()}


def _arm_timeout(arm: str) -> float:
    for part in arm.split(","):
        key, _, value = part.partition("=")
        if key.strip() == _TIMEOUT_KEY:
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def zero_timeout_arms(arms: Iterable[str]) -> list[str]:
    """Preserve other parameters and deduplicate zero-timeout fallback arms."""
    result = []
    for arm in arms:
        parts = [
            "fast_path_timeout=0" if part.partition("=")[0].strip() == _TIMEOUT_KEY else part
            for part in arm.split(",")
        ]
        result.append(",".join(parts))
    return list(dict.fromkeys(result))


def grid_timeout_upper_bound(
    cap_ms: float,
    timeouts: Iterable[float],
    factor: float = TIMEOUT_GRID_SAFETY_FACTOR,
) -> Optional[float]:
    """Smallest catalog timeout that is at least ``factor × cap``.

    Returns the largest catalog timeout when the cap is above the grid.
    """
    target = float(factor) * float(cap_ms)
    grid = sorted({float(timeout) for timeout in timeouts if math.isfinite(float(timeout))})
    if not grid:
        return None
    for timeout in grid:
        if timeout >= target:
            return timeout
    return grid[-1]


def max_fast_path_delta_ms(matrix: np.ndarray) -> Optional[float]:
    """Smallest gap (ms) that lets every leader take the fast path.

    Δ_i = last vote − quorum vote, using ICMP RTT. A vote arrives after
    propose-out + vote-back, so the delay is the ping RTT, not RTT/2.
    The quorum is the same 2N/3+1 stake threshold as the protocol.
    Returns None unless every leader has a finite RTT to every node.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return None
    n = int(matrix.shape[0])
    # Same stake rule as Committee::quorum_threshold. If that quorum is
    # already every node, the extra wait is 0.
    quorum_index = min((2 * n) // 3 + 1, n) - 1
    deltas: list[float] = []
    for i in range(n):
        delays: list[float] = []
        for j in range(n):
            rtt = matrix[i, j]
            if j == i:
                delays.append(0.0)
            elif np.isfinite(rtt):
                delays.append(float(rtt))
            else:
                return None
        delays.sort()
        deltas.append(delays[-1] - delays[quorum_index])
    return float(max(deltas))


def fast_path_timeout_cap_ms(delta_ms: float, margin_ms: float = TIMEOUT_CAP_MARGIN_MS) -> float:
    """Integer timeout that covers every leader, plus `margin_ms`.

    110.2 ms becomes ceil(110.2) + 10 = 121.
    """
    return float(math.ceil(float(delta_ms)) + float(margin_ms))


_GCP_CREDENTIALS_NAME = "bright-meridian-486618-b3-a76d90aff895.json"


def _prepare_gcp_credentials() -> None:
    """Point ADC at the service-account file before any GCP client starts.

    ``~/.bashrc`` returns immediately in a non-interactive probe, so the
    export has to happen in this process.
    """
    cred = Path.home() / _GCP_CREDENTIALS_NAME
    if cred.is_file():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred)
    else:
        logger.warning("ACCELERATOR GCP credentials file missing: %s", cred)


def _ensure_system_fabric() -> None:
    """Let the venv trainer import the system fabric used by `fab`."""
    try:
        import fabric  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "/usr/bin/python3",
                "-c",
                "import fabric, os; print(os.path.dirname(os.path.dirname(fabric.__file__)))",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return
    if out and out not in sys.path:
        sys.path.append(out)


def _fabfile():
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")
    import site
    for p in (site.getusersitepackages(), "/usr/lib/python3/dist-packages", str(_BENCH_DIR)):
        if p and p not in sys.path:
            sys.path.append(p)
    _ensure_system_fabric()
    import fabfile as mod

    return mod


class TrainingAccelerator:
    def __init__(
        self,
        period: int = 100,
        apply_delay: int = APPLY_DELAY_EPOCHS,
        is_master: bool = False,
        hint_path: Optional[str | Path] = None,
        enable_timeout_cap: bool = True,
        enable_rtt_timeout_filter: bool = False,
    ):
        self.enable_timeout_cap = bool(enable_timeout_cap)
        self.enable_rtt_timeout_filter = bool(enable_rtt_timeout_filter)
        self.period = max(1, int(period))
        self.apply_delay = max(1, int(apply_delay))
        self.is_master = bool(is_master)
        self.hint_path = Path(hint_path) if hint_path else Path.home() / HINT_NAME
        self._applied_cap: Optional[float] = None
        self._lock = threading.Lock()
        self._last_detect_epoch: Optional[int] = None
        self._probe_thread: Optional[threading.Thread] = None

    @property
    def timeout_cap(self) -> Optional[float]:
        with self._lock:
            return self._applied_cap

    def on_epoch(self, epoch: Optional[int]) -> None:
        if epoch is None:
            return
        epoch = int(epoch)
        if self.is_master:
            self._maybe_probe(epoch)
        self._apply_hint(epoch)

    def stop(self) -> None:
        thread = self._probe_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._probe_thread = None

    def _maybe_probe(self, epoch: int) -> None:
        if epoch <= 0 or epoch % self.period != 0:
            return
        if self._last_detect_epoch == epoch:
            return
        thread = self._probe_thread
        if thread is not None and thread.is_alive():
            return
        self._last_detect_epoch = epoch
        self._probe_thread = threading.Thread(
            target=self._probe,
            args=(epoch,),
            name="accelerator-probe",
            daemon=True,
        )
        self._probe_thread.start()
        logger.info(
            "ACCELERATOR master probe started detect_epoch=%d apply_epoch=%d",
            epoch,
            epoch + self.apply_delay,
        )

    def _probe(self, detect_epoch: int) -> None:
        cwd = os.getcwd()
        try:
            os.chdir(_BENCH_DIR)
            _prepare_gcp_credentials()
            fab = _fabfile()
            matrix = fab.collect_latency_matrix(quiet=True)
            delta_ms = max_fast_path_delta_ms(matrix)
            if delta_ms is None:
                logger.warning("ACCELERATOR no valid Δ for every leader; skip publish")
                return
            cap = fast_path_timeout_cap_ms(delta_ms)
            hint = {
                "timeout_cap": cap,
                "detect_epoch": detect_epoch,
                "apply_epoch": detect_epoch + self.apply_delay,
            }
            fab.publish_accelerator_hint(hint, str(self.hint_path))
            logger.info("ACCELERATOR published %s", hint)
        except Exception:
            logger.exception("ACCELERATOR probe failed")
        finally:
            os.chdir(cwd)

    def _apply_hint(self, epoch: int) -> None:
        hint = self._read_hint()
        if hint is None:
            return
        apply_epoch = hint.get("apply_epoch")
        cap = hint.get("timeout_cap")
        if apply_epoch is None or cap is None:
            return
        try:
            apply_epoch = int(apply_epoch)
            cap = float(cap)
        except (TypeError, ValueError):
            return
        if epoch < apply_epoch:
            return
        with self._lock:
            if self._applied_cap != cap:
                logger.info(
                    "ACCELERATOR apply cap=%s at epoch=%d (scheduled=%d)",
                    cap,
                    epoch,
                    apply_epoch,
                )
            self._applied_cap = cap

    def _read_hint(self) -> Optional[dict]:
        if not self.hint_path.exists():
            return None
        try:
            raw = self.hint_path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def covering_timeout(self, timeouts: Iterable[float] | None = None) -> Optional[float]:
        """First catalog timeout that is at least the raw cap.

        With catalog values ``[0, 100, 200, 300]``, a 62 ms cap returns 100
        and a 120 ms cap returns 200.
        """
        cap = self.timeout_cap
        if cap is None:
            return None
        if timeouts is None:
            return float(TIMEOUT_GRID_SAFETY_FACTOR) * float(cap)
        return grid_timeout_upper_bound(cap, timeouts)

    def _filter_timeout_cap(self, arms: Iterable[str]) -> list[str]:
        """Keep catalog arms whose timeout is at most the raw-cap grid bound."""
        arms = list(arms)
        cap = self.timeout_cap
        if cap is None:
            return arms
        timeouts = [_arm_timeout(a) for a in arms]
        limit = grid_timeout_upper_bound(cap, timeouts)
        if limit is None:
            return arms
        kept = [a for a, t in zip(arms, timeouts) if t <= limit]
        if not kept:
            smallest = min(timeouts)
            kept = [a for a, t in zip(arms, timeouts) if t == smallest]
            logger.warning(
                "ACCELERATOR cap=%.1f grid_hi=%.1f is below every catalog timeout; "
                "keeping smallest timeout %.1f",
                cap,
                limit,
                smallest,
            )
        if len(kept) < len(arms):
            logger.info(
                "ACCELERATOR prune timeout_cap=%.1f grid_hi=%.1f arms %d -> %d remaining=%s",
                cap,
                limit,
                len(arms),
                len(kept),
                _remaining_dims(kept),
            )
        return kept

    def filter_arms(self, arms: Iterable[str]) -> list[str]:
        """Keep <= grid bound for accelerator, > bound for the reverse filter."""
        original = list(arms)
        kept = self._filter_timeout_cap(original) if self.enable_timeout_cap else original
        if self.enable_rtt_timeout_filter:
            # Calculate from the full catalog so both switches share a boundary.
            limit = self.covering_timeout(_arm_timeout(a) for a in original)
            if limit is not None:
                filtered = [a for a in kept if _arm_timeout(a) > limit]
                if len(filtered) != len(kept):
                    logger.info(
                        "RTT_TIMEOUT_FILTER grid_lo=%.1f arms %d -> %d remaining=%s",
                        limit, len(kept), len(filtered), _remaining_dims(filtered),
                    )
                if not filtered and original:
                    filtered = [a for a in original if _arm_timeout(a) == 0.0]
                    if not filtered:
                        filtered = zero_timeout_arms(original)
                    logger.warning(
                        "RTT_TIMEOUT_FILTER no candidates remain; falling back to "
                        "fast_path_timeout=0 (%d arms)", len(filtered),
                    )
                kept = filtered
        return kept
