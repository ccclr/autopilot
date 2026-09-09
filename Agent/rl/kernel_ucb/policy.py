"""
KernelUCB for Autopilot mixed action space.

Classical KernelUCB (Valko / Chu style):
  μ(x) = k(X,x)^T (K + λI)^{-1} y
  σ²(x) = k(x,x) - k(X,x)^T (K + λI)^{-1} k(X,x)
  UCB(x) = μ(x) + β σ(x)

Action = discrete base × continuous fast_path_timeout.
Timeout is chosen by bounded continuous maximization of UCB
(multi-start L-BFGS-B from Uniform random seeds), not by a timeout grid.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from gp_bo.mixed_space import (
    TIMEOUT_KEY_INDEX,
    MixedActionSpace,
    decode_arm_values,
)

logger = logging.getLogger(__name__)


class KernelUCBPolicy:
    def __init__(
        self,
        mixed_space: MixedActionSpace,
        policy_name: str = "kernel_ucb",
        beta: float = 2.0,
        lambda_reg: float = 1e-2,
        length_scale: float = 0.4,
        signal_var: float = 1.0,
        min_samples_to_fit: int = 5,
        uses_context: bool = True,
        random_state: int = 0,
        replay_window: int = 200,
        n_restarts: int = 8,
    ):
        self.mixed_space = mixed_space
        self._bases = mixed_space.list_bases()
        self._arms = mixed_space.list_placeholder_arms()
        self.policy_name = policy_name
        self.beta = float(beta)
        # Keep `kappa` alias so CMAB-style callers / logs stay compatible.
        self.kappa = self.beta
        self.lambda_reg = float(lambda_reg)
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self._min_samples_to_fit = int(min_samples_to_fit)
        self._uses_context = bool(uses_context)
        self.uses_context = self._uses_context
        self._random_state = int(random_state)
        self._replay_window = max(1, int(replay_window))
        self._n_restarts = max(2, int(n_restarts))

        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._is_ready = False
        self._update_count = 0
        # After resume (or explicit start_warmup), force this many cold-start
        # selections even if the KernelUCB model is already fitted.
        self._forced_cold_start_remaining = 0
        self.arm_counts: dict[str, int] = {}
        self._base_counts = {base: 0 for base in self._bases}
        self._recent_decisions: deque = deque(maxlen=self._replay_window)
        self._monitor_topk = 5

        # Cached model on the replay window.
        self._X_mat: Optional[np.ndarray] = None
        self._y_vec: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._chol: Optional[np.ndarray] = None  # chol(K + λI)
        self._alpha: Optional[np.ndarray] = None  # (K+λI)^{-1} y_norm

        mat = np.asarray(self._bases, dtype=np.float64)
        logged = np.log1p(np.maximum(mat, 0.0))
        self._disc_lo = logged.min(axis=0)
        self._disc_hi = logged.max(axis=0)

    def _stable_int(self, *parts) -> int:
        payload = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _shared_rng(self, seed_hex: str | None, label: str) -> np.random.Generator:
        seed_u64 = self._stable_int(
            seed_hex if seed_hex is not None else f"fallback:{self._random_state}",
            label,
            self._update_count,
            len(self._y),
        )
        return np.random.default_rng(seed_u64)

    def _shared_rng_index(self, size: int, seed_hex: str | None, label: str) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        return int(self._shared_rng(seed_hex, label).integers(size))

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Isotropic RBF: k(a,b) = σ² exp(-||a-b||² / (2 ℓ²))."""
        A = np.atleast_2d(A).astype(np.float64)
        B = np.atleast_2d(B).astype(np.float64)
        # (n,m) squared distances
        aa = np.sum(A * A, axis=1)[:, None]
        bb = np.sum(B * B, axis=1)[None, :]
        dist2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
        return self.signal_var * np.exp(-0.5 * dist2 / (self.length_scale**2))

    def _normalize_action_vector(self, arm_vec: np.ndarray) -> np.ndarray:
        v = np.asarray(arm_vec, dtype=np.float64).flatten()
        disc = np.array([v[0], v[1], v[2], v[4]], dtype=np.float64)
        logged = np.log1p(np.maximum(disc, 0.0))
        denom = np.where(self._disc_hi > self._disc_lo, self._disc_hi - self._disc_lo, 1.0)
        disc_n = (logged - self._disc_lo) / denom
        timeout_n = self.mixed_space.normalize_timeout(v[TIMEOUT_KEY_INDEX])
        return np.array(
            [disc_n[0], disc_n[1], disc_n[2], timeout_n, disc_n[3]],
            dtype=np.float64,
        )

    def _feature_row(self, context, arm) -> np.ndarray:
        if isinstance(arm, str) and "=" in arm:
            arm_vec = np.asarray(decode_arm_values(arm), dtype=np.float64)
        else:
            arm_vec = np.asarray(arm, dtype=np.float64).flatten()
        arm_n = self._normalize_action_vector(arm_vec)
        if self._uses_context and context is not None:
            ctx_vec = np.asarray(context, dtype=np.float64).flatten()
            return np.concatenate([ctx_vec, arm_n])
        return arm_n

    def _feature_row_from_parts(
        self, context, base: tuple[int, int, int, int], timeout_ms: float
    ) -> np.ndarray:
        b, h, c, k = base
        arm_vec = np.asarray([b, h, c, float(timeout_ms), k], dtype=np.float64)
        arm_n = self._normalize_action_vector(arm_vec)
        if self._uses_context and context is not None:
            ctx_vec = np.asarray(context, dtype=np.float64).flatten()
            return np.concatenate([ctx_vec, arm_n])
        return arm_n

    def _rebuild_model(self):
        replay = min(len(self._y), self._replay_window)
        X = np.asarray(self._X[-replay:], dtype=np.float64)
        y = np.asarray(self._y[-replay:], dtype=np.float64)
        self._X_mat = X
        self._y_mean = float(y.mean()) if len(y) else 0.0
        self._y_std = float(y.std()) if len(y) > 1 else 1.0
        if self._y_std < 1e-8:
            self._y_std = 1.0
        y_norm = (y - self._y_mean) / self._y_std

        K = self._kernel(X, X)
        K = K + self.lambda_reg * np.eye(K.shape[0])
        try:
            chol = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Jitter and retry.
            K = K + 1e-6 * np.eye(K.shape[0])
            chol = np.linalg.cholesky(K)
        # Solve (K+λI) alpha = y_norm via Cholesky.
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
        self._chol = chol
        self._alpha = alpha
        self._y_vec = y_norm
        self._is_ready = True

    def _predict_ucb(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xq = np.atleast_2d(features).astype(np.float64)
        K_x = self._kernel(self._X_mat, Xq)  # (n, m)
        # μ_norm = K_x^T alpha
        mu_norm = K_x.T @ self._alpha
        # v = chol^{-1} K_x
        v = np.linalg.solve(self._chol, K_x)
        var = np.maximum(self.signal_var - np.sum(v * v, axis=0), 1e-12)
        std_norm = np.sqrt(var)
        # Un-normalize mean; keep std in reward units.
        mean = mu_norm * self._y_std + self._y_mean
        std = std_norm * self._y_std
        ucb = mean + self.beta * std
        return mean, std, ucb

    def _ucb_at_timeout(self, context, base, timeout_ms: float):
        feat = self._feature_row_from_parts(context, base, timeout_ms)
        mean, std, ucb = self._predict_ucb(feat)
        arm = self.mixed_space.make_arm(base, timeout_ms)
        return float(ucb[0]), float(mean[0]), float(std[0]), arm

    def _maximize_ucb_timeout(
        self, context, base, shared_seed_hex: str | None
    ) -> tuple[float, float, float, float, str]:
        lo = self.mixed_space.timeout_lo
        hi = self.mixed_space.timeout_search_hi
        if hi <= lo:
            ucb, mean, std, arm = self._ucb_at_timeout(context, base, lo)
            return lo, ucb, mean, std, arm

        rng = self._shared_rng(shared_seed_hex, f"kernel_ucb_restarts|{base}")
        # Continuous seeds: endpoints + Uniform draws (no timeout lattice).
        starts = [lo, hi]
        starts.extend(float(x) for x in rng.uniform(lo, hi, size=self._n_restarts))

        best_timeout = starts[0]
        best_ucb, best_mean, best_std, best_arm = self._ucb_at_timeout(
            context, base, best_timeout
        )

        def neg_ucb(x: np.ndarray) -> float:
            ucb, _, _, _ = self._ucb_at_timeout(context, base, float(x[0]))
            return -ucb

        for t0 in starts:
            ucb0, mean0, std0, arm0 = self._ucb_at_timeout(context, base, float(t0))
            if ucb0 > best_ucb + 1e-12 or (
                np.isclose(ucb0, best_ucb) and float(t0) < best_timeout
            ):
                best_timeout, best_ucb, best_mean, best_std, best_arm = (
                    float(t0),
                    ucb0,
                    mean0,
                    std0,
                    arm0,
                )
            try:
                res = minimize(
                    neg_ucb,
                    x0=np.asarray([float(t0)], dtype=np.float64),
                    method="L-BFGS-B",
                    bounds=[(lo, hi)],
                    options={"maxiter": 64, "ftol": 1e-9},
                )
            except Exception as e:
                logger.debug("KernelUCB L-BFGS-B failed base=%s t0=%.3f: %s", base, t0, e)
                continue
            t_star = float(np.clip(res.x[0], lo, hi))
            ucb, mean, std, arm = self._ucb_at_timeout(context, base, t_star)
            if ucb > best_ucb + 1e-12 or (
                np.isclose(ucb, best_ucb) and t_star < best_timeout
            ):
                best_timeout, best_ucb, best_mean, best_std, best_arm = (
                    t_star,
                    ucb,
                    mean,
                    std,
                    arm,
                )

        return best_timeout, best_ucb, best_mean, best_std, best_arm

    def start_warmup(self, n: int) -> None:
        """Force `n` cold-start arm selections (used after checkpoint resume)."""
        n = max(0, int(n))
        self._forced_cold_start_remaining = n
        logger.info(
            "KernelUCB warmup armed: %d forced cold-start iteration(s) "
            "(model_ready=%s samples=%d)",
            n,
            self._is_ready,
            len(self._y),
        )

    def _cold_start_arm(self, shared_seed_hex: str | None) -> str:
        counts = [self._base_counts[b] for b in self._bases]
        min_count = min(counts) if counts else 0
        candidates = [b for b, c in zip(self._bases, counts) if c == min_count]
        bidx = self._shared_rng_index(len(candidates), shared_seed_hex, "kernel_ucb_base")
        base = candidates[bidx]
        # Continuous uniform timeout in [lo, hi].
        u = float(self._shared_rng(shared_seed_hex, "kernel_ucb_timeout_u").random())
        timeout = self.mixed_space.denormalize_timeout(u)
        chosen = self.mixed_space.make_arm(base, timeout)
        logger.info(
            "KernelUCB cold start: base=%s timeout=%.1f arm=%s "
            "(forced_remaining=%d samples=%d ready=%s)",
            base,
            timeout,
            chosen,
            self._forced_cold_start_remaining,
            len(self._y),
            self._is_ready,
        )
        return chosen

    def select_arm(self, context, shared_seed_hex: str | None = None, epoch: int | None = None):
        # Post-resume / explicit warmup: explore even if model is already fitted.
        if self._forced_cold_start_remaining > 0:
            self._forced_cold_start_remaining -= 1
            logger.info(
                "KernelUCB warmup select: forced cold-start "
                "(%d remaining after this pick)",
                self._forced_cold_start_remaining,
            )
            return self._cold_start_arm(shared_seed_hex)

        if not self._is_ready or len(self._y) < self._min_samples_to_fit:
            return self._cold_start_arm(shared_seed_hex)

        logger.info(
            "KERNELUCB_INFERENCE_START bases=%d restarts=%d samples=%d",
            len(self._bases),
            self._n_restarts,
            len(self._y),
        )
        results = []
        try:
            for base in self._bases:
                timeout, ucb, mean, std, arm = self._maximize_ucb_timeout(
                    context, base, shared_seed_hex
                )
                results.append(
                    {
                        "base": base,
                        "timeout": timeout,
                        "ucb": ucb,
                        "mean": mean,
                        "std": std,
                        "arm": arm,
                    }
                )
        except Exception as e:
            logger.warning("KernelUCB select failed (%s); cold start", e)
            return self._cold_start_arm(shared_seed_hex)
        logger.info("KERNELUCB_INFERENCE_DONE")

        results.sort(key=lambda r: (-r["ucb"], r["timeout"]))
        topk = min(self._monitor_topk, len(results))
        logger.info("================================================================================")
        logger.info("KernelUCB SELECT ARM (continuous timeout)")
        logger.info("Context: %s", np.asarray(context))
        logger.info(
            "beta=%.3f lambda=%.3g length_scale=%.3f samples=%d bounds=[%.1f, %.1f]",
            self.beta,
            self.lambda_reg,
            self.length_scale,
            len(self._y),
            self.mixed_space.timeout_lo,
            self.mixed_space.timeout_search_hi,
        )
        for rank, row in enumerate(results[:topk], start=1):
            logger.info(
                "  #%d: arm=%s mean=%.6f std=%.6f ucb=%.6f timeout=%.3f",
                rank,
                row["arm"],
                row["mean"],
                row["std"],
                row["ucb"],
                row["timeout"],
            )

        best_ucb = results[0]["ucb"]
        tied = [i for i, r in enumerate(results) if np.isclose(r["ucb"], best_ucb)]
        if len(tied) > 1 and shared_seed_hex:
            pick = self._shared_rng_index(len(tied), shared_seed_hex, "kernel_ucb_tie")
            chosen = results[tied[pick]]
        else:
            chosen = results[tied[0]]

        logger.info(
            "✓ SELECTED ARM: %s (ucb=%.6f mean=%.6f std=%.6f timeout=%.3f)",
            chosen["arm"],
            chosen["ucb"],
            chosen["mean"],
            chosen["std"],
            chosen["timeout"],
        )
        logger.info("================================================================================")
        return chosen["arm"]

    def _bump_counts(self, arm: str):
        self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
        vals = decode_arm_values(arm)
        base = (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[4]))
        if base in self._base_counts:
            self._base_counts[base] += 1

    def update(self, decisions, rewards, contexts=None, shared_seed_hex: str | None = None):
        if contexts is None:
            contexts = [None] * len(decisions)

        for arm, reward, context in zip(decisions, rewards, contexts):
            if float(reward) > 15:
                logger.warning(
                    "DROP_TRAIN_SAMPLE arm=%s reward=%.6f reason=reward_gt_15",
                    arm,
                    float(reward),
                )
                continue
            feature_row = self._feature_row(context, arm)
            self._X.append(np.asarray(feature_row, dtype=np.float64))
            self._y.append(float(reward))
            self._bump_counts(arm)
            self._recent_decisions.append(arm)
            logger.info(
                "TRAIN_SAMPLE idx=%d arm=%s reward=%.6f",
                len(self._y),
                arm,
                float(reward),
            )

        self._update_count += 1
        if len(self._y) < self._min_samples_to_fit:
            logger.info(
                "KernelUCB warmup samples=%d/%d; skip model rebuild",
                len(self._y),
                self._min_samples_to_fit,
            )
            return

        try:
            self._rebuild_model()
            logger.info(
                "KernelUCB Updated: samples=%d replay=%d beta=%.3f",
                len(self._y),
                min(len(self._y), self._replay_window),
                self.beta,
            )
        except Exception as e:
            self._is_ready = False
            logger.warning("KernelUCB rebuild failed: %s", e)

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "algo": "kernel_ucb",
                "X": self._X,
                "y": self._y,
                "is_ready": self._is_ready,
                "update_count": self._update_count,
                "arm_counts": self.arm_counts,
                "base_counts": {str(k): v for k, v in self._base_counts.items()},
                "beta": self.beta,
                "lambda_reg": self.lambda_reg,
                "length_scale": self.length_scale,
                "signal_var": self.signal_var,
                "n_restarts": self._n_restarts,
                "forced_cold_start_remaining": self._forced_cold_start_remaining,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        algo = data.get("algo")
        if algo not in (None, "kernel_ucb", "gp_bo", "gp_bo_mixed"):
            logger.warning("Loading checkpoint with algo=%s into KernelUCBPolicy", algo)
        self._X = data.get("X", [])
        self._y = data.get("y", [])
        self._update_count = int(data.get("update_count", 0))
        self.arm_counts = dict(data.get("arm_counts") or {})
        saved_bases = data.get("base_counts") or {}
        for base in self._base_counts:
            self._base_counts[base] = int(saved_bases.get(str(base), 0))
        if "beta" in data:
            self.beta = float(data["beta"])
            self.kappa = self.beta
        if "lambda_reg" in data:
            self.lambda_reg = float(data["lambda_reg"])
        if "length_scale" in data:
            self.length_scale = float(data["length_scale"])
        if "signal_var" in data:
            self.signal_var = float(data["signal_var"])
        if "n_restarts" in data:
            self._n_restarts = int(data["n_restarts"])
        # Warmup after resume is re-armed by train_kernel_ucb via start_warmup().
        self._forced_cold_start_remaining = 0
        if len(self._y) >= self._min_samples_to_fit and self._X:
            try:
                self._rebuild_model()
            except Exception as e:
                self._is_ready = False
                logger.warning("KernelUCB rebuild on load failed: %s", e)
        else:
            self._is_ready = bool(data.get("is_ready", False))
        logger.info(
            "KernelUCB loaded: samples=%d ready=%s forced_warmup_remaining=%d",
            len(self._y),
            self._is_ready,
            self._forced_cold_start_remaining,
        )
