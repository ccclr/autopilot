"""
Gaussian Process Bayesian Optimization with mixed action space.

- Discrete dims: batch_size, header_size, cut_condition_type, k (enumerated)
- Continuous dim: fast_path_timeout_ms; per-base UCB maximized continuously
  with multi-start L-BFGS-B from Uniform random seeds (timeout_grid_size = n_restarts)

API stays CMAB-compatible: select_arm / update / save / load.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .mixed_space import (
    TIMEOUT_KEY_INDEX,
    MixedActionSpace,
    decode_arm_values,
)

logger = logging.getLogger(__name__)


class GPBOPolicy:
    def __init__(
        self,
        arms: Iterable[str] | None = None,
        feature_dim: int = 5,
        policy_name: str = "gp_bo",
        kappa: float = 2.0,
        min_samples_to_fit: int = 5,
        fit_every: int = 1,
        uses_context: bool = True,
        random_state: int = 0,
        replay_window: int = 200,
        n_restarts_optimizer: int = 2,
        mixed_space: MixedActionSpace | None = None,
        timeout_grid_size: int = 31,
    ):
        self.mixed_space = mixed_space
        # Number of multi-start seeds for continuous timeout UCB maximization.
        self._timeout_grid_size = max(2, int(timeout_grid_size))
        if mixed_space is not None:
            self._bases = mixed_space.list_bases()
            # Placeholder arms only used for bookkeeping / legacy list APIs.
            self._arms = mixed_space.list_placeholder_arms()
        else:
            self._bases = None
            self._arms = list(arms or [])
        self.policy_name = policy_name
        self.kappa = float(kappa)
        self._min_samples_to_fit = int(min_samples_to_fit)
        self._fit_every = max(1, int(fit_every))
        self._uses_context = bool(uses_context)
        self.uses_context = self._uses_context
        self._random_state = int(random_state)
        self._replay_window = max(1, int(replay_window))
        self._feature_dim = int(feature_dim)
        self._n_restarts_optimizer = int(n_restarts_optimizer)

        self._gp: Optional[GaussianProcessRegressor] = None
        self._is_fitted = False
        self._update_count = 0
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self.arm_counts: dict[str, int] = {}
        self._base_counts = {base: 0 for base in (self._bases or [])}
        self._recent_decisions: deque = deque(maxlen=self._replay_window)
        self._monitor_topk = 5
        self._input_dim: Optional[int] = None
        self._disc_lo: Optional[np.ndarray] = None
        self._disc_hi: Optional[np.ndarray] = None
        self._init_discrete_norm_bounds()

    def _init_discrete_norm_bounds(self):
        if self.mixed_space is not None:
            mat = np.asarray(self._bases, dtype=np.float64)
            # columns: batch, header, cut, k — log1p normalize
            logged = np.log1p(np.maximum(mat, 0.0))
            self._disc_lo = logged.min(axis=0)
            self._disc_hi = logged.max(axis=0)
        elif self._arms:
            mat = np.stack([self._arm_to_vector(a) for a in self._arms], axis=0)
            logged = np.log1p(np.maximum(mat, 0.0))
            self._disc_lo = logged.min(axis=0)
            self._disc_hi = logged.max(axis=0)

    def _stable_int(self, *parts) -> int:
        payload = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _shared_rng_index(self, size: int, seed_hex: str | None, label: str) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        digest = hashlib.sha256(f"{seed_hex}|{label}".encode("utf-8")).digest()
        seed_u64 = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed_u64)
        return int(rng.integers(size))

    def _shared_rng_uniform(self, seed_hex: str | None, label: str) -> float:
        seed_u64 = self._stable_int(
            seed_hex if seed_hex is not None else f"fallback:{self._random_state}",
            label,
            self._update_count,
            len(self._y),
        )
        rng = np.random.default_rng(seed_u64)
        return float(rng.random())

    def _arm_to_vector(self, arm) -> np.ndarray:
        if isinstance(arm, str) and "=" in arm:
            return np.asarray(decode_arm_values(arm), dtype=np.float32)
        return np.asarray(arm, dtype=np.float32).flatten()

    def _normalize_action_vector(self, arm_vec: np.ndarray) -> np.ndarray:
        """
        Normalize [batch, header, cut, timeout, k] -> unit features.
        timeout uses continuous bounds when mixed_space is set.
        """
        v = np.asarray(arm_vec, dtype=np.float64).flatten()
        if self.mixed_space is not None:
            # discrete order in vector: b,h,c,timeout,k
            disc = np.array([v[0], v[1], v[2], v[4]], dtype=np.float64)
            logged = np.log1p(np.maximum(disc, 0.0))
            denom = np.where(self._disc_hi > self._disc_lo, self._disc_hi - self._disc_lo, 1.0)
            disc_n = (logged - self._disc_lo) / denom
            timeout_n = self.mixed_space.normalize_timeout(v[TIMEOUT_KEY_INDEX])
            # Keep feature order aligned with arm vector for readability.
            out = np.array(
                [disc_n[0], disc_n[1], disc_n[2], timeout_n, disc_n[3]],
                dtype=np.float32,
            )
            return out

        logged = np.log1p(np.maximum(v, 0.0))
        denom = np.where(self._disc_hi > self._disc_lo, self._disc_hi - self._disc_lo, 1.0)
        return ((logged - self._disc_lo) / denom).astype(np.float32)

    def _feature_row(self, context, arm) -> np.ndarray:
        arm_vec = self._normalize_action_vector(self._arm_to_vector(arm))
        if self._uses_context and context is not None:
            ctx_vec = np.asarray(context, dtype=np.float32).flatten()
            return np.concatenate([ctx_vec, arm_vec])
        return arm_vec

    def _make_gp(self, n_features: int) -> GaussianProcessRegressor:
        length_scale = np.ones(n_features, dtype=np.float64)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
            alpha=1e-6,
        )

    def _cold_start_arm(self, shared_seed_hex: str | None) -> str:
        if self.mixed_space is not None:
            counts = [self._base_counts[b] for b in self._bases]
            min_count = min(counts) if counts else 0
            candidates = [b for b, c in zip(self._bases, counts) if c == min_count]
            bidx = self._shared_rng_index(len(candidates), shared_seed_hex, "gp_bo_base")
            base = candidates[bidx]
            # Continuous uniform timeout in [lo, hi].
            u = self._shared_rng_uniform(shared_seed_hex, "gp_bo_timeout_u")
            timeout = self.mixed_space.denormalize_timeout(u)
            chosen = self.mixed_space.make_arm(base, timeout)
            logger.info(
                "GP-BO mixed cold start: base=%s timeout=%.1f arm=%s",
                base,
                timeout,
                chosen,
            )
            return chosen

        counts = [self.arm_counts.get(a, 0) for a in self._arms]
        min_count = min(counts) if counts else 0
        candidates = [a for a, c in zip(self._arms, counts) if c == min_count]
        idx = self._shared_rng_index(len(candidates), shared_seed_hex, "gp_bo_cold_start")
        chosen = candidates[idx]
        logger.info(
            "GP-BO cold start: selecting least-tried arm idx=%d arm=%s (min_count=%d)",
            idx,
            chosen,
            min_count,
        )
        return chosen

    def _predict_ucb(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(features, return_std=True)
        mean = np.asarray(mean, dtype=np.float64)
        std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
        ucb = mean + self.kappa * std
        return mean, std, ucb

    def _feature_row_from_parts(
        self, context, base: tuple[int, int, int, int], timeout_ms: float
    ) -> np.ndarray:
        b, h, c, k = base
        arm_vec = np.asarray([b, h, c, float(timeout_ms), k], dtype=np.float32)
        arm_n = self._normalize_action_vector(arm_vec)
        if self._uses_context and context is not None:
            ctx_vec = np.asarray(context, dtype=np.float32).flatten()
            return np.concatenate([ctx_vec, arm_n])
        return arm_n

    def _ucb_at_timeout(self, context, base, timeout_ms: float) -> tuple[float, float, float, str]:
        # Evaluate UCB on the continuous timeout; round only when emitting the arm id.
        feat = np.asarray(
            [self._feature_row_from_parts(context, base, timeout_ms)],
            dtype=np.float32,
        )
        mean, std, ucb = self._predict_ucb(feat)
        arm = self.mixed_space.make_arm(base, timeout_ms)
        return float(ucb[0]), float(mean[0]), float(std[0]), arm

    def _maximize_ucb_timeout_for_base(
        self, context, base
    ) -> tuple[float, float, float, float, str]:
        """
        Continuously maximize UCB over timeout in [lo, hi] for a fixed discrete base.

        Returns (timeout_ms, ucb, mean, std, arm).
        """
        lo = self.mixed_space.timeout_lo
        hi = self.mixed_space.timeout_search_hi
        if hi <= lo:
            ucb, mean, std, arm = self._ucb_at_timeout(context, base, lo)
            return lo, ucb, mean, std, arm

        starts = np.asarray(
            [
                lo,
                hi,
                *[
                    float(x)
                    for x in np.random.default_rng(
                        self._stable_int(
                            "gp_bo_restarts",
                            base,
                            self._update_count,
                            len(self._y),
                        )
                    ).uniform(lo, hi, size=self._timeout_grid_size)
                ],
            ],
            dtype=np.float64,
        )
        # Evaluate random / endpoint seeds, then refine the best few with L-BFGS-B.
        seed_feats = np.asarray(
            [self._feature_row_from_parts(context, base, float(t)) for t in starts],
            dtype=np.float32,
        )
        seed_mean, seed_std, seed_ucb = self._predict_ucb(seed_feats)
        best_idx = int(np.argmax(seed_ucb))
        best_timeout = float(starts[best_idx])
        best_ucb = float(seed_ucb[best_idx])
        best_mean = float(seed_mean[best_idx])
        best_std = float(seed_std[best_idx])
        best_arm = self.mixed_space.make_arm(base, best_timeout)

        n_refine = min(5, len(starts))
        refine_order = np.argsort(seed_ucb)[::-1][:n_refine]
        # Always include endpoints among refine starts.
        endpoint_idxs = [0, len(starts) - 1]
        refine_idxs = list(dict.fromkeys([*refine_order.tolist(), *endpoint_idxs]))

        def neg_ucb(x: np.ndarray) -> float:
            ucb, _, _, _ = self._ucb_at_timeout(context, base, float(x[0]))
            return -ucb

        for idx in refine_idxs:
            t0 = float(starts[idx])
            try:
                res = minimize(
                    neg_ucb,
                    x0=np.asarray([t0], dtype=np.float64),
                    method="L-BFGS-B",
                    bounds=[(lo, hi)],
                    options={"maxiter": 64, "ftol": 1e-9},
                )
            except Exception as e:
                logger.debug("L-BFGS-B failed for base=%s t0=%.3f: %s", base, t0, e)
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

    def _select_arm_mixed_continuous(self, context, shared_seed_hex: str | None = None):
        logger.info(
            "GPBO_INFERENCE_START mixed=continuous_ucb bases=%d restarts=%d",
            len(self._bases),
            self._timeout_grid_size,
        )
        results = []
        try:
            for base in self._bases:
                timeout, ucb, mean, std, arm = self._maximize_ucb_timeout_for_base(
                    context, base
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
            logger.warning("GP continuous UCB max failed (%s); falling back to cold start", e)
            return self._cold_start_arm(shared_seed_hex)
        logger.info("GPBO_INFERENCE_DONE")

        results.sort(key=lambda r: (-r["ucb"], r["timeout"]))
        topk = min(self._monitor_topk, len(results))
        logger.info("================================================================================")
        logger.info("GP-BO SELECT ARM (UCB, continuous timeout)")
        logger.info("Context: %s", np.asarray(context))
        logger.info(
            "kappa=%.3f samples=%d timeout_restarts=%d bounds=[%.1f, %.1f]",
            self.kappa,
            len(self._y),
            self._timeout_grid_size,
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
            pick = self._shared_rng_index(len(tied), shared_seed_hex, "gp_bo_tie")
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

    def select_arm(self, context, shared_seed_hex: str | None = None, epoch: int | None = None):
        if (
            not self._is_fitted
            or self._gp is None
            or len(self._y) < self._min_samples_to_fit
        ):
            return self._cold_start_arm(shared_seed_hex)

        if self.mixed_space is not None:
            return self._select_arm_mixed_continuous(context, shared_seed_hex)

        candidates = list(self._arms)
        features = np.asarray(
            [self._feature_row(context, arm) for arm in candidates],
            dtype=np.float32,
        )
        logger.info(
            "GPBO_INFERENCE_START candidates=%d mixed=%s",
            len(candidates),
            False,
        )
        try:
            mean, std, ucb = self._predict_ucb(features)
        except Exception as e:
            logger.warning("GP predict failed (%s); falling back to cold start", e)
            return self._cold_start_arm(shared_seed_hex)
        logger.info("GPBO_INFERENCE_DONE")

        topk = min(self._monitor_topk, len(candidates))
        top_ucb_idx = np.argsort(ucb)[::-1][:topk]
        logger.info("================================================================================")
        logger.info("GP-BO SELECT ARM (UCB)")
        logger.info("Context: %s", np.asarray(context))
        logger.info("kappa=%.3f samples=%d", self.kappa, len(self._y))
        for rank, idx in enumerate(top_ucb_idx, start=1):
            logger.info(
                "  #%d: arm=%s mean=%.6f std=%.6f ucb=%.6f",
                rank,
                candidates[idx],
                mean[idx],
                std[idx],
                ucb[idx],
            )

        max_ucb = ucb.max()
        max_indices = np.flatnonzero(np.isclose(ucb, max_ucb))
        if len(max_indices) > 1 and shared_seed_hex:
            pick = self._shared_rng_index(len(max_indices), shared_seed_hex, "gp_bo_tie")
            chosen_idx = int(max_indices[pick])
        else:
            chosen_idx = int(max_indices.min())
        chosen = candidates[chosen_idx]
        logger.info(
            "✓ SELECTED ARM: %s (ucb=%.6f mean=%.6f std=%.6f)",
            chosen,
            ucb[chosen_idx],
            mean[chosen_idx],
            std[chosen_idx],
        )
        logger.info("================================================================================")
        return chosen

    def _bump_counts(self, arm: str):
        self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
        if self.mixed_space is not None:
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
            self._X.append(np.asarray(feature_row, dtype=np.float32))
            self._y.append(float(reward))
            self._bump_counts(arm)
            self._recent_decisions.append(arm)
            logger.info(
                "TRAIN_SAMPLE idx=%d arm=%s reward=%.6f context=%s feature=%s",
                len(self._y),
                arm,
                float(reward),
                np.asarray(context) if context is not None else None,
                np.asarray(feature_row),
            )

        self._update_count += 1
        if len(self._y) < self._min_samples_to_fit:
            logger.info(
                "GP-BO warmup samples=%d/%d; skip fit",
                len(self._y),
                self._min_samples_to_fit,
            )
            return
        if self._update_count % self._fit_every != 0:
            return

        replay_length = min(len(self._y), self._replay_window)
        X = np.asarray(self._X[-replay_length:], dtype=np.float64)
        y = np.asarray(self._y[-replay_length:], dtype=np.float64)
        self._input_dim = X.shape[1]
        self._gp = self._make_gp(X.shape[1])
        try:
            self._gp.fit(X, y)
            self._is_fitted = True
            pred = self._gp.predict(X)
            mse = float(np.mean((pred - y) ** 2))
            logger.info(
                "GP Updated: samples=%d replay=%d MSE=%.6f kernel=%s",
                len(self._y),
                replay_length,
                mse,
                self._gp.kernel_,
            )
        except Exception as e:
            self._is_fitted = False
            logger.warning("GP fit failed: %s", e)

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "algo": "gp_bo_mixed",
                "gp": self._gp,
                "X": self._X,
                "y": self._y,
                "is_fitted": self._is_fitted,
                "update_count": self._update_count,
                "arm_counts": self.arm_counts,
                "base_counts": {str(k): v for k, v in self._base_counts.items()},
                "kappa": self.kappa,
                "input_dim": self._input_dim,
                "timeout_grid_size": self._timeout_grid_size,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        algo = data.get("algo")
        if algo not in (None, "gp_bo", "gp_bo_mixed"):
            logger.warning("Loading checkpoint with algo=%s into GPBOPolicy", algo)
        self._gp = data.get("gp")
        self._X = data.get("X", [])
        self._y = data.get("y", [])
        self._is_fitted = bool(data.get("is_fitted", False))
        self._update_count = int(data.get("update_count", 0))
        self.arm_counts = dict(data.get("arm_counts") or {})
        saved_bases = data.get("base_counts") or {}
        for base in self._base_counts:
            self._base_counts[base] = int(saved_bases.get(str(base), 0))
        if "kappa" in data:
            self.kappa = float(data["kappa"])
        if "timeout_grid_size" in data:
            self._timeout_grid_size = int(data["timeout_grid_size"])
        self._input_dim = data.get("input_dim")
        if self._gp is not None and self._X:
            self._is_fitted = True
