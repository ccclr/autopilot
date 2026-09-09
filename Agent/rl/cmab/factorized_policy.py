"""Factorized (hierarchical) CMAB reward model.

Global-reward RF scores a complete 5-tuple as one arm. This policy instead
predicts

    r̂(s, a) = Σ_i f_i(s, a_i) + Σ_{(i,j)∈E} f_ij(s, a_i, a_j)

with hierarchical residual fits: each main-effect forest is trained on the
leftover of the previous factor (so intercepts are not counted five times),
then pair forests fit the remaining residual. Default pair edges are
(cut, k) and (timeout, k). Cut is one-hot (not treated as ordered 2 < 3 < 4).
Selection enumerates the catalog and picks the arm with the highest
predicted reward. Exploration is the same outer-bootstrap Thompson
sampling as the global RF: each update refits on a bootstrap of the
replay window, then select_arm is greedy on that posterior draw.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .policy import CMABPolicy

logger = logging.getLogger(__name__)

FACTOR_KEYS = (
    "batch_size",
    "header_size",
    "cut_condition_type",
    "fast_path_timeout",
    "k",
)
DEFAULT_PAIR_EDGES = (
    ("cut_condition_type", "k"),
    ("fast_path_timeout", "k"),
)
CATEGORICAL_FACTORS = frozenset({"cut_condition_type"})
REWARD_MODEL = "factorized"


def parse_arm_factors(arm) -> dict[str, int]:
    """Parse `key=value,...` arm ids used by ArmCatalog."""
    if isinstance(arm, dict):
        return {key: int(arm[key]) for key in FACTOR_KEYS}
    if not isinstance(arm, str):
        raise ValueError(f"Unsupported CMAB arm type: {type(arm)!r}")
    values: dict[str, int] = {}
    for part in arm.split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = int(value)
    missing = [key for key in FACTOR_KEYS if key not in values]
    if missing:
        raise ValueError(f"Arm is missing factors {missing}: {arm!r}")
    return {key: values[key] for key in FACTOR_KEYS}


def _one_hot(value: int, catalog: list[int]) -> np.ndarray:
    encoded = np.zeros(len(catalog), dtype=np.float32)
    try:
        encoded[catalog.index(int(value))] = 1.0
    except ValueError:
        pass
    return encoded


class FactorizedCMABPolicy(CMABPolicy):
    REWARD_MODEL = REWARD_MODEL

    def __init__(
        self,
        arms,
        feature_dim=5,
        policy_name="rf_ts",
        epsilon=0,
        min_samples_to_fit=4,
        fit_every=1,
        n_estimators=20,
        uses_context=True,
        random_state: int = 0,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0,
        replay_window: int = 200,
        action_encoding: str = "numeric",
        pair_edges: Iterable[tuple[str, str]] | None = None,
        include_batch_header_pair: bool = False,
        max_depth: int = 4,
        min_samples_leaf: int = 4,
        **_ignored,
    ):
        super().__init__(
            arms=arms,
            feature_dim=feature_dim,
            policy_name=policy_name,
            epsilon=epsilon,
            min_samples_to_fit=min_samples_to_fit,
            fit_every=fit_every,
            n_estimators=n_estimators,
            uses_context=uses_context,
            random_state=random_state,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            replay_window=replay_window,
            # Parent encoding is unused for scoring; keep numeric so a
            # factorized checkpoint cannot be loaded as one_hot by mistake.
            action_encoding="numeric",
        )
        edges = list(pair_edges) if pair_edges is not None else list(DEFAULT_PAIR_EDGES)
        if include_batch_header_pair:
            edges.append(("batch_size", "header_size"))
        self._pair_edges = tuple(edges)
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._min_samples_leaf = int(min_samples_leaf)
        self._factor_catalogs = self._build_factor_catalogs(self._arms)
        self._cache_arm_feature_blocks()
        self._records: list[dict[str, Any]] = []
        self._main: dict[str, RandomForestRegressor] = {
            key: self._new_forest(offset)
            for offset, key in enumerate(FACTOR_KEYS, start=1)
        }
        self._pair: dict[tuple[str, str], RandomForestRegressor] = {
            edge: self._new_forest(10 + offset)
            for offset, edge in enumerate(self._pair_edges, start=1)
        }

    def _new_forest(self, offset: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            min_samples_split=2,
            bootstrap=True,
            n_jobs=1,
            verbose=0,
            random_state=self._random_state + int(offset),
        )

    @staticmethod
    def _build_factor_catalogs(arms) -> dict[str, list[int]]:
        catalogs: dict[str, set[int]] = {key: set() for key in FACTOR_KEYS}
        for arm in arms:
            factors = parse_arm_factors(arm)
            for key, value in factors.items():
                catalogs[key].add(int(value))
        return {key: sorted(values) for key, values in catalogs.items()}

    def _context_vec(self, context) -> np.ndarray:
        if not self._uses_context or context is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(context, dtype=np.float32).flatten()

    def _factor_vec(self, key: str, value: int) -> np.ndarray:
        if key in CATEGORICAL_FACTORS:
            return _one_hot(value, self._factor_catalogs[key])
        return np.asarray([float(value)], dtype=np.float32)

    def _cache_arm_feature_blocks(self) -> None:
        self._arm_factors = [parse_arm_factors(arm) for arm in self._arms]
        self._main_factor_mat = {
            key: np.stack(
                [self._factor_vec(key, factors[key]) for factors in self._arm_factors]
            )
            for key in FACTOR_KEYS
        }
        self._pair_factor_mat = {
            edge: np.concatenate(
                [self._main_factor_mat[edge[0]], self._main_factor_mat[edge[1]]],
                axis=1,
            )
            for edge in self._pair_edges
        }

    def _main_row(self, context, factors: dict[str, int], key: str) -> np.ndarray:
        return np.concatenate(
            [self._context_vec(context), self._factor_vec(key, factors[key])]
        )

    def _pair_row(
        self, context, factors: dict[str, int], edge: tuple[str, str]
    ) -> np.ndarray:
        left, right = edge
        return np.concatenate(
            [
                self._context_vec(context),
                self._factor_vec(left, factors[left]),
                self._factor_vec(right, factors[right]),
            ]
        )

    @staticmethod
    def _tile_context(ctx: np.ndarray, n: int) -> np.ndarray:
        if ctx.size == 0:
            return np.zeros((n, 0), dtype=np.float32)
        return np.broadcast_to(ctx.astype(np.float32, copy=False), (n, ctx.size)).copy()

    @staticmethod
    def _run_forest_predicts(
        tasks: list[tuple[RandomForestRegressor, np.ndarray]],
    ) -> list[np.ndarray]:
        if not tasks:
            return []
        if len(tasks) == 1:
            model, features = tasks[0]
            return [np.asarray(model.predict(features), dtype=np.float64)]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(model.predict, features) for model, features in tasks]
            return [np.asarray(fut.result(), dtype=np.float64) for fut in futures]

    def _model_ready(self, model: RandomForestRegressor) -> bool:
        estimators = getattr(model, "estimators_", None)
        return bool(estimators)

    def _models_ready(self) -> bool:
        if not self._is_fitted or len(self._y) < self._min_samples_to_fit:
            return False
        return all(self._model_ready(model) for model in self._main.values())

    def _predict_components(
        self, context, factors: dict[str, int]
    ) -> tuple[float, float, dict[str, float]]:
        mains: dict[str, float] = {}
        main_sum = 0.0
        for key, model in self._main.items():
            if not self._model_ready(model):
                mains[key] = 0.0
                continue
            row = self._main_row(context, factors, key)[None, :]
            value = float(model.predict(row)[0])
            mains[key] = value
            main_sum += value
        pair_sum = 0.0
        for edge, model in self._pair.items():
            if not self._model_ready(model):
                continue
            row = self._pair_row(context, factors, edge)[None, :]
            pair_sum += float(model.predict(row)[0])
        return main_sum, pair_sum, mains

    def _predict_reward(self, context, arm) -> float:
        factors = parse_arm_factors(arm)
        main_sum, pair_sum, _ = self._predict_components(context, factors)
        return main_sum + pair_sum

    def _predict_rewards_all_arms(self, context) -> np.ndarray:
        n_arms = len(self._arms)
        scores = np.zeros(n_arms, dtype=np.float64)
        ctx = self._tile_context(self._context_vec(context), n_arms)
        tasks: list[tuple[RandomForestRegressor, np.ndarray]] = []
        for key, model in self._main.items():
            if not self._model_ready(model):
                continue
            tasks.append((model, np.concatenate([ctx, self._main_factor_mat[key]], axis=1)))
        for edge, model in self._pair.items():
            if not self._model_ready(model):
                continue
            tasks.append((model, np.concatenate([ctx, self._pair_factor_mat[edge]], axis=1)))
        for preds in self._run_forest_predicts(tasks):
            scores += preds
        return scores

    def _predict_rewards_rows(
        self, contexts: list, factors_list: list[dict[str, int]]
    ) -> np.ndarray:
        n = len(factors_list)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        scores = np.zeros(n, dtype=np.float64)
        context_rows = np.stack([self._context_vec(ctx) for ctx in contexts])
        tasks: list[tuple[RandomForestRegressor, np.ndarray]] = []
        for key, model in self._main.items():
            if not self._model_ready(model):
                continue
            factor_rows = np.stack(
                [self._factor_vec(key, factors[key]) for factors in factors_list]
            )
            tasks.append((model, np.concatenate([context_rows, factor_rows], axis=1)))
        for edge, model in self._pair.items():
            if not self._model_ready(model):
                continue
            left, right = edge
            factor_rows = np.stack(
                [
                    np.concatenate(
                        [
                            self._factor_vec(left, factors[left]),
                            self._factor_vec(right, factors[right]),
                        ]
                    )
                    for factors in factors_list
                ]
            )
            tasks.append((model, np.concatenate([context_rows, factor_rows], axis=1)))
        for preds in self._run_forest_predicts(tasks):
            scores += preds
        return scores

    def _window_arm_counts_from_replay(self):
        window_counts = {arm: 0 for arm in self._arms}
        recent = self._records[-self._replay_window :]
        matched_rows = 0
        for record in recent:
            arm = record.get("arm")
            if arm in window_counts:
                window_counts[arm] += 1
                matched_rows += 1
        return window_counts, matched_rows

    def select_arm(self, context, shared_seed_hex: str | None = None):
        non_learned = self._select_non_learned_arm(shared_seed_hex)
        if non_learned is not None:
            return non_learned

        if not self._models_ready():
            idx = self._shared_rng_index(len(self._arms), shared_seed_hex, "cold_start")
            logger.info("Factorized model not ready: selecting exploration arm idx=%d.", idx)
            return self._arms[idx]

        current_epsilon = self._current_epsilon()
        epsilon_probe = self._shared_rng_uniform(shared_seed_hex, "epsilon_explore")
        if epsilon_probe < current_epsilon:
            window_counts, matched_rows = self._window_arm_counts_from_replay()
            min_count = min(window_counts.values())
            least_tried = sorted(
                [arm for arm, cnt in window_counts.items() if cnt == min_count]
            )
            idx = self._shared_rng_index(len(least_tried), shared_seed_hex, "epsilon_random_arm")
            chosen = least_tried[idx]
            logger.info(
                "EPSILON_EXPLORATION eps=%.6f probe=%.6f update=%d -> least-tried arm "
                "(window=%d, min_count=%d, candidates=%d) idx=%d arm=%s",
                current_epsilon,
                epsilon_probe,
                self._update_count,
                matched_rows,
                min_count,
                len(least_tried),
                idx,
                chosen,
            )
            return chosen

        logger.info("INFERENCE_START reward_model=factorized")
        rewards = self._predict_rewards_all_arms(context)
        logger.info("INFERENCE_DONE")

        def _fmt_list(values, idxs):
            return "[" + ", ".join(f"{values[i]:.6f}" for i in idxs) + "]"

        top5_idx = np.argsort(rewards)[::-1][:5]
        logger.info("================================================================================")
        logger.info("SELECT ARM - Factorized Prediction Analysis")
        logger.info("Context: %s", np.asarray(context))
        logger.info("  Mean predictions (top 5 arms): %s", _fmt_list(rewards, top5_idx))
        logger.info("Top 5 arms by factorized bootstrap-TS mean:")
        for rank, idx in enumerate(top5_idx, start=1):
            logger.info(
                "  #%d: arm=%s, mean=%.6f",
                rank,
                self._arms[idx],
                rewards[idx],
            )

        max_pred = rewards.max()
        max_indices = np.flatnonzero(np.isclose(rewards, max_pred))
        if len(max_indices) > 1 and shared_seed_hex:
            tie_pick_pos = self._shared_rng_index(len(max_indices), shared_seed_hex, "tie_break")
            chosen_idx = int(max_indices[tie_pick_pos])
        else:
            chosen_idx = int(max_indices.min())
        chosen = self._arms[chosen_idx]

        topk = min(self._monitor_topk, len(self._arms))
        topk_idx = np.argsort(rewards)[::-1][:topk]
        topk_arms = [self._arms[i] for i in topk_idx]
        topk_mean = [float(rewards[i]) for i in topk_idx]
        logger.info("MONITOR_TOP_ARMS k=%d arms=%s means=%s", topk, topk_arms, topk_mean)
        logger.info(
            "✓ SELECTED ARM: %s (mean_prediction=%.6f)",
            chosen,
            max_pred,
        )
        logger.info("================================================================================")
        return chosen

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
            factors = parse_arm_factors(arm)
            context_list = (
                self._context_vec(context).tolist() if context is not None else []
            )
            self._records.append(
                {
                    "arm": arm,
                    "context": context_list,
                    "factors": factors,
                    "reward": float(reward),
                }
            )
            self._y.append(float(reward))
            self.arm_counts[arm] += 1
            self._recent_decisions.append(arm)
            logger.info(
                "TRAIN_SAMPLE idx=%d arm=%s reward=%.6f context=%s factors=%s",
                len(self._y),
                arm,
                float(reward),
                np.asarray(context) if context is not None else None,
                factors,
            )

        self._update_count += 1

        if len(self._y) >= self._min_samples_to_fit and self._update_count % self._fit_every == 0:
            replay_length = min(len(self._records), self._replay_window)
            recent = self._records[-replay_length:]
            bootstrapped_idx = self._bootstrap_indices(replay_length, shared_seed_hex)
            sampled = [recent[int(i)] for i in bootstrapped_idx]
            training_y = np.asarray([row["reward"] for row in sampled], dtype=np.float64)
            preview_n = min(20, replay_length)
            unique_count = int(np.unique(bootstrapped_idx).size) if replay_length > 0 else 0
            idx_digest = hashlib.sha256(bootstrapped_idx.tobytes()).hexdigest()[:16]
            logger.info(
                "MONITOR_BOOTSTRAP update=%d replay=%d unique=%d idx_head(%d)=%s idx_hash=%s seed=%s",
                self._update_count,
                replay_length,
                unique_count,
                preview_n,
                bootstrapped_idx[:preview_n].tolist(),
                idx_digest,
                shared_seed_hex if shared_seed_hex is not None else f"fallback:{self._random_state}",
            )

            residual = training_y.copy()
            # Hierarchical residual: each main forest sees the leftover of
            # earlier factors, then pair forests fit what mains cannot explain.
            for key in FACTOR_KEYS:
                features = np.stack(
                    [self._main_row(row["context"], row["factors"], key) for row in sampled]
                )
                self._main[key].fit(features, residual)
                residual = residual - self._main[key].predict(features)

            for edge in self._pair_edges:
                features = np.stack(
                    [self._pair_row(row["context"], row["factors"], edge) for row in sampled]
                )
                self._pair[edge].fit(features, residual)
                residual = residual - self._pair[edge].predict(features)

            self._is_fitted = True
            fitted_rewards = self._predict_rewards_rows(
                [row["context"] for row in self._records],
                [row["factors"] for row in self._records],
            )
            mse = float(np.mean((fitted_rewards - np.asarray(self._y)) ** 2))
            logger.info(
                "Factorized RF updated: samples=%d, replay=%d, bootstrap=%d, MSE=%.6f",
                len(self._y),
                replay_length,
                len(bootstrapped_idx),
                mse,
            )

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "reward_model": REWARD_MODEL,
                "action_encoding": self.action_encoding,
                "arms": list(self._arms),
                "records": self._records,
                "y": self._y,
                "is_fitted": self._is_fitted,
                "update_count": self._update_count,
                "arm_counts": self.arm_counts,
                "main_models": self._main,
                "pair_models": {
                    f"{left}|{right}": model
                    for (left, right), model in self._pair.items()
                },
                "pair_edges": list(self._pair_edges),
                "factor_catalogs": self._factor_catalogs,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        reward_model = data.get("reward_model")
        if reward_model != REWARD_MODEL:
            raise ValueError(
                "CMAB checkpoint reward model mismatch: "
                f"checkpoint={reward_model or 'global'}, configured=factorized"
            )
        checkpoint_arms = data.get("arms")
        if (
            checkpoint_arms is not None
            and tuple(checkpoint_arms) != tuple(self._arms)
        ):
            raise ValueError("CMAB checkpoint arm catalog does not match current arms")
        self._records = list(data.get("records") or [])
        self._y = list(data.get("y") or [row["reward"] for row in self._records])
        self._is_fitted = bool(data.get("is_fitted"))
        self._update_count = int(data.get("update_count") or 0)
        self.arm_counts = data.get("arm_counts") or {
            arm: 0 for arm in self._arms
        }
        self._main = data["main_models"]
        loaded_pairs = data.get("pair_models") or {}
        restored: dict[tuple[str, str], RandomForestRegressor] = {}
        for edge in self._pair_edges:
            key = f"{edge[0]}|{edge[1]}"
            if key in loaded_pairs:
                restored[edge] = loaded_pairs[key]
        self._pair = restored or self._pair
        if data.get("factor_catalogs"):
            self._factor_catalogs = data["factor_catalogs"]
            self._cache_arm_feature_blocks()
        for record in self._records[-self._replay_window :]:
            self._recent_decisions.append(record["arm"])
