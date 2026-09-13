"""Combined CMAB reward model: large global RF plus a small residual RF.

The inherited global forest scores a complete arm from state + action
features. A second, low-capacity forest sees only the current state plus
``cut_condition_type`` and ``fast_path_timeout``, and is trained on

    residual = y - GlobalRF(state, full_arm)

using the *current* Global RF after every fit (not a mixture of older
models). Selection uses

    score(arm) = GlobalRF_mean(arm) + PairRF_mean(arm)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .policy import CMABPolicy

logger = logging.getLogger(__name__)

REWARD_MODEL = "combined"


class CombinedCMABPolicy(CMABPolicy):
    PAIRWISE_RESIDUAL_KEYS = ("cut_condition_type", "fast_path_timeout")
    PAIRWISE_FEATURE_SCHEMA = "state_plus_cut_condition_type_fast_path_timeout_v2"
    PAIRWISE_TARGET_SCHEMA = "current_global_training_residual_v1"
    REWARD_MODEL = REWARD_MODEL

    def __init__(
        self,
        arms,
        feature_dim,
        policy_name="rf_ts",
        epsilon=0,
        min_samples_to_fit=0,
        fit_every=1,
        n_estimators=50,
        uses_context=True,
        random_state: int = 0,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0,
        replay_window: int = 200,
        action_encoding: str = "numeric",
        pair_n_estimators: int = 20,
        pair_max_depth: int = 3,
        pair_min_samples_leaf: int = 4,
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
            action_encoding=action_encoding,
        )
        self._pair_rf = RandomForestRegressor(
            n_estimators=int(pair_n_estimators),
            max_depth=int(pair_max_depth),
            min_samples_leaf=int(pair_min_samples_leaf),
            bootstrap=True,
            verbose=0,
            random_state=self._random_state + 1,
        )
        self._pair_X = []
        self._pair_y = []
        self._pair_start_index = 0
        self._pair_rf_is_fitted = False

    def _bootstrap_indices(
        self,
        replay_length: int,
        shared_seed_hex: str | None,
        label: str = "outer_bootstrap",
    ) -> np.ndarray:
        if replay_length <= 0:
            return np.array([], dtype=np.int64)
        if shared_seed_hex is not None:
            seed_u64 = self._stable_int(
                shared_seed_hex, label, self._update_count, replay_length
            )
        else:
            seed_u64 = self._stable_int(
                self._random_state, label, self._update_count, replay_length
            )
        rng = np.random.default_rng(seed_u64)
        return rng.choice(replay_length, replay_length, replace=True)

    @staticmethod
    def _arm_values(arm) -> dict[str, float]:
        if not isinstance(arm, str) or "=" not in arm:
            raise ValueError(
                "Combined residual RF requires named CMAB arms, got "
                f"{arm!r}"
            )
        values = {}
        for part in arm.split(","):
            key, value = part.split("=", 1)
            values[key] = float(value)
        return values

    def _pair_feature_row(self, context, arm) -> np.ndarray:
        if context is None:
            raise ValueError("Combined residual RF requires a state context")
        state = np.asarray(context, dtype=np.float32).flatten()
        values = self._arm_values(arm)
        try:
            pair = np.asarray(
                [values[key] for key in self.PAIRWISE_RESIDUAL_KEYS],
                dtype=np.float32,
            )
        except KeyError as error:
            raise ValueError(
                f"CMAB arm is missing pairwise parameter {error.args[0]!r}: {arm!r}"
            ) from error
        return np.concatenate([state, pair])

    def _pair_feature_matrix(self, context, arms=None) -> np.ndarray:
        candidates = self._arms if arms is None else arms
        return np.asarray(
            [self._pair_feature_row(context, arm) for arm in candidates]
        )

    def _score_arms(self, context):
        features = self._feature_matrix(context)
        all_preds = np.stack(
            [tree.predict(features) for tree in self._rf.estimators_]
        )
        global_mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        mean = global_mean.copy()
        pair_mean = None
        if self._pair_rf_is_fitted:
            pair_features = self._pair_feature_matrix(context, self._arms)
            pair_preds = np.stack(
                [tree.predict(pair_features) for tree in self._pair_rf.estimators_]
            )
            pair_mean = pair_preds.mean(axis=0)
            mean = global_mean + pair_mean
            logger.info(
                "PAIRWISE_RESIDUAL_INFERENCE pair=%s global_top=%.6f "
                "pair_at_global_top=%.6f final_top=%.6f",
                "x".join(self.PAIRWISE_RESIDUAL_KEYS),
                float(global_mean.max()),
                float(pair_mean[int(np.argmax(global_mean))]),
                float(mean.max()),
            )
        return mean, std, global_mean, pair_mean

    def select_arm(
        self,
        context,
        shared_seed_hex: str | None = None,
        epoch: int | None = None,
    ):
        non_learned = self._select_non_learned_arm(shared_seed_hex, epoch=epoch)
        if non_learned is not None:
            return non_learned

        if (
            not self._is_fitted
            or len(self._y) < self._min_samples_to_fit
            or not hasattr(self._rf, "estimators_")
            or len(getattr(self._rf, "estimators_", [])) == 0
        ):
            idx = self._shared_rng_index(len(self._arms), shared_seed_hex, "cold_start")
            logger.info("Model not ready: selecting deterministic exploration arm idx=%d.", idx)
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

        logger.info("INFERENCE_START reward_model=combined")
        mean, std, _global_mean, _pair_mean = self._score_arms(context)
        logger.info("INFERENCE_DONE")

        def _fmt_list(values, idxs):
            return "[" + ", ".join(f"{values[i]:.6f}" for i in idxs) + "]"

        logger.info("================================================================================")
        logger.info("SELECT ARM - Combined Prediction Analysis")
        logger.info("Context: %s", np.asarray(context))
        logger.info("Number of trees in forest: %d", len(self._rf.estimators_))
        logger.info("")
        logger.info("Prediction statistics across all trees:")
        top5_mean_idx = np.argsort(mean)[::-1][:5]
        logger.info("  Mean predictions (top 5 arms): %s", _fmt_list(mean, top5_mean_idx))
        logger.info("  Std predictions (top 5 arms): %s", _fmt_list(std, top5_mean_idx))
        logger.info("")
        logger.info("Top 5 arms by combined mean prediction:")
        for rank, idx in enumerate(top5_mean_idx, start=1):
            logger.info(
                "  #%d: arm=%s, mean=%.6f, std=%.6f",
                rank, self._arms[idx], mean[idx], std[idx]
            )

        max_pred = mean.max()
        max_indices = np.flatnonzero(np.isclose(mean, max_pred))
        if len(max_indices) > 1:
            logger.info("")
            logger.info(
                "  Found %d arms with same max mean prediction (%.6f), deterministically selecting one...",
                len(max_indices), max_pred
            )

        if len(max_indices) > 1 and shared_seed_hex:
            tie_pick_pos = self._shared_rng_index(len(max_indices), shared_seed_hex, "tie_break")
            chosen_idx = int(max_indices[tie_pick_pos])
        else:
            chosen_idx = int(max_indices.min())
        chosen = self._arms[chosen_idx]

        topk = min(self._monitor_topk, len(self._arms))
        topk_idx = np.argsort(mean)[::-1][:topk]
        topk_arms = [self._arms[i] for i in topk_idx]
        topk_mean = [float(mean[i]) for i in topk_idx]
        logger.info("MONITOR_TOP_ARMS k=%d arms=%s means=%s", topk, topk_arms, topk_mean)

        logger.info("")
        logger.info("✓ SELECTED ARM: %s (mean_prediction=%.6f)", chosen, max_pred)
        logger.info("================================================================================")
        return chosen

    def _fit_pairwise_residual(self, shared_seed_hex: str | None) -> None:
        if not self._pair_X:
            return
        X = np.array(self._X)
        y = np.array(self._y)
        pair_X = np.asarray(self._pair_X)
        pair_global_X = X[self._pair_start_index:]
        pair_base_y = y[self._pair_start_index:]
        if len(pair_X) != len(pair_base_y):
            raise ValueError(
                "Pairwise feature history is not aligned with Global RF "
                f"training history: pair={len(pair_X)} "
                f"global_since_pair_start={len(pair_base_y)}"
            )
        pair_y = pair_base_y - self._rf.predict(pair_global_X)
        self._pair_y = pair_y.astype(float).tolist()
        pair_replay_length = min(len(pair_y), self._replay_window)
        pair_bootstrapped_idx = self._bootstrap_indices(
            pair_replay_length,
            shared_seed_hex,
            label="pair_outer_bootstrap",
        )
        pair_training_X = pair_X[-pair_replay_length:][pair_bootstrapped_idx, :]
        pair_training_y = pair_y[-pair_replay_length:][pair_bootstrapped_idx]
        self._pair_rf.fit(pair_training_X, pair_training_y)
        self._pair_rf_is_fitted = True
        pair_window_X = pair_X[-pair_replay_length:]
        pair_window_y = pair_y[-pair_replay_length:]
        pair_mse = float(
            np.mean((self._pair_rf.predict(pair_window_X) - pair_window_y) ** 2)
        )
        logger.info(
            "PAIRWISE_RESIDUAL_RF_UPDATED samples=%d replay=%d "
            "bootstrap=%d feature_dim=%d feature_schema=%s "
            "target_schema=%s residual_mean=%.6f "
            "residual_abs_mean=%.6f mse=%.6f",
            len(pair_y),
            pair_replay_length,
            len(pair_bootstrapped_idx),
            int(pair_X.shape[1]),
            self.PAIRWISE_FEATURE_SCHEMA,
            self.PAIRWISE_TARGET_SCHEMA,
            float(np.mean(pair_window_y)),
            float(np.mean(np.abs(pair_window_y))),
            pair_mse,
        )

    def update(self, decisions, rewards, contexts=None, shared_seed_hex: str | None = None):
        if self.skips_learning():
            super().update(decisions, rewards, contexts, shared_seed_hex)
            return
        if contexts is None:
            contexts = [None] * len(decisions)

        new_pair_rows = []
        for arm, reward, context in zip(decisions, rewards, contexts):
            if float(reward) > 15:
                continue
            new_pair_rows.append(self._pair_feature_row(context, arm))

        super().update(decisions, rewards, contexts, shared_seed_hex)
        self._pair_X.extend(new_pair_rows)

        if (
            self._is_fitted
            and self._pair_X
            and len(self._y) >= self._min_samples_to_fit
            and self._update_count % self._fit_every == 0
        ):
            self._fit_pairwise_residual(shared_seed_hex)

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "reward_model": REWARD_MODEL,
                "rf": self._rf,
                "X": self._X,
                "y": self._y,
                "is_fitted": self._is_fitted,
                "update_count": self._update_count,
                "action_encoding": self.action_encoding,
                "arms": list(self._arms),
                "round_robin_step": self._round_robin_step,
                "round_robin_order": list(self._round_robin_order),
                "pair_feature_schema": self.PAIRWISE_FEATURE_SCHEMA,
                "pair_target_schema": self.PAIRWISE_TARGET_SCHEMA,
                "pair_rf": self._pair_rf,
                "pair_X": self._pair_X,
                "pair_y": self._pair_y,
                "pair_start_index": self._pair_start_index,
                "pair_rf_is_fitted": self._pair_rf_is_fitted,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        checkpoint_reward_model = data.get("reward_model", "global")
        checkpoint_encoding = data.get("action_encoding", "numeric")
        if checkpoint_encoding != self.action_encoding:
            raise ValueError(
                "CMAB checkpoint action encoding mismatch: "
                f"checkpoint={checkpoint_encoding}, configured={self.action_encoding}"
            )
        checkpoint_arms = data.get("arms")
        if (
            checkpoint_arms is not None
            and tuple(checkpoint_arms) != tuple(self._arms)
        ):
            raise ValueError("CMAB checkpoint arm catalog does not match current arms")

        if checkpoint_reward_model == REWARD_MODEL:
            checkpoint_pair_schema = data.get("pair_feature_schema")
            if checkpoint_pair_schema != self.PAIRWISE_FEATURE_SCHEMA:
                raise ValueError(
                    "CMAB combined checkpoint feature schema mismatch: "
                    f"checkpoint={checkpoint_pair_schema!r}, "
                    f"configured={self.PAIRWISE_FEATURE_SCHEMA!r}."
                )
            checkpoint_target_schema = data.get("pair_target_schema")
            if checkpoint_target_schema != self.PAIRWISE_TARGET_SCHEMA:
                raise ValueError(
                    "CMAB combined checkpoint target schema mismatch: "
                    f"checkpoint={checkpoint_target_schema!r}, "
                    f"configured={self.PAIRWISE_TARGET_SCHEMA!r}."
                )
            self._rf = data["rf"]
            self._X = data["X"]
            self._y = data["y"]
            self._is_fitted = data["is_fitted"]
            self._update_count = data["update_count"]
            self._pair_rf = data["pair_rf"]
            self._pair_X = data.get("pair_X", [])
            self._pair_y = data.get("pair_y", [])
            self._pair_start_index = int(
                data.get("pair_start_index", len(self._y) - len(self._pair_X))
            )
            self._pair_rf_is_fitted = data.get("pair_rf_is_fitted", False)
            if data.get("round_robin_order") is not None:
                self._round_robin_order = [int(idx) for idx in data["round_robin_order"]]
            if data.get("round_robin_step") is not None:
                self._round_robin_step = int(data["round_robin_step"])
            return

        if checkpoint_reward_model != "global":
            raise ValueError(
                "CMAB checkpoint reward model mismatch: "
                f"checkpoint={checkpoint_reward_model}, configured=combined"
            )

        self._rf = data["rf"]
        self._X = data["X"]
        self._y = data["y"]
        self._is_fitted = data["is_fitted"]
        self._update_count = data["update_count"]
        if data.get("round_robin_order") is not None:
            self._round_robin_order = [int(idx) for idx in data["round_robin_order"]]
        if data.get("round_robin_step") is not None:
            self._round_robin_step = int(data["round_robin_step"])
        self._pair_start_index = len(self._y)
        self._pair_X = []
        self._pair_y = []
        self._pair_rf_is_fitted = False
        logger.info(
            "Loaded global-only CMAB checkpoint; combined residual RF "
            "will start from the next valid sample"
        )
