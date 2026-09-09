"""XGBoost contextual bandit aligned with CMAB-RF selection.

CMAB-RF: each tree is a bootstrap draw; at decision time take the mean
prediction across trees and pick the max-mean arm. XGBoost trees are
additive, so a single booster is not a bag. We keep a committee of XGB
models, each fit on its own bootstrap of the replay window, then average
those predictions and argmax — same rule as RF.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import xgboost as xgb

from .policy import CMABPolicy

logger = logging.getLogger(__name__)


class XGBoostPolicy(CMABPolicy):
    def __init__(
        self,
        arms,
        feature_dim,
        policy_name="xgboost",
        min_samples_to_fit=0,
        fit_every=1,
        n_estimators=50,
        boosting_rounds=20,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=4,
        reg_lambda=1.0,
        uses_context=True,
        random_state: int = 0,
        replay_window: int = 200,
        action_encoding: str = "numeric",
        **_ignored,
    ):
        super().__init__(
            arms=arms,
            feature_dim=feature_dim,
            policy_name=policy_name,
            epsilon=0,
            min_samples_to_fit=min_samples_to_fit,
            fit_every=fit_every,
            n_estimators=n_estimators,
            uses_context=uses_context,
            random_state=random_state,
            replay_window=replay_window,
            action_encoding=action_encoding,
        )
        self._n_committee = max(1, int(n_estimators))
        self._xgb_kwargs = {
            "n_estimators": int(boosting_rounds),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": float(min_child_weight),
            "reg_lambda": float(reg_lambda),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": self._random_state,
            "verbosity": 0,
        }
        self._committee: list[xgb.XGBRegressor] = []

    def _committee_ready(self) -> bool:
        return (
            self._is_fitted
            and len(self._y) >= self._min_samples_to_fit
            and len(self._committee) > 0
        )

    def _bootstrap_indices_member(
        self,
        replay_length: int,
        shared_seed_hex: str | None,
        member: int,
    ) -> np.ndarray:
        if replay_length <= 0:
            return np.array([], dtype=np.int64)
        if shared_seed_hex is not None:
            seed_u64 = self._stable_int(
                shared_seed_hex, "outer_bootstrap", self._update_count, replay_length, member
            )
        else:
            seed_u64 = self._stable_int(
                self._random_state, "outer_bootstrap", self._update_count, replay_length, member
            )
        rng = np.random.default_rng(seed_u64)
        return rng.choice(replay_length, replay_length, replace=True)

    def select_arm(self, context, shared_seed_hex: str | None = None, epoch: int | None = None):
        if self.policy_name == "random":
            idx = self._shared_rng_index(len(self._arms), shared_seed_hex, "random_policy")
            return self._arms[idx]

        if not self._committee_ready():
            idx = self._shared_rng_index(len(self._arms), shared_seed_hex, "cold_start")
            logger.info("XGBoost not ready: selecting deterministic exploration arm idx=%d.", idx)
            return self._arms[idx]

        logger.info("INFERENCE_START")
        features = self._feature_matrix(context)
        all_preds = np.stack(
            [np.asarray(model.predict(features), dtype=np.float64) for model in self._committee]
        )
        logger.info("INFERENCE_DONE")

        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)

        def _fmt_list(values, idxs):
            return "[" + ", ".join(f"{values[i]:.6f}" for i in idxs) + "]"

        logger.info("================================================================================")
        logger.info("SELECT ARM - XGBoost ensemble mean (same rule as CMAB-RF)")
        logger.info("Context: %s", np.asarray(context))
        logger.info("Bootstrap committee size: %d", len(self._committee))
        logger.info("")
        logger.info("Prediction statistics across bootstrap models:")
        top5_mean_idx = np.argsort(mean)[::-1][:5]
        logger.info("  Mean predictions (top 5 arms): %s", _fmt_list(mean, top5_mean_idx))
        logger.info("  Std predictions (top 5 arms): %s", _fmt_list(std, top5_mean_idx))
        logger.info("")
        logger.info("Top 5 arms by mean prediction:")
        for rank, idx in enumerate(top5_mean_idx, start=1):
            logger.info(
                "  #%d: arm=%s, mean=%.6f, std=%.6f",
                rank,
                self._arms[idx],
                mean[idx],
                std[idx],
            )

        max_pred = float(mean.max())
        max_indices = np.flatnonzero(np.isclose(mean, max_pred))
        if len(max_indices) > 1:
            logger.info(
                "  Found %d arms with same max mean prediction (%.6f), deterministically selecting one...",
                len(max_indices),
                max_pred,
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
        logger.info("✓ SELECTED ARM: %s (mean_prediction=%.6f)", chosen, max_pred)
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
            feature_row = self._feature_row(context, arm)
            self._X.append(feature_row)
            self._y.append(float(reward))
            self.arm_counts[arm] += 1
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

        if len(self._y) >= self._min_samples_to_fit and self._update_count % self._fit_every == 0:
            X = np.array(self._X)
            y = np.array(self._y)
            replay_length = min(len(y), self._replay_window)
            committee: list[xgb.XGBRegressor] = []
            first_idx = None
            for member in range(self._n_committee):
                bootstrapped_idx = self._bootstrap_indices_member(
                    replay_length, shared_seed_hex, member
                )
                if first_idx is None:
                    first_idx = bootstrapped_idx
                training_X = X[-replay_length:][bootstrapped_idx, :]
                training_y = y[-replay_length:][bootstrapped_idx]
                model = xgb.XGBRegressor(**self._xgb_kwargs)
                model.fit(training_X, training_y)
                committee.append(model)

            self._committee = committee
            self._is_fitted = True

            preview_n = min(20, replay_length)
            unique_count = int(np.unique(first_idx).size) if first_idx is not None and replay_length > 0 else 0
            idx_digest = (
                hashlib.sha256(first_idx.tobytes()).hexdigest()[:16]
                if first_idx is not None
                else "none"
            )
            logger.info(
                "MONITOR_BOOTSTRAP update=%d replay=%d committee=%d unique0=%d idx_head(%d)=%s idx_hash0=%s seed=%s",
                self._update_count,
                replay_length,
                len(self._committee),
                unique_count,
                preview_n,
                first_idx[:preview_n].tolist() if first_idx is not None else [],
                idx_digest,
                shared_seed_hex if shared_seed_hex is not None else f"fallback:{self._random_state}",
            )

            ensemble = np.mean(
                [model.predict(X) for model in self._committee],
                axis=0,
            )
            mse = float(np.mean((ensemble - y) ** 2))
            logger.info(
                "XGBoost TS Updated: samples=%d, replay=%d, bootstrap_models=%d, MSE=%.6f",
                len(y),
                replay_length,
                len(self._committee),
                mse,
            )

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "committee": self._committee,
                "xgb_kwargs": self._xgb_kwargs,
                "n_committee": self._n_committee,
                "X": self._X,
                "y": self._y,
                "is_fitted": self._is_fitted,
                "update_count": self._update_count,
                "action_encoding": self.action_encoding,
                "arms": list(self._arms),
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        checkpoint_encoding = data.get("action_encoding", "numeric")
        if checkpoint_encoding != self.action_encoding:
            raise ValueError(
                "XGBoost checkpoint action encoding mismatch: "
                f"checkpoint={checkpoint_encoding}, configured={self.action_encoding}"
            )
        checkpoint_arms = data.get("arms")
        if checkpoint_arms is not None and tuple(checkpoint_arms) != tuple(self._arms):
            raise ValueError("XGBoost checkpoint arm catalog does not match current arms")
        if data.get("xgb_kwargs"):
            self._xgb_kwargs.update(data["xgb_kwargs"])
        if data.get("n_committee"):
            self._n_committee = int(data["n_committee"])
        if data.get("committee") is not None:
            self._committee = list(data["committee"])
        elif data.get("xgb") is not None:
            self._committee = [data["xgb"]]
        self._X = data["X"]
        self._y = data["y"]
        self._is_fitted = data["is_fitted"]
        self._update_count = data["update_count"]
