import numpy as np
import random
import logging
import hashlib
from collections import deque
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class CMABPolicy:
    PAIRWISE_RESIDUAL_KEYS = ("cut_condition_type", "fast_path_timeout")
    PAIRWISE_FEATURE_SCHEMA = "state_plus_cut_condition_type_fast_path_timeout_v2"
    PAIRWISE_TARGET_SCHEMA = "current_global_training_residual_v1"

    def __init__(
        self,
        arms,
        feature_dim,
        policy_name="random",
        epsilon=0,
        min_samples_to_fit=0,
        fit_every=1,
        n_estimators=50,
        uses_context=True,
        random_state: int = 0,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0,
        replay_window: int = 200,
        enable_pairwise_residual_rf: bool = False,
    ):
        self._arms = list(arms)
        if len(set(self._arms)) != len(self._arms):
            raise ValueError("CMAB arms must be unique")
        self.policy_name = policy_name
        self.epsilon = epsilon
        self._min_samples_to_fit = min_samples_to_fit
        self._fit_every = fit_every
        self._uses_context = uses_context
        self._random_state = int(random_state)
        self._epsilon_decay = float(epsilon_decay)
        self._min_epsilon = float(min_epsilon)
        self._replay_window = max(1, int(replay_window))
        self.enable_pairwise_residual_rf = bool(enable_pairwise_residual_rf)
        
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=4, 
            bootstrap=True,  # Keep RF internal bootstrap enabled.
            verbose=2,
            random_state=self._random_state,
        )
        # Optional low-capacity correction model. Its input is the current
        # state plus cut_condition_type and fast_path_timeout. The legacy
        # global-only path remains unchanged while the feature is disabled.
        self._pair_rf = None
        if self.enable_pairwise_residual_rf:
            self._pair_rf = RandomForestRegressor(
                n_estimators=20,
                max_depth=3,
                min_samples_leaf=4,
                bootstrap=True,
                verbose=0,
                random_state=self._random_state + 1,
            )
        
        self._is_fitted = False
        self._update_count = 0
        self._X = []
        self._y = []
        self._pair_X = []
        self._pair_y = []
        self._pair_start_index = 0
        self._pair_rf_is_fitted = False
        
        self.arm_counts = {arm: 0 for arm in arms}
        self.uses_context = uses_context
        self._monitor_topk = 5
        # 滑动窗口，记录最近 replay_window 条决策，用于探索时优先选择近期尝试最少的 arm
        self._recent_decisions: deque = deque(maxlen=self._replay_window)

    def _stable_int(self, *parts) -> int:
        payload = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _deterministic_index(self, size: int, *parts) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        return self._stable_int(*parts) % size

    def _shared_rng_index(self, size: int, seed_hex: str | None, label: str) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        # Derive independent streams for different sampling stages (tree/arm).
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

    def _current_epsilon(self) -> float:
        # Decay exploration over training updates: more exploration early, more exploitation later.
        decayed = self.epsilon * (self._epsilon_decay ** self._update_count)
        return max(self._min_epsilon, decayed)

    def _bootstrap_indices(
        self,
        replay_length: int,
        shared_seed_hex: str | None,
        label: str = "outer_bootstrap",
    ) -> np.ndarray:
        if replay_length <= 0:
            return np.array([], dtype=np.int64)
        # Keep external bootstrap deterministic across nodes.
        if shared_seed_hex is not None:
            seed_u64 = self._stable_int(shared_seed_hex, label, self._update_count, replay_length)
        else:
            seed_u64 = self._stable_int(self._random_state, label, self._update_count, replay_length)
        rng = np.random.default_rng(seed_u64)
        return rng.choice(replay_length, replay_length, replace=True)

    def _arm_to_vector(self, arm) -> np.ndarray:
        if isinstance(arm, str) and "=" in arm:
            values = []
            for part in arm.split(","):
                _, value = part.split("=", 1)
                values.append(float(value))
            return np.asarray(values, dtype=np.float32)
        return np.asarray(arm, dtype=np.float32).flatten()

    def _feature_row(self, context, arm):
        """将 context 和 arm 组合成特征向量"""
        # 假设 context 是一个 list/ndarray，arm 是一个数值或向量
        arm_vec = self._arm_to_vector(arm)
        if self._uses_context:
            ctx_vec = np.array(context).flatten()
            return np.concatenate([ctx_vec, arm_vec])
        return arm_vec

    @staticmethod
    def _arm_values(arm) -> dict[str, float]:
        if not isinstance(arm, str) or "=" not in arm:
            raise ValueError(
                "Pairwise residual RF requires named CMAB arms, got "
                f"{arm!r}"
            )
        values = {}
        for part in arm.split(","):
            key, value = part.split("=", 1)
            values[key] = float(value)
        return values

    def _pair_feature_row(self, context, arm) -> np.ndarray:
        if context is None:
            raise ValueError("Pairwise residual RF requires a state context")
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

    def _pair_feature_matrix(self, context, arms) -> np.ndarray:
        return np.asarray(
            [self._pair_feature_row(context, arm) for arm in arms]
        )

    def _feature_matrix(self, context, arms=None):
        """为候选 arm 构建特征矩阵，用于一次性预测。"""
        candidates = self._arms if arms is None else arms
        return np.array([self._feature_row(context, arm) for arm in candidates])

    def _window_arm_counts_from_replay(self):
        """从最近 replay_window 的训练样本中统计各 arm 次数（checkpoint 兼容）。"""
        window_counts = {arm: 0 for arm in self._arms}
        if not self._X:
            return window_counts, 0

        arm_vectors = {arm: self._arm_to_vector(arm) for arm in self._arms}
        recent_features = self._X[-self._replay_window:]
        matched_rows = 0

        for row in recent_features:
            feature = np.asarray(row, dtype=np.float32).flatten()
            matched_arm = None
            # 固定顺序匹配，保证在恢复训练时行为可复现。
            for arm in sorted(self._arms, key=str):
                arm_vec = arm_vectors[arm]
                arm_len = arm_vec.size
                if feature.size < arm_len:
                    continue
                if np.allclose(feature[-arm_len:], arm_vec, rtol=1e-6, atol=1e-8):
                    matched_arm = arm
                    break
            if matched_arm is not None:
                window_counts[matched_arm] += 1
                matched_rows += 1

        return window_counts, matched_rows

    def select_arm(
        self,
        context,
        shared_seed_hex: str | None = None,
        allowed_arms=None,
    ):
        candidates = list(self._arms if allowed_arms is None else allowed_arms)
        if not candidates:
            raise ValueError("CMAB select_arm requires at least one candidate")
        unknown = [arm for arm in candidates if arm not in self._arms]
        if unknown:
            raise ValueError(f"Unknown CMAB candidate arms: {unknown}")

        # random mode
        if self.policy_name == "random":
            idx = self._shared_rng_index(len(candidates), shared_seed_hex, "random_policy")
            return candidates[idx]

        if (
            not self._is_fitted
            or len(self._y) < self._min_samples_to_fit
            or not hasattr(self._rf, "estimators_")
            or len(getattr(self._rf, "estimators_", [])) == 0
        ):
            idx = self._shared_rng_index(len(candidates), shared_seed_hex, "cold_start")
            logger.info("Model not ready: selecting deterministic exploration arm idx=%d.", idx)
            return candidates[idx]

        current_epsilon = self._current_epsilon()
        epsilon_probe = self._shared_rng_uniform(shared_seed_hex, "epsilon_explore")
        if epsilon_probe < current_epsilon:
            # 从 replay 样本（X/y）统计最近窗口内各 arm 的出现次数，避免 checkpoint 恢复后局部状态丢失。
            window_counts, matched_rows = self._window_arm_counts_from_replay()
            candidate_counts = {arm: window_counts[arm] for arm in candidates}
            min_count = min(candidate_counts.values())
            least_tried = sorted(
                [arm for arm, cnt in candidate_counts.items() if cnt == min_count]
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

        # Aggregate predictions across trees.
        logger.info("INFERENCE_START")
        features = self._feature_matrix(context, candidates)
        all_preds = np.stack([tree.predict(features) for tree in self._rf.estimators_])
        logger.info("INFERENCE_DONE")

        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        global_mean = mean.copy()
        if (
            self.enable_pairwise_residual_rf
            and self._pair_rf is not None
            and self._pair_rf_is_fitted
        ):
            pair_features = self._pair_feature_matrix(context, candidates)
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

        def _fmt_list(values, idxs):
            return "[" + ", ".join(f"{values[i]:.6f}" for i in idxs) + "]"

        logger.info("================================================================================")
        logger.info("SELECT ARM - Detailed Prediction Analysis")
        logger.info("Context: %s", np.asarray(context))
        logger.info("Number of trees in forest: %d", len(self._rf.estimators_))
        logger.info("")
        logger.info("Prediction statistics across all trees:")
        top5_mean_idx = np.argsort(mean)[::-1][:5]
        logger.info("  Mean predictions (top 5 arms): %s", _fmt_list(mean, top5_mean_idx))
        logger.info("  Std predictions (top 5 arms): %s", _fmt_list(std, top5_mean_idx))
        logger.info("")
        logger.info("Top 5 arms by mean prediction:")
        for rank, idx in enumerate(top5_mean_idx, start=1):
            logger.info(
                "  #%d: arm=%s, mean=%.6f, std=%.6f",
                rank, candidates[idx], mean[idx], std[idx]
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
            # Fallback deterministic tie-break for reproducibility when no shared seed is provided.
            chosen_idx = int(max_indices.min())
        chosen = candidates[chosen_idx]

        topk = min(self._monitor_topk, len(candidates))
        topk_idx = np.argsort(mean)[::-1][:topk]
        topk_arms = [candidates[i] for i in topk_idx]
        topk_mean = [float(mean[i]) for i in topk_idx]
        logger.info("MONITOR_TOP_ARMS k=%d arms=%s means=%s", topk, topk_arms, topk_mean)

        logger.info("")
        logger.info("✓ SELECTED ARM: %s (mean_prediction=%.6f)", chosen, max_pred)
        logger.info("================================================================================")
        return chosen

    def update(self, decisions, rewards, contexts=None, shared_seed_hex: str | None = None):
        if contexts is None:
            contexts = [None] * len(decisions)
            
        for arm, reward, context in zip(decisions, rewards, contexts):
            if float(reward) > 15:
                logger.warning(
                    "DROP_TRAIN_SAMPLE arm=%s reward=%.6f reason=reward_gt_3",
                    arm,
                    float(reward),
                )
                continue
            feature_row = self._feature_row(context, arm)
            if self.enable_pairwise_residual_rf:
                # Store the raw pairwise feature only. Residual targets are
                # regenerated after every Global RF fit so they always match
                # the current Global RF instead of a mixture of old models.
                self._pair_X.append(self._pair_feature_row(context, arm))
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

            # External bootstrap: resample replay data to approximate posterior.
            replay_length = min(len(y), self._replay_window)
            bootstrapped_idx = self._bootstrap_indices(replay_length, shared_seed_hex)
            training_X = X[-replay_length:][bootstrapped_idx, :]
            training_y = y[-replay_length:][bootstrapped_idx]
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

            self._rf.fit(training_X, training_y)
            self._is_fitted = True
            global_predictions = self._rf.predict(X)

            if (
                self.enable_pairwise_residual_rf
                and self._pair_rf is not None
                and self._pair_X
            ):
                pair_X = np.asarray(self._pair_X)
                pair_global_X = X[self._pair_start_index:]
                pair_base_y = y[self._pair_start_index:]
                if len(pair_X) != len(pair_base_y):
                    raise ValueError(
                        "Pairwise feature history is not aligned with Global RF "
                        f"training history: pair={len(pair_X)} "
                        f"global_since_pair_start={len(pair_base_y)}"
                    )
                # Recompute every stored target against the newly fitted Global
                # RF. Only the replay suffix is used for Pairwise RF training.
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
                    np.mean(
                        (self._pair_rf.predict(pair_window_X) - pair_window_y) ** 2
                    )
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

            mse = np.mean((global_predictions - y)**2)
            logger.info(
                "RF Updated: samples=%d, replay=%d, bootstrap=%d, MSE=%.6f",
                len(y), replay_length, len(bootstrapped_idx), mse
            )

    def save(self, path):
        import joblib
        joblib.dump({
            'rf': self._rf,
            'X': self._X,
            'y': self._y,
            'is_fitted': self._is_fitted,
            'update_count': self._update_count,
            'arms': list(self._arms),
            'enable_pairwise_residual_rf': self.enable_pairwise_residual_rf,
            'pair_feature_schema': self.PAIRWISE_FEATURE_SCHEMA,
            'pair_target_schema': self.PAIRWISE_TARGET_SCHEMA,
            'pair_rf': self._pair_rf,
            'pair_X': self._pair_X,
            'pair_y': self._pair_y,
            'pair_start_index': self._pair_start_index,
            'pair_rf_is_fitted': self._pair_rf_is_fitted,
        }, path)

    def load(self, path):
        import joblib
        data = joblib.load(path)
        checkpoint_arms = data.get('arms')
        if (
            checkpoint_arms is not None
            and tuple(checkpoint_arms) != tuple(self._arms)
        ):
            raise ValueError("CMAB checkpoint arm catalog does not match current arms")
        self._rf = data['rf']
        self._X = data['X']
        self._y = data['y']
        self._is_fitted = data['is_fitted']
        self._update_count = data['update_count']
        if self.enable_pairwise_residual_rf and data.get(
            'enable_pairwise_residual_rf', False
        ):
            checkpoint_pair_schema = data.get('pair_feature_schema')
            if checkpoint_pair_schema != self.PAIRWISE_FEATURE_SCHEMA:
                raise ValueError(
                    "CMAB pairwise checkpoint feature schema mismatch: "
                    f"checkpoint={checkpoint_pair_schema!r}, "
                    f"configured={self.PAIRWISE_FEATURE_SCHEMA!r}. "
                    "Old cut+timeout checkpoints cannot initialize the "
                    "state+cut+timeout pairwise RF."
                )
            checkpoint_target_schema = data.get('pair_target_schema')
            if checkpoint_target_schema != self.PAIRWISE_TARGET_SCHEMA:
                raise ValueError(
                    "CMAB pairwise checkpoint target schema mismatch: "
                    f"checkpoint={checkpoint_target_schema!r}, "
                    f"configured={self.PAIRWISE_TARGET_SCHEMA!r}. "
                    "Checkpoints containing pre-update residual targets cannot "
                    "initialize the current-Global-RF residual model."
                )
            self._pair_rf = data['pair_rf']
            self._pair_X = data.get('pair_X', [])
            self._pair_y = data.get('pair_y', [])
            self._pair_start_index = int(
                data.get('pair_start_index', len(self._y) - len(self._pair_X))
            )
            self._pair_rf_is_fitted = data.get('pair_rf_is_fitted', False)
        elif self.enable_pairwise_residual_rf:
            self._pair_start_index = len(self._y)
            logger.info(
                "Loaded global-only CMAB checkpoint; pairwise residual RF "
                "will start from the next valid sample"
            )
