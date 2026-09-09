import json
import numpy as np
import random
import logging
import hashlib
from collections import deque
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class CMABPolicy:
    ACTION_ENCODINGS = ("numeric", "one_hot")

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
        action_encoding: str = "numeric",
    ):
        self._arms = list(arms)
        if len(set(self._arms)) != len(self._arms):
            raise ValueError("CMAB arms must be unique")
        action_encoding = str(action_encoding).lower()
        if action_encoding not in self.ACTION_ENCODINGS:
            raise ValueError(
                "CMAB action encoding must be one of "
                f"{self.ACTION_ENCODINGS}, got {action_encoding!r}"
            )
        self.action_encoding = action_encoding
        self._arm_indices = {
            arm: index for index, arm in enumerate(self._arms)
        }
        self.policy_name = policy_name
        self.epsilon = epsilon
        self._min_samples_to_fit = min_samples_to_fit
        self._fit_every = fit_every
        self._uses_context = uses_context
        self._random_state = int(random_state)
        self._epsilon_decay = float(epsilon_decay)
        self._min_epsilon = float(min_epsilon)
        self._replay_window = max(1, int(replay_window))
        
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=4, 
            bootstrap=True,  # Keep RF internal bootstrap enabled.
            verbose=2,
            random_state=self._random_state,
        )
        
        self._is_fitted = False
        self._update_count = 0
        self._X = []
        self._y = []
        
        self.arm_counts = {arm: 0 for arm in arms}
        self.uses_context = uses_context
        self._monitor_topk = 5
        # 滑动窗口，记录最近 replay_window 条决策，用于探索时优先选择近期尝试最少的 arm
        self._recent_decisions: deque = deque(maxlen=self._replay_window)
        self._round_robin_step = 0
        self._round_robin_order = list(range(len(self._arms)))
        random.Random(self._random_state).shuffle(self._round_robin_order)
        self._round_robin_state_path = None

    def skips_learning(self) -> bool:
        return self.policy_name == "round_robin"

    def persist_round_robin(self, path=None) -> None:
        if self.policy_name != "round_robin":
            return
        raw = path or self._round_robin_state_path
        if not raw:
            return
        target = Path(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_name": self.policy_name,
            "random_state": self._random_state,
            "arms": list(self._arms),
            "order": list(self._round_robin_order),
            "step": int(self._round_robin_step),
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(target)

    def restore_round_robin(self, path=None) -> bool:
        if self.policy_name != "round_robin":
            return False
        raw = path or self._round_robin_state_path
        if not raw:
            return False
        target = Path(raw)
        if not target.exists():
            return False
        data = json.loads(target.read_text())
        saved_arms = tuple(data.get("arms") or [])
        if saved_arms != tuple(self._arms):
            logger.warning(
                "round_robin state catalog mismatch (%d vs %d arms); not restoring",
                len(saved_arms),
                len(self._arms),
            )
            return False
        order = data.get("order")
        if not isinstance(order, list) or len(order) != len(self._arms):
            logger.warning("round_robin state order invalid; not restoring")
            return False
        self._round_robin_order = [int(idx) for idx in order]
        self._round_robin_step = int(data.get("step") or 0)
        self._round_robin_state_path = str(target)
        logger.info(
            "Restored round_robin step=%d cycle=%d from %s",
            self._round_robin_step,
            self._round_robin_step // max(1, len(self._arms)),
            target,
        )
        return True

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

    def _bootstrap_indices(self, replay_length: int, shared_seed_hex: str | None) -> np.ndarray:
        if replay_length <= 0:
            return np.array([], dtype=np.int64)
        # Keep external bootstrap deterministic across nodes.
        if shared_seed_hex is not None:
            seed_u64 = self._stable_int(shared_seed_hex, "outer_bootstrap", self._update_count, replay_length)
        else:
            seed_u64 = self._stable_int(self._random_state, "outer_bootstrap", self._update_count, replay_length)
        rng = np.random.default_rng(seed_u64)
        return rng.choice(replay_length, replay_length, replace=True)

    def _arm_to_vector(self, arm) -> np.ndarray:
        if self.action_encoding == "one_hot":
            try:
                arm_index = self._arm_indices[arm]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Unknown CMAB arm for one-hot encoding: {arm!r}"
                ) from error
            encoded = np.zeros(len(self._arms), dtype=np.float32)
            encoded[arm_index] = 1.0
            return encoded
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

    def _feature_matrix(self, context):
        """为所有 arm 构建特征矩阵，用于一次性预测"""
        return np.array([self._feature_row(context, arm) for arm in self._arms])

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

    def _select_non_learned_arm(self, shared_seed_hex: str | None = None):
        """Return an arm for random / round-robin policies; otherwise None."""
        if self.policy_name == "random":
            idx = self._shared_rng_index(len(self._arms), shared_seed_hex, "random_policy")
            return self._arms[idx]
        if self.policy_name == "round_robin":
            n = len(self._arms)
            if n == 0:
                raise ValueError("round_robin policy requires a non-empty arm catalog")
            pos = self._round_robin_step % n
            idx = self._round_robin_order[pos]
            chosen = self._arms[idx]
            logger.info(
                "ROUND_ROBIN step=%d cycle=%d pos=%d arm=%s",
                self._round_robin_step,
                self._round_robin_step // n,
                pos,
                chosen,
            )
            self._round_robin_step += 1
            self.persist_round_robin()
            return chosen
        return None

    def select_arm(self, context, shared_seed_hex: str | None = None):
        non_learned = self._select_non_learned_arm(shared_seed_hex)
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
            # 从 replay 样本（X/y）统计最近窗口内各 arm 的出现次数，避免 checkpoint 恢复后局部状态丢失。
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

        # Aggregate predictions across trees.
        logger.info("INFERENCE_START")
        features = self._feature_matrix(context)
        all_preds = np.stack([tree.predict(features) for tree in self._rf.estimators_])
        logger.info("INFERENCE_DONE")

        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)

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
            # Fallback deterministic tie-break for reproducibility when no shared seed is provided.
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

    def update(self, decisions, rewards, contexts=None, shared_seed_hex: str | None = None):
        if self.skips_learning():
            logger.info("Skip RF update: policy=%s is selection-only", self.policy_name)
            return
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

            mse = np.mean((self._rf.predict(X) - y)**2)
            logger.info(
                "RF Updated: samples=%d, replay=%d, bootstrap=%d, MSE=%.6f",
                len(y), replay_length, len(bootstrapped_idx), mse
            )

    def save(self, path):
        import joblib
        joblib.dump({
            'reward_model': 'global',
            'rf': self._rf,
            'X': self._X,
            'y': self._y,
            'is_fitted': self._is_fitted,
            'update_count': self._update_count,
            'action_encoding': self.action_encoding,
            'arms': list(self._arms),
            'round_robin_step': self._round_robin_step,
            'round_robin_order': list(self._round_robin_order),
        }, path)

    def load(self, path):
        import joblib
        data = joblib.load(path)
        checkpoint_reward_model = data.get('reward_model', 'global')
        if checkpoint_reward_model != 'global':
            raise ValueError(
                "CMAB checkpoint reward model mismatch: "
                f"checkpoint={checkpoint_reward_model}, configured=global"
            )
        # Checkpoints written before action encoding was configurable used the
        # legacy numeric representation.
        checkpoint_encoding = data.get('action_encoding', 'numeric')
        if checkpoint_encoding != self.action_encoding:
            raise ValueError(
                "CMAB checkpoint action encoding mismatch: "
                f"checkpoint={checkpoint_encoding}, configured={self.action_encoding}"
            )
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
        if data.get('round_robin_order') is not None:
            self._round_robin_order = [int(idx) for idx in data['round_robin_order']]
        if data.get('round_robin_step') is not None:
            self._round_robin_step = int(data['round_robin_step'])