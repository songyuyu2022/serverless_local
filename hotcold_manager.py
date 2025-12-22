# hotcold_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import math

@dataclass
class HotColdStats:
    hot_ratio: float
    active_hot_ratio: float
    hot_set_size: int
    hot_set_jaccard: float
    expert_load_entropy: float
    hot_flip_cnt: int

class HotColdManager:
    """
    动态冷热专家识别：
    - 对每个 expert 的 dispatch 次数做 EMA
    - 每 step 选 EMA top-K 为 hot，其余为 cold
    """
    def __init__(self, num_experts: int, hot_k: int, ema_alpha: float = 0.2):
        assert 1 <= hot_k <= num_experts
        self.E = num_experts
        self.hot_k = hot_k
        self.alpha = ema_alpha

        self.ema = [0.0] * num_experts
        self._prev_hot: Set[int] = set()
        self._hot_flip_cnt = 0

    def update(self, dispatch_counts: List[int]) -> HotColdStats:
        assert len(dispatch_counts) == self.E

        # EMA 更新
        for i, c in enumerate(dispatch_counts):
            self.ema[i] = (1 - self.alpha) * self.ema[i] + self.alpha * float(c)

        # 选 top-K
        ranked = sorted(range(self.E), key=lambda i: self.ema[i], reverse=True)
        hot = set(ranked[: self.hot_k])
        cold = set(ranked[self.hot_k :])

        # flip/jaccard
        if self._prev_hot:
            inter = len(hot & self._prev_hot)
            union = len(hot | self._prev_hot)
            jaccard = inter / union if union > 0 else 1.0
        else:
            jaccard = 1.0

        if hot != self._prev_hot and self._prev_hot:
            self._hot_flip_cnt += 1

        self._prev_hot = hot

        # entropy（用 ema 归一化）
        s = sum(self.ema) + 1e-9
        p = [x / s for x in self.ema]
        ent = -sum(pi * math.log(pi + 1e-12) for pi in p) / math.log(self.E)

        # active_hot_ratio：本 step 内真正被调用的 expert 中 hot 占比
        active = {i for i, c in enumerate(dispatch_counts) if c > 0}
        if active:
            active_hot_ratio = len(active & hot) / len(active)
        else:
            active_hot_ratio = 0.0

        return HotColdStats(
            hot_ratio=len(hot) / self.E,
            active_hot_ratio=active_hot_ratio,
            hot_set_size=len(hot),
            hot_set_jaccard=jaccard,
            expert_load_entropy=ent,
            hot_flip_cnt=self._hot_flip_cnt,
        )

    def is_hot(self, expert_id: int) -> bool:
        return expert_id in self._prev_hot
