# hotcold.py
import os
import math
import time
from collections import defaultdict
from typing import Iterable, Set, Dict


class HotColdManager:
    """
    根据专家在时间窗口内的调用频率 + EMA，区分「热专家」和「冷专家」。

    主要接口：
    - observe_batch(expert_ids):  每次前向/反向时，把本 batch 用到的专家 id 列表喂进来
    - classify() -> (hot_ids, cold_ids): 在当前统计下划分热/冷集合
    - cold_delay_steps(): 返回建议的「冷专家梯度累计步数」(COLD_ACC_STEPS)
    """

    def __init__(self,
                 ema_alpha: float = 0.9,
                 window_sec: int = 5,
                 hot_top_pct: float = 0.3,
                 hot_min_freq: int = 1,
                 cold_acc_steps: int = 10) -> None:
        # EMA 系数
        self.alpha: float = float(os.getenv("HOT_EMA_ALPHA", ema_alpha))
        # 统计窗口（秒）
        self.window_sec: float = float(os.getenv("HOT_WINDOW_SEC", window_sec))
        # 认为是「热专家」的 top 百分比
        self.hot_top_pct: float = float(os.getenv("HOT_TOP_PCT", hot_top_pct))
        # 直接按频次判定为热的阈值
        self.hot_min_freq: int = int(os.getenv("HOT_MIN_FREQ", hot_min_freq))
        # 冷专家的梯度累计步数建议
        self._cold_acc_steps: int = int(os.getenv("COLD_ACC_STEPS", cold_acc_steps))

        # 统计量
        self._ema: Dict[int, float] = defaultdict(float)   # EMA
        self._count: Dict[int, int] = defaultdict(int)     # 当前窗口内的计数
        self._last_roll: float = time.time()

    def observe_batch(self, expert_ids: Iterable[int]) -> None:
        """在一次 batch 中观察到的专家调用情况。"""
        for eid in expert_ids:
            self._count[int(eid)] += 1

    def _maybe_roll_window(self) -> None:
        now = time.time()
        if now - self._last_roll < self.window_sec:
            return

        # 将当前窗口内的 count 融入 EMA，并清零 count
        for eid, c in list(self._count.items()):
            self._ema[eid] = self.alpha * self._ema[eid] + (1.0 - self.alpha) * c
            self._count[eid] = 0

        self._last_roll = now

    def classify(self) -> (Set[int], Set[int]):
        """
        返回 (hot_ids, cold_ids)。
        注意：如果当前还没有任何数据，返回空集合。
        """
        self._maybe_roll_window()
        if not self._ema and not self._count:
            return set(), set()

        # 将 EMA + 当前窗口 count 统一考虑
        score: Dict[int, float] = {}
        for eid in set(list(self._ema.keys()) + list(self._count.keys())):
            score[eid] = self._ema[eid] + self._count[eid]

        items = sorted(score.items(), key=lambda x: x[1], reverse=True)
        if not items:
            return set(), set()

        k = max(1, int(math.ceil(len(items) * self.hot_top_pct)))
        hot: Set[int] = {eid for eid, _ in items[:k]}

        # 再把“频次足够高”的也纳入 hot
        for eid, c in self._count.items():
            if c >= self.hot_min_freq:
                hot.add(eid)

        all_ids = set(score.keys())
        cold = all_ids - hot
        return hot, cold

    def cold_delay_steps(self) -> int:
        """建议的冷专家梯度累计步数。"""
        return self._cold_acc_steps
