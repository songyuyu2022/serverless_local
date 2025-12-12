# scheduler_hybrid.py
from typing import Any, Dict, List, Tuple
import numpy as np
import os

# [修改] 引入新的 Heuristic 调度器
from scheduler_heuristic import HeuristicScheduler, DEFAULT_HEURISTIC_SCHED
from scheduler_nn import NNScheduler, NN_SCHED
from utils.logger import log


class HybridScheduler:
    """
    Hybrid = Heuristic (Base) + Online NN (Correction)
    """

    def __init__(
            self,
            base_sched: HeuristicScheduler | None = None,
            nn_sched: NNScheduler | None = None,
    ) -> None:
        self.base_sched = base_sched or DEFAULT_HEURISTIC_SCHED
        self.nn_sched = nn_sched or NN_SCHED

        # 权重控制：0.0 = 纯规则, >0 = 开启 AI 优化
        self.nn_weight = float(os.getenv("HYBRID_NN_WEIGHT", "0.5"))

    def select_instances(
            self,
            func_type: str,
            logical_id: int,
            instances: List[Dict[str, Any]],
            req: Dict[str, Any],
            top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:

        if not instances:
            raise RuntimeError("HybridScheduler: instances empty")

        # 1. 获取基础分 (稳定、托底)
        base_scores = self.base_sched.get_scores(func_type, logical_id, instances, req)

        # 2. 获取 AI 预测分 (动态、学习)
        nn_scores = self.nn_sched.get_scores(func_type, logical_id, instances, req)

        # 3. 融合: Final = Base + w * NN
        final_scores = []
        for b, n in zip(base_scores, nn_scores):
            final_scores.append(b + self.nn_weight * n)

        # 4. 排序
        order = np.argsort(final_scores)
        chosen_idx = order[:top_k]

        return [instances[i] for i in chosen_idx], [final_scores[i] for i in chosen_idx]

    def select_instance(
            self,
            func_type: str,
            logical_id: int,
            instances: List[Dict[str, Any]],
            req: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        chosen, scores = self.select_instances(func_type, logical_id, instances, req, top_k=1)
        return chosen[0], scores[0]

    def update_stats(
            self,
            func_type: str,
            logical_id: int,
            inst: Dict[str, Any],
            req: Dict[str, Any],
            latency_ms: float
    ):
        """在线学习闭环：将真实 Latency 反馈给 NN"""
        try:
            self.nn_sched.update(func_type, logical_id, inst, req, latency_ms)
        except Exception as e:
            log("hybrid", f"update_stats failed: {e}")


HYBRID_SCHED = HybridScheduler()