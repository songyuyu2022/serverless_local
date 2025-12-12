# scheduler_heuristic.py
from typing import Any, Dict, List, Tuple
import numpy as np
from utils.logger import log


class HeuristicScheduler:
    """
    纯启发式调度器：作为 Online NN 的基准（Base Score）。
    不依赖 LightGBM，直接使用物理规则打分。
    """

    def __init__(self):
        log("sched-base", "Initialized HeuristicScheduler (Rule-based).")

    def _calculate_cost(self, inst: Dict[str, Any]) -> float:
        """
        核心打分公式 (Cost 越小越好):
        Cost = 网络延迟 + (冷启动惩罚) + (价格权重) - (GPU 奖励)
        """
        meta = inst.get("meta", {})

        # 1. 基础延迟 (RTT)
        rtt = float(meta.get("rtt_ms", 10.0))

        # 2. 冷启动惩罚 (假设冷启动概率为 10%)
        cold_start = float(meta.get("cold_start_ms", 200.0))

        # 3. 价格惩罚 (分越贵，Cost 越高)
        price = float(meta.get("price_cents_s", 0.001))

        # 4. 设备偏好 (GPU 减分/奖励)
        device = str(meta.get("device", "cpu")).lower()
        is_cuda = 1.0 if "cuda" in device or "gpu" in device else 0.0

        # 公式系数可微调
        score = rtt + (0.05 * cold_start) + (1000.0 * price) - (10.0 * is_cuda)
        return score

    def get_scores(
            self,
            func_type: str,
            logical_id: int,
            instances: List[Dict[str, Any]],
            req: Dict[str, Any]
    ) -> List[float]:
        """批量计算所有实例的 heuristic cost"""
        return [self._calculate_cost(inst) for inst in instances]

    def select_instances(
            self,
            func_type: str,
            logical_id: int,
            instances: List[Dict[str, Any]],
            req: Dict[str, Any],
            top_k: int = 1
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        scores = self.get_scores(func_type, logical_id, instances, req)
        indices = np.argsort(scores)
        chosen_idx = indices[:top_k]

        return [instances[i] for i in chosen_idx], [scores[i] for i in chosen_idx]


# 全局单例
DEFAULT_HEURISTIC_SCHED = HeuristicScheduler()