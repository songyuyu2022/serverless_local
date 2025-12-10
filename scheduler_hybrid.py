# scheduler_hybrid.py
"""
Hybrid 调度器（LGBM + 启发式）：

- 对外提供统一接口：
    HYBRID_SCHED.select_instances(func_type, logical_id, instances, req, top_k=1)

- 内部调用 scheduler_lgbm.LGBMScheduler：
    * 若存在 LightGBM 模型，则用模型根据资源特征（cpu/mem/device/runtime/libs/rtt/price 等）
      预测每个实例的延迟，并选择预测成本最低的实例；
    * 若 LightGBM 未安装或模型文件不存在，则退化为启发式规则：
        - 优先选择 GPU / CUDA 实例
        - rtt_ms 越低越好
        - cold_start_ms 越低越好
        - price_cents_s 越低越好
"""

from typing import Any, Dict, List, Tuple

from scheduler_lgbm import LGBMScheduler, DEFAULT_LGBM_SCHED
from utils.logger import log


class HybridScheduler:
    """
    统一对外的调度接口：

    select_instances(
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
        top_k: int = 1,
    ) -> (chosen_instances, scores)

    当前实现：
      - 直接调用 LGBMScheduler（内部已经包含：LightGBM + 启发式回退）；
      - 预留位置，后续可以在这里加入 NN 残差调度等再做融合。
    """

    def __init__(self, lgbm_sched: LGBMScheduler | None = None) -> None:
        # 若外部没有传入，就使用 scheduler_lgbm 中的全局默认实例
        self.lgbm_sched = lgbm_sched or DEFAULT_LGBM_SCHED

    def select_instances(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
        top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        选择多个实例：
          - 返回 (chosen_instances, scores)，其中 scores 是“预测成本”或启发式 cost（越小越好）。

        参数：
          - func_type: 逻辑函数名，如 "moe.pre_fwd" / "moe.post_fwd" / "moe.expert_apply_grad:0"
          - logical_id: 逻辑 id（expert id 或 0）
          - instances: 候选实例列表（来自 instances.json + func_map.json）
          - req: 请求信息（至少包含 tokens / emb_dim）
          - top_k: 需要选择多少个实例（会自动 clip 到实例总数）
        """
        if not instances:
            raise RuntimeError("HybridScheduler.select_instances: instances 为空")

        chosen, scores = self.lgbm_sched.select_instances(
            func_type=func_type,
            logical_id=logical_id,
            instances=instances,
            req=req,
            top_k=top_k,
        )

        log(
            "hybrid-scheduler",
            f"func={func_type} id={logical_id}, "
            f"chosen={[inst.get('id') for inst in chosen]}, "
            f"scores={scores}",
        )
        return chosen, scores

    def select_instance(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """
        兼容旧接口：只选一个实例（等价于 top_k=1）
        """
        chosen, scores = self.select_instances(
            func_type=func_type,
            logical_id=logical_id,
            instances=instances,
            req=req,
            top_k=1,
        )
        return chosen[0], scores[0]


# controller.py 直接从这里 import HYBRID_SCHED 使用
HYBRID_SCHED = HybridScheduler()
