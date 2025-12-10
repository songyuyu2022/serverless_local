# moe_config.py
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MoeConfig:
    """全局 MoE 配置，集中管理与专家相关的超参数。

    字段说明：
    - num_experts: 逻辑专家个数（与 experts.json 中的 key 数量对应）
    - top_k:       每个 token 选择的专家数量
    - d_model:     模型隐层维度（embedding / expert 输入输出维度）
    - num_pre_layers:  pre_fn 中的 Transformer 层数
    - num_post_layers: post_fn 中的 Transformer 层数（如果需要可以使用）
    """
    num_experts: int
    top_k: int
    d_model: int
    num_pre_layers: int
    num_post_layers: int

# 默认 MoE 配置，后续调整超参时可以直接修改这里，无需设置环境变量
DEFAULT_MOE_CONFIG = MoeConfig(
    num_experts=1,
    top_k=2,
    d_model=256,
    num_pre_layers=2,
    num_post_layers=2,
)

def load_moe_config(expert_instances: Optional[Dict[str, Any]] = None) -> MoeConfig:
    """从环境变量 + experts.json 推断 MoE 配置。

    优先级：
    1) NUM_EXPERTS 环境变量（如果设置了，就以它为准）
    2) experts.json 里逻辑专家的个数（expert_instances 的 key 数）
    3) 默认退化为单专家 (1)

    这样你后期只需：
    - 改 NUM_EXPERTS（+ 对应 experts.json），就能控制专家数量
    - 改 TOP_K，就能控制 top-k 数量
    """

    # 从默认配置读取，若设置了环境变量则以环境变量覆盖，保持兼容
    d_model = int(os.getenv("EMB_DIM", str(DEFAULT_MOE_CONFIG.d_model)))
    num_pre_layers = int(
        os.getenv("N_LAYERS_PRE", str(DEFAULT_MOE_CONFIG.num_pre_layers))
    )
    num_post_layers = int(
        os.getenv("N_LAYERS_POST", os.getenv("N_LAYERS", str(DEFAULT_MOE_CONFIG.num_post_layers)))
    )

    # 专家数量：优先 NUM_EXPERTS，其次 experts.json
    num_experts_env = os.getenv("NUM_EXPERTS")
    if num_experts_env is not None:
        try:
            num_experts = int(num_experts_env)
        except ValueError:
            num_experts = 1
    else:
        if expert_instances:
            num_experts = max(1, len(expert_instances))
        else:
            num_experts = DEFAULT_MOE_CONFIG.num_experts

    # top-k：统一由 TOP_K 控制
    top_k = int(os.getenv("TOP_K", str(DEFAULT_MOE_CONFIG.top_k)))

    return MoeConfig(
        num_experts=num_experts,
        top_k=top_k,
        d_model=d_model,
        num_pre_layers=num_pre_layers,
        num_post_layers=num_post_layers,
    )
