# filename: model_interface.py
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch

class MoEPartitionInterface(ABC):
    """
    MoE 模型拆分标准接口。
    使得调度系统可以统一调用不同结构的 MoE 模型。
    """

    @abstractmethod
    def get_pre_stage(self) -> nn.Module:
        """
        返回 Pre-Stage 模块。
        功能：Embedding -> Layers -> Router
        输出必须包含：
            - 'hidden_states': 路由前的隐层状态 (Batch, Seq, Dim)
            - 'router_logits': 路由器的原始输出 (用于辅助 Loss)
            - 'expert_weights': 经过 Softmax 的权重 (Batch, Seq, TopK)
            - 'expert_indices': 选中的专家索引 (Batch, Seq, TopK)
        """
        pass

    @abstractmethod
    def get_expert_stage(self, expert_id: int) -> nn.Module:
        """
        返回指定 ID 的 Expert 模块。
        功能：输入 hidden_states，输出 expert_output
        """
        pass

    @abstractmethod
    def get_post_stage(self) -> nn.Module:
        """
        返回 Post-Stage 模块。
        功能：Layers -> Norm -> Head -> Loss
        输入：聚合后的 hidden_states (即 Pre-Stage 输出 + Expert 加权结果)
        """
        pass

    @abstractmethod
    def create_expert_instance(self, expert_id: int) -> nn.Module:
        """
        用于 Worker 端独立初始化一个专家实例（不加载整个模型）。
        """
        pass