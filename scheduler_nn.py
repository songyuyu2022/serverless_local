import os
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import log

# 自动检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"


# [修复点]：将原本在 scheduler_lgbm.py 中的函数移到这里
def encode_func_type(func_type: str) -> int:
    """
    将函数类型字符串转换为整数 ID，用于神经网络输入特征。
    """
    # 移除可能的实例ID后缀 (例如 "moe.expert_fwd:0" -> "moe.expert_fwd")
    base = func_type.split(":", 1)[0]
    mapping = {
        "moe.pre_fwd": 0,
        "moe.post_fwd": 1,
        "moe.expert_fwd": 2,
        "moe.expert_apply_grad": 3
    }
    # 默认为 4 (unknown)
    return mapping.get(base, 4)


class TinyRegressor(nn.Module):
    """
    一个轻量级的 MLP 回归模型，用于预测实例的 Latency 或 Cost。
    结构：Input -> 32 -> ReLU -> 16 -> ReLU -> 1 -> Output
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class NNScheduler:
    def __init__(self, lr: float = 1e-3, warmup: int = 50):
        # 特征维度说明 (共7维):
        # 1. func_type_id (int)
        # 2. expert_id (int)
        # 3. tokens (float)
        # 4. emb_dim (float)
        # 5. rtt_ms (float)
        # 6. price (float)
        # 7. avg_q (float)
        self.in_dim = 7

        self.model = TinyRegressor(self.in_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.warmup = warmup
        self.num_updates = 0

    def build_feature(
            self,
            func_type: str,
            expert_id: int,
            inst: Dict[str, Any],
            req: Dict[str, Any],
    ) -> torch.Tensor:
        """构建特征向量"""
        meta = inst.get("meta", {})
        dyn = inst.get("dyn", {})

        # 使用本文件定义的 encode_func_type
        ft_id = float(encode_func_type(func_type))

        tokens = float(req.get("tokens", 0.0))
        emb_dim = float(req.get("emb_dim", 0.0))
        rtt_ms = float(meta.get("rtt_ms", 0.0))
        price = float(meta.get("price_cents_s", 0.0))
        # 动态队列延迟 (如果有监控数据的话，没有则是0)
        avg_q = float(dyn.get("avg_q_ms", 0.0))

        feat = np.array([
            ft_id,
            float(expert_id),
            tokens,
            emb_dim,
            rtt_ms,
            price,
            avg_q
        ], dtype=np.float32)

        return torch.from_numpy(feat).to(device)

    @torch.no_grad()
    def get_scores(
            self,
            func_type: str,
            expert_id: int,
            instances: List[Dict[str, Any]],
            req: Dict[str, Any],
    ) -> List[float]:
        """
        批量预测所有实例的 Cost。
        在 Warmup 阶段返回全 0，让系统完全依赖 Heuristic 规则。
        """
        if not instances:
            return []

        # Warmup 期：模型没训练好，暂不干预调度
        if self.num_updates < self.warmup:
            return [0.0] * len(instances)

        self.model.eval()
        feats = []
        for inst in instances:
            feats.append(self.build_feature(func_type, expert_id, inst, req))

        if not feats:
            return []

        X = torch.stack(feats, dim=0)  # [Batch, in_dim]
        preds = self.model(X).squeeze(-1)  # [Batch]

        return preds.cpu().numpy().tolist()

    def update(
            self,
            func_type: str,
            expert_id: int,
            inst: Dict[str, Any],
            req: Dict[str, Any],
            latency_ms: float,
    ):
        """
        在线训练：使用真实的 latency_ms 作为 label 更新网络
        """
        self.model.train()
        x = self.build_feature(func_type, expert_id, inst, req).unsqueeze(0)  # [1, in_dim]
        y = torch.tensor([[float(latency_ms)]], dtype=torch.float32, device=device)  # [1, 1]

        self.opt.zero_grad()
        pred = self.model(x)

        # 损失函数：均方误差 (MSE)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt.step()

        self.num_updates += 1
        # 每 50 次更新打印一次 Loss，方便观察收敛情况
        if self.num_updates % 50 == 0:
            log("nn-sched", f"updates={self.num_updates}, loss={loss.item():.4f}")


# 全局单例
NN_SCHED = NNScheduler(
    lr=float(os.getenv("NN_SCHED_LR", "1e-3")),
    warmup=int(os.getenv("NN_SCHED_WARMUP", "50")),
)