# scheduler_nn.py
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import log
from scheduler_lgbm import encode_func_type  # 复用编码逻辑

device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyRegressor(nn.Module):
    """一个很小的 MLP 回归模型"""

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
        # 特征维度更新：
        # func_type_id, expert_id, tokens, emb_dim, rtt_ms, price, avg_q_ms
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
        meta = inst.get("meta", {})
        dyn = inst.get("dyn", {})

        ft_id = float(encode_func_type(func_type))
        tokens = float(req.get("tokens", 0.0))
        emb_dim = float(req.get("emb_dim", 0.0))
        rtt_ms = float(meta.get("rtt_ms", 0.0))
        price = float(meta.get("price_cents_s", 0.0))
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
        批量预测所有实例的代价（Score）。
        Warmup 阶段返回 0（不影响 LGBM 决策）或仅返回 RTT。
        """
        if not instances:
            return []

        # Warmup：模型未稳定前，返回 0，让 LGBM 主导
        if self.num_updates < self.warmup:
            return [0.0] * len(instances)

        self.model.eval()
        feats = []
        for inst in instances:
            feats.append(self.build_feature(func_type, expert_id, inst, req))

        if not feats:
            return []

        X = torch.stack(feats, dim=0)  # [N, in_dim]
        preds = self.model(X).squeeze(-1)  # [N]
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
        在线训练：用真实 Latency 做 SGD 更新
        """
        self.model.train()
        x = self.build_feature(func_type, expert_id, inst, req).unsqueeze(0)
        y = torch.tensor([[float(latency_ms)]], dtype=torch.float32, device=device)

        self.opt.zero_grad()
        pred = self.model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt.step()

        self.num_updates += 1
        # 每50次打印一次 Loss
        if self.num_updates % 50 == 0:
            log("nn-sched", f"updates={self.num_updates}, loss={loss.item():.4f}")

    # 兼容旧接口（如果还有地方用到）
    def select_instance(self, expert_id, instances, req):
        # 这是一个简单的 Wrapper，假设是 expert 任务
        scores = self.get_scores("moe.expert_fwd", expert_id, instances, req)
        best_idx = np.argmin(scores)
        return instances[best_idx], scores[best_idx]


NN_SCHED = NNScheduler(
    lr=float(os.getenv("NN_SCHED_LR", "1e-3")),
    warmup=int(os.getenv("NN_SCHED_WARMUP", "50")),
)