# scheduler_nn.py
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import log

device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyRegressor(nn.Module):
    """一个很小的 MLP 回归模型，用来预测实例的代价（延迟/成本）"""
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
    """
    在线调度器：
      - select_instance: 用当前模型预测各实例代价，选最小的
      - update: 用真实 latency 做一次 SGD 更新
    """

    def __init__(self, lr: float = 1e-3, warmup: int = 50):
        # 特征维度：expert_id, tokens, emb_dim, rtt_ms, price, avg_q_ms
        self.in_dim = 6
        self.model = TinyRegressor(self.in_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.warmup = warmup  # 前几次可以用简单策略防止模型乱选
        self.num_updates = 0

    def build_feature(
        self,
        expert_id: int,
        inst: Dict[str, Any],
        req: Dict[str, Any],
    ) -> torch.Tensor:
        meta = inst.get("meta", {})
        dyn = inst.get("dyn", {})

        tokens = float(req.get("tokens", 0.0))
        emb_dim = float(req.get("emb_dim", 0.0))
        rtt_ms = float(meta.get("rtt_ms", 0.0))
        price = float(meta.get("price_cents_s", 0.0))
        avg_q = float(dyn.get("avg_q_ms", 0.0))

        # 注意：expert_id 也作为一个数值特征（如果 experts 不多，这样够用）
        feat = np.array(
            [
                float(expert_id),
                tokens,
                emb_dim,
                rtt_ms,
                price,
                avg_q,
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(feat).to(device)

    @torch.no_grad()
    def select_instance(
        self,
        expert_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """
        选择实例：
          - 如果模型还在 warmup 阶段，用简单 heuristic（比如 rtt_ms 最小）
          - 否则，用 MLP 预测每个实例的代价，选最小
        """
        if not instances:
            raise ValueError(f"No instances for expert {expert_id}")

        # warmup：先用 RTT 最小的来防止模型初期乱选
        if self.num_updates < self.warmup:
            best_inst = min(
                instances,
                key=lambda inst: float(inst.get("meta", {}).get("rtt_ms", 0.0)),
            )
            log(
                "nn-scheduler",
                f"[warmup] expert={expert_id}, choose inst={best_inst.get('id')} "
                f"by min rtt_ms",
            )
            return best_inst, 0.0

        # 正常模式：用模型预测
        self.model.eval()
        feats = []
        for inst in instances:
            feats.append(self.build_feature(expert_id, inst, req))
        X = torch.stack(feats, dim=0)  # [N, in_dim]
        preds = self.model(X).squeeze(-1)  # [N]

        best_idx = int(torch.argmin(preds).item())
        best_inst = instances[best_idx]
        best_score = float(preds[best_idx].item())

        log(
            "nn-scheduler",
            f"expert={expert_id}, candidates={len(instances)}, "
            f"best={best_inst.get('id')}, pred_cost={best_score:.4f}",
        )
        return best_inst, best_score

    def update(
        self,
        expert_id: int,
        inst: Dict[str, Any],
        req: Dict[str, Any],
        latency_ms: float,
    ):
        """
        用一条新的 (feature, latency_ms) 做一次 SGD 更新。
        注意：我们把 latency 当“代价”，用 MSE 回归。
        """
        self.model.train()
        x = self.build_feature(expert_id, inst, req).unsqueeze(0)  # [1, in_dim]
        y = torch.tensor([[float(latency_ms)]], dtype=torch.float32, device=device)  # [1, 1]

        self.opt.zero_grad()
        pred = self.model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt.step()

        self.num_updates += 1
        if self.num_updates % 50 == 0:
            log(
                "nn-scheduler",
                f"updates={self.num_updates}, last_loss={loss.item():.4f}",
            )


# 全局一个 scheduler 实例（类似你现在的 LGBMScheduler）
NN_SCHED = NNScheduler(
    lr=float(os.getenv("NN_SCHED_LR", "1e-3")),
    warmup=int(os.getenv("NN_SCHED_WARMUP", "50")),
)
