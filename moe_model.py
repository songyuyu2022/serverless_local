# moe_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# [新增] 单个 Expert 的权威实现（可被 serverless 复用）
# ============================================================
class ExpertMLP(nn.Module):
    """
    MoE 中单个 Expert 的标准 FFN 实现
    当前结构与你原本 SimpleMoE 中完全一致：
      Linear(d_model -> 2*d_model) -> ReLU -> Linear(2*d_model -> d_model)
    """
    def __init__(self, d_model: int, hidden_mult: int = 2, act: str = "relu"):
        super().__init__()
        hidden = int(d_model) * int(hidden_mult)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

        act = act.lower()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build_expert(
    d_model: int,
    *,
    hidden_mult: int = 2,
    act: str = "relu",
) -> nn.Module:
    """
    构造一个 Expert（供 SimpleMoE / expert_app 复用）
    """
    return ExpertMLP(d_model, hidden_mult=hidden_mult, act=act)


# ============================================================
# SimpleMoE（几乎不变，只是用 build_expert）
# ============================================================
class SimpleMoE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Pre
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gate = nn.Linear(d_model, num_experts)

        # Experts（✅ 关键：统一走 build_expert）
        self.experts = nn.ModuleList([
            build_expert(d_model, hidden_mult=2, act="relu")
            for _ in range(num_experts)
        ])

        # Post
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward_pre(self, x: torch.Tensor):
        h = self.embed(x)                 # [B,T,D]
        logits = self.gate(h)             # [B,T,E]
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        return h, topk_vals, topk_idx

    def forward_single_expert(self, eid: int, x: torch.Tensor) -> torch.Tensor:
        return self.experts[int(eid)](x)

    def forward_post(self, combined: torch.Tensor) -> torch.Tensor:
        out = self.norm(combined)
        logits = self.head(out)
        return logits.view(-1, self.vocab_size)
