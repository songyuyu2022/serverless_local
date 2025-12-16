import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoE(nn.Module):
    """
    一个真实可训练 MoE LM block（最小但正确）：
    - Pre: Embedding + Router (gate)
    - Expert: per-expert MLP
    - Post: LayerNorm + vocab head

    注意：
    - forward_pre 保留 [B,T,D]，不做 mean
    - forward_post 返回 [B*T, vocab_size] 与 controller 的 y.reshape(-1) 对齐
    """

    def __init__(self, vocab_size: int, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.vocab_size = int(vocab_size)

        # Pre
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.gate = nn.Linear(self.d_model, self.num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Linear(self.d_model * 2, self.d_model),
            )
            for _ in range(self.num_experts)
        ])

        # Post
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward_pre(self, x: torch.Tensor):
        """
        x: [B,T] (long)
        return:
          h: [B,T,D]
          topk_vals: [B,T,K]
          topk_idx:  [B,T,K]
        """
        h = self.embed(x)  # [B,T,D]
        logits = self.gate(h)  # [B,T,E]
        probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # [B,T,K]
        return h, topk_vals, topk_idx

    def forward_single_expert(self, eid: int, x_input: torch.Tensor) -> torch.Tensor:
        """
        x_input: [M,D]
        return:  [M,D]
        """
        return self.experts[int(eid)](x_input)

    def forward_post(self, combined_output: torch.Tensor) -> torch.Tensor:
        """
        combined_output: [B,T,D]
        return logits:   [B*T, vocab_size]
        """
        out = self.norm(combined_output)
        logits = self.head(out)  # [B,T,V]
        return logits.reshape(-1, self.vocab_size)

    def forward(self, x: torch.Tensor):
        """
        仅用于快速验证（controller 走分发逻辑，不靠这个）
        """
        h, topk_vals, topk_idx = self.forward_pre(x)
        # 这里不做真实 dispatch，只用于 sanity check
        y = self.experts[0](h) * topk_vals[..., 0:1]
        return self.forward_post(y), topk_idx
