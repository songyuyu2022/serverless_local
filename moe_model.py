# moe_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoE(nn.Module):
    def __init__(self, vocab_size, d_model, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_experts = num_experts

        # === Pre 部分 ===
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gate = nn.Linear(d_model, num_experts)

        # === Expert 部分 ===
        # 使用 ModuleList 模拟不同的专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
            for _ in range(num_experts)
        ])

        # === Post 部分 ===
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 20)  # 假设 20 分类任务

    # 1. Pre 阶段：Embedding + Gating
    def forward_pre(self, x):
        # x: [Batch, SeqLen]
        emb = self.embed(x)  # [B, T, D]
        # 简单 Pooling
        h = torch.mean(emb, dim=1)  # [B, D]

        # 计算 Gating
        logits = self.gate(h)
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        return h, topk_vals, topk_idx

    # 2. Expert 阶段：执行单个专家
    def forward_single_expert(self, eid, x_input):
        # x_input: [Batch_subset, D]
        return self.experts[eid](x_input)

    # 3. Post 阶段：分类 + Loss (在 Controller 里算 Loss)
    def forward_post(self, combined_output):
        out = self.norm(combined_output)
        logits = self.head(out)
        return logits

    # 完整 Forward (用于验证或简单调用)
    def forward(self, x):
        h, topk_vals, topk_idx = self.forward_pre(x)

        # 简单的串行执行模拟
        batch_size = h.size(0)
        combined_output = torch.zeros_like(h)

        for i in range(self.top_k):
            idx = topk_idx[:, i]
            val = topk_vals[:, i].unsqueeze(1)

            for b in range(batch_size):
                eid = idx[b].item()
                expert_out = self.experts[eid](h[b].unsqueeze(0))
                combined_output[b] += val[b] * expert_out.squeeze(0)

        return self.forward_post(combined_output), topk_idx