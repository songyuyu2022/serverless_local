import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoE(nn.Module):
    def __init__(self, vocab_size, d_model, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # === Pre 部分 ===
        self.embed = nn.Embedding(vocab_size, d_model)
        # Gating 网络：决定每个 token 去哪个专家
        self.gate = nn.Linear(d_model, num_experts)

        # === Expert 部分 ===
        # 这里的 MLP 会作用于每个 token [..., d_model]
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
        # [重点] 输出维度必须是 vocab_size (12000)，不能是 20
        self.head = nn.Linear(d_model, vocab_size)

    # 1. Pre 阶段：Embedding + Gating
    def forward_pre(self, x):
        # x: [Batch, SeqLen]
        emb = self.embed(x)  # [B, T, D]

        # ==========================================================
        # [关键修复] 绝对不要在这里加 torch.mean！
        # 语言模型训练必须保留序列长度 (SeqLen)。
        # ==========================================================
        h = emb  # [B, T, D]

        # 计算 Gating
        logits = self.gate(h)  # [B, T, NumExperts]
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        return h, topk_vals, topk_idx

    # 2. Expert 阶段：执行单个专家
    def forward_single_expert(self, eid, x_input):
        # x_input: [..., D]
        return self.experts[eid](x_input)

    # 3. Post 阶段：分类
    def forward_post(self, combined_output):
        # combined_output: [Batch, SeqLen, D]
        out = self.norm(combined_output)
        logits = self.head(out)  # [Batch, SeqLen, VocabSize]

        # [关键修复] 将输出 Flatten，变成二维 [N, C]
        # 这样才能跟 controller.py 里的 y_mb.view(-1) 对应上
        return logits.view(-1, self.vocab_size)

    # 完整 Forward (用于验证或简单调用)
    def forward(self, x):
        h, topk_vals, topk_idx = self.forward_pre(x)
        batch_size = h.size(0)
        combined_output = torch.zeros_like(h)

        # 简单的串行执行模拟
        for k in range(self.top_k):
            val = topk_vals[:, :, k].unsqueeze(-1)  # [B, T, 1]

            # 为了跑通流程，简单地用第一个专家处理
            expert_out = self.experts[0](h)
            combined_output += val * expert_out

        return self.forward_post(combined_output), topk_idx