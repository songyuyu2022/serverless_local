# filename: makeMoE.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from model_interface import MoEPartitionInterface
from moe_config import MoeConfig


# -----------------------------------------------------------------------------
# 原始组件 (保留 MakeMoE 的核心逻辑)
# -----------------------------------------------------------------------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Expert(nn.Module):
    """ 单个专家网络 """

    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


# -----------------------------------------------------------------------------
# 拆分后的 Stage 定义
# -----------------------------------------------------------------------------

class MakeMoEPreStage(nn.Module):
    def __init__(self, config: MoeConfig, vocab_size: int, block_size: int):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding_table = nn.Embedding(block_size, config.d_model)

        # 简单起见，MakeMoE 的 PreStage 包含 Attention 层，但在 Expert 之前
        # 注意：原版 MakeMoE 是在 Block 里同时有 SA 和 MoE。
        # 为了符合 serverless 拆分，我们把 Self-Attention 放在 PreStage
        head_size = config.d_model // 8  # 假设 n_head=8
        self.sa = MultiHeadAttention(8, head_size, config.d_model, block_size, 0.1)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.router = NoisyTopkRouter(config.d_model, config.num_experts, config.top_k)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Self-Attention Block (原 Block 的前半部分)
        x = x + self.sa(self.ln1(x))

        # 准备进入 MoE，先做 LayerNorm
        x_norm = self.ln2(x)

        # Router 决策
        router_probs, indices = self.router(x_norm)

        return {
            "hidden_states": x_norm,  # 发送给 Expert 的数据
            "router_probs": router_probs,
            "expert_indices": indices,
            "residual": x  # 保留残差连接
        }


class MakeMoEPostStage(nn.Module):
    def __init__(self, config: MoeConfig, vocab_size: int):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size)

    def forward(self, combined_x, residual=None, targets=None):
        # combined_x 是 Expert 输出加权后的结果
        if residual is not None:
            x = residual + combined_x  # 完成 MoE 层的残差连接
        else:
            x = combined_x

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        # 简单的 top1/top5 计算用于指标
        acc_top1 = 0.0
        acc_top5 = 0.0
        if targets is not None:
            # 简单估算一下
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                _, pred = probs.topk(5, 1, largest=True, sorted=True)
                targets_view = targets.view(-1, 1)
                correct = pred.eq(targets_view)
                acc_top1 = correct[:, 0].float().mean().item()
                acc_top5 = correct.float().sum(1).mean().item()

        return logits, loss, acc_top1, acc_top5


# -----------------------------------------------------------------------------
# 适配器类 (实现接口)
# -----------------------------------------------------------------------------

class MakeMoEAdapter(MoEPartitionInterface):
    def __init__(self, config: MoeConfig):
        # 这里的 vocab_size 和 block_size 最好也放入 Config，这里暂时写死或透传
        self.config = config
        self.vocab_size = 65  # 对应 input.txt 里的字符数，实际应动态获取
        self.block_size = 64  # 对应 controller.py

        self.pre_stage = MakeMoEPreStage(config, self.vocab_size, self.block_size)
        self.post_stage = MakeMoEPostStage(config, self.vocab_size)

    def get_pre_stage(self) -> nn.Module:
        return self.pre_stage

    def get_expert_stage(self, expert_id: int) -> nn.Module:
        # 在单体模式下这里可能没用，但在 Worker 里有用
        return Expert(self.config.d_model)

    def get_post_stage(self) -> nn.Module:
        return self.post_stage

    def create_expert_instance(self, expert_id: int) -> nn.Module:
        return Expert(self.config.d_model)