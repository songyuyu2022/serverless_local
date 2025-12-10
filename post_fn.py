import os
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response

from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from utils.logger import log
from moe_config import load_moe_config

app = FastAPI()

# 为了本地模拟，先统一用 CPU
device = "cpu"

# 缓存每个 micro_id 最近一次前向的 hidden，用于构造专家梯度
_last_hidden: Dict[int, Dict[str, Any]] = {}

# ====================== 模型定义 ======================


class PostBlock(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(h)


class PostModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.block = PostBlock(d_model, n_layers, n_heads, dropout)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B,T,C)
        return logits: (B,T,V)
        """
        h = self.block(h)
        h = self.ln(h)
        logits = self.head(h)
        return logits


def init_post_model() -> Dict[str, Any]:
    vocab_size = int(os.getenv("VOCAB_SIZE", "2000"))
    moe_cfg = load_moe_config()
    d_model = moe_cfg.d_model
    n_layers = moe_cfg.num_post_layers
    n_heads = int(os.getenv("N_HEADS_POST", "4"))
    dropout = float(os.getenv("DROPOUT_POST", "0.1"))

    lr = float(os.getenv("LR_POST", os.getenv("LR", "1e-3")))
    weight_decay = float(os.getenv("WD_POST", "0.0"))

    model = PostModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log(
        "post-fn",
        f"PostModel init: vocab={vocab_size}, d_model={d_model}, "
        f"layers={n_layers}, heads={n_heads}, lr={lr}, wd={weight_decay}, device={device}",
    )

    return {"model": model, "optim": optim}


_state = init_post_model()
post_model: PostModel = _state["model"]
post_optim: torch.optim.Optimizer = _state["optim"]


def compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, Any]:
    """
    logits:  (B, T, V)
    targets: (B, T)
    """
    B, T, V = logits.shape

    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)

    loss = F.cross_entropy(logits_2d, targets_1d)

    with torch.no_grad():
        pred = logits_2d.argmax(dim=-1)  # (B*T,)
        acc_top1 = (pred == targets_1d).float().mean().item()

        topk = torch.topk(logits_2d, k=min(5, V), dim=-1).indices  # (B*T, K)
        correct_top5 = (
            (topk == targets_1d.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

    return {"loss": loss, "acc_top1": acc_top1, "acc_top5": correct_top5}


# ====================== HTTP 接口 ======================


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": device}


@app.post("/fwd")
async def post_forward(req: Request) -> Response:
    """
    与 controller.call_post_fwd 对齐的接口。

    输入 msgpack:
      {
        "y": packed_tensor(B,T,C),        # 由 controller 传入的 hidden（优先）
        "targets": packed_tensor(B,T),    # 标签
        "micro_id": int,
        "tokens": int,
        "emb_dim": int,
        "train": bool,
      }

    为兼容旧版本，也接受 "h" 作为 hidden 键（若没有 "y" 则退回用 "h"）。

    输出 msgpack:
      {
        "loss": float,
        "acc_top1": float,
        "acc_top5": float,
        "grads": packed_tensor(B,T,C),    # d(loss)/d(hidden)，给 /bwd 使用
      }
    """
    body = await req.body()
    obj = loads(body)

    # 1) 解析 hidden（优先 y，其次 h）
    if "y" in obj:
        hidden_pack = obj["y"]
    elif "h" in obj:
        hidden_pack = obj["h"]
    else:
        raise KeyError("post_fn /fwd: request must contain 'y' or 'h'")

    targets_pack = obj["targets"]
    train = bool(obj.get("train", True))

    # 2) 解包为张量
    h = pack_to_tensor(hidden_pack).to(device)         # (B, T, C)
    targets = pack_to_tensor(targets_pack).to(device)  # (B, T)

    # 3) 前向 & （可选）反向
    post_model.train()  # 一律 train 模式，用 train 标志控制是否更新

    h_detached = h.detach().clone().requires_grad_(True)
    logits = post_model(h_detached)

    stats = compute_loss_and_metrics(logits, targets)
    loss = stats["loss"]

    if train:
        post_optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_h = h_detached.grad.detach()
        post_optim.step()
        grads_pack = tensor_to_pack(grad_h.cpu())
    else:
        # 验证阶段不更新参数，也不需要真实梯度（controller 在 val 阶段不会调用 /bwd）
        grad_h = torch.zeros_like(h_detached)
        grads_pack = tensor_to_pack(grad_h.cpu())

    log(
        "post-fn",
        f"[fwd] train={train} loss={float(loss.item()):.4f} "
        f"acc1={stats['acc_top1']:.4f} acc5={stats['acc_top5']:.4f}",
    )

    out = {
        "loss": float(loss.item()),
        "acc_top1": float(stats["acc_top1"]),
        "acc_top5": float(stats["acc_top5"]),
        "grads": grads_pack,
    }
    return Response(content=dumps(out), media_type="application/octet-stream")


@app.post("/bwd")
async def post_backward(req: Request) -> Response:
    """
    controller 在 train 阶段会调用：

        grads_pack = post_resp["grads"]
        POST /bwd  { "grads": grads_pack, "micro_id": ... }

    这里我们做一个简化实现：
      - 不再对 PostModel 做额外反向（已经在 /fwd 里完成一次更新）
      - 只负责把 grads_pack 作为 pre_grads 原样返回给 controller
      - expert_grads 返回一个空 dict，这样 NSGA-II 那段会自动跳过
    """
    body = await req.body()
    obj = loads(body)

    grads_pack = obj.get("grads", None)
    if grads_pack is None:
        out = {"pre_grads": None, "expert_grads": {}}
        return Response(content=dumps(out), media_type="application/octet-stream")

    out = {
        "pre_grads": grads_pack,  # 给 pre_fn /bwd 用
        "expert_grads": {},       # 暂不向专家回传梯度
    }
    return Response(content=dumps(out), media_type="application/octet-stream")


@app.post("/step")
async def post_step() -> Response:
    """
    显式 step 接口。

    当前实现中，参数更新已经在 /fwd 内完成，
    因此这里不再做任何事情，只返回 ok=True，保证与 controller 兼容。
    """
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")
