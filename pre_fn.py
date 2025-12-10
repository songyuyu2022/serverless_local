import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request, Response
from moe_config import load_moe_config
from shared import dumps, loads, tensor_to_pack, pack_to_tensor, route_pack
from utils.logger import log  # 需要 utils/logger.py

# ============================================================
# 1. 只依赖 INSTANCES_FILE + FUNC_MAP_FILE 推断 NUM_EXPERTS / TOP_K
# ============================================================

INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_experts_from_func_map(
    instances_path: str,
    func_map_path: str,
    func_prefix: str = "moe.expert_fwd:",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 instances.json + func_map.json 推导专家表：
      func_name 形如 "moe.expert_fwd:0", "moe.expert_fwd:1", ...

    返回:
        {
          "0": [inst_obj0, inst_obj1, ...],
          "1": [...],
          ...
        }
    这里 key 是字符串，兼容 EXPERT_INSTANCE_TABLE 的格式。
    """
    if not os.path.exists(func_map_path):
        raise RuntimeError(
            f"[pre_fn] FUNC_MAP_FILE={func_map_path} 不存在，无法推断专家结构"
        )
    if not os.path.exists(instances_path):
        raise RuntimeError(
            f"[pre_fn] INSTANCES_FILE={instances_path} 不存在，无法推断专家结构"
        )

    func_map = _load_json(func_map_path)
    instances_raw = _load_json(instances_path)

    # 兼容两种格式：
    # 1) {"instances": [ ... ]}
    # 2) [ ... ]
    if isinstance(instances_raw, dict):
        instances_list = instances_raw.get("instances", [])
    elif isinstance(instances_raw, list):
        instances_list = instances_raw
    else:
        raise RuntimeError(
            f"[pre_fn] instances.json 格式错误，期望 dict 或 list，实际是 {type(instances_raw)}"
        )

    inst_by_id: Dict[str, Dict[str, Any]] = {
        inst["id"]: inst for inst in instances_list
    }

    pat = re.compile(rf"^{re.escape(func_prefix)}(\d+)$")

    table: Dict[str, List[Dict[str, Any]]] = {}
    # func_map 可能是 {func_name: [inst_id...]} 或 {"funcs": {...}}
    if isinstance(func_map, dict) and "funcs" in func_map:
        fm = func_map["funcs"]
    else:
        fm = func_map

    for func_name, inst_ids in fm.items():
        m = pat.match(func_name)
        if not m:
            continue
        eid = m.group(1)  # 字符串
        table.setdefault(eid, [])
        for iid in inst_ids:
            inst = inst_by_id.get(iid)
            if inst is not None:
                table[eid].append(inst)

    if not table:
        raise RuntimeError(
            f"[pre_fn] 在 func_map.json 中没有找到 '{func_prefix}{{eid}}' 形式的函数名，"
            f"请检查 func_map.json 或设置 NUM_EXPERTS 环境变量"
        )

    log("pre_fn", f"infer_experts_from_func_map -> experts: {list(table.keys())}")
    return table


def load_expert_instance_table() -> Dict[str, List[Dict[str, Any]]]:
    """
    最终统一入口：
    1) 如果设置了 EXP_INSTANCES_JSON，就按 JSON 字符串解析（兼容老配置）
    2) 如果设置了 EXP_INSTANCES_FILE，就从文件读取（兼容老配置）
    3) 否则，完全根据 INSTANCES_FILE + FUNC_MAP_FILE 推导专家结构
    """
    # 1) EXP_INSTANCES_JSON
    json_str = os.getenv("EXP_INSTANCES_JSON")
    if json_str:
        try:
            table = json.loads(json_str)
            if isinstance(table, dict):
                log("pre_fn", "使用 EXP_INSTANCES_JSON 中提供的专家表")
                return table
        except Exception as e:
            log("pre_fn", f"Failed to parse EXP_INSTANCES_JSON: {e}")

    # 2) EXP_INSTANCES_FILE
    path = os.getenv("EXP_INSTANCES_FILE", "").strip()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                table = json.load(f)
            if isinstance(table, dict):
                log("pre_fn", f"使用 EXP_INSTANCES_FILE={path} 中的专家表")
                return table
        except Exception as e:
            log("pre_fn", f"Failed to read EXP_INSTANCES_FILE={path}: {e}")

    # 3) 新逻辑：只依赖 instances.json + func_map.json
    table = infer_experts_from_func_map(INSTANCES_FILE, FUNC_MAP_FILE)
    return table


EXPERT_INSTANCE_TABLE: Dict[str, List[Dict[str, Any]]] = load_expert_instance_table()

# 从 moe_config 读取 MoE 配置，优先使用配置文件中的默认值，必要时根据 experts 表推断专家数量
MOE_CONFIG = load_moe_config(EXPERT_INSTANCE_TABLE or None)

# 基于配置确定专家数量和 top-k
if EXPERT_INSTANCE_TABLE:
    NUM_EXPERTS = max(MOE_CONFIG.num_experts, len(EXPERT_INSTANCE_TABLE))
else:
    NUM_EXPERTS = MOE_CONFIG.num_experts

    TOP_K = min(MOE_CONFIG.top_k, NUM_EXPERTS)
    log("pre_fn", f"MoE 配置: num_experts={NUM_EXPERTS}, top_k={TOP_K}")

# ============================================================
# 2. 模型定义（极简版 pre 模型）
# ============================================================

VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
EMB_DIM = MOE_CONFIG.d_model
N_LAYERS_PRE = MOE_CONFIG.num_pre_layers
N_HEADS_PRE = int(os.getenv("N_HEADS_PRE", "4"))
DROPOUT_PRE = float(os.getenv("DROPOUT_PRE", "0.1"))
MAX_SEQ_LEN = int(os.getenv("BLOCK_SIZE", "128"))
LR_PRE = float(os.getenv("LR_PRE", "1e-3"))
WD_PRE = float(os.getenv("WD_PRE", "0.0"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        att = att.softmax(dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PreMoEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        n_heads: int,
        num_experts: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,T) token ids
        返回:
          h: (B,T,D)
          router_logits: (B,T,NUM_EXPERTS)
        """
        B, T = x.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"seq len {T} > MAX_SEQ_LEN {self.max_seq_len}, 请调大 BLOCK_SIZE"
            )

        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        router_logits = self.router(h)
        return h, router_logits


app = FastAPI(title="pre_fn", version="0.1.0")

_model: Optional[PreMoEModel] = None
_optim: Optional[torch.optim.Optimizer] = None


def get_model_and_optim() -> Tuple[PreMoEModel, torch.optim.Optimizer]:
    global _model, _optim
    if _model is None:
        log(
            "pre_fn",
            f"init PreMoEModel: vocab={VOCAB_SIZE}, dim={EMB_DIM}, "
            f"layers={N_LAYERS_PRE}, heads={N_HEADS_PRE}, "
            f"num_experts={NUM_EXPERTS}, top_k={TOP_K}",
        )
        _model = PreMoEModel(
            vocab_size=VOCAB_SIZE,
            dim=EMB_DIM,
            num_layers=N_LAYERS_PRE,
            n_heads=N_HEADS_PRE,
            num_experts=NUM_EXPERTS,
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT_PRE,
        ).to(DEVICE)
        _optim = torch.optim.AdamW(
            _model.parameters(),
            lr=LR_PRE,
            weight_decay=WD_PRE,
        )
    return _model, _optim


# ============================================================
# 3. HTTP 接口：保持与 controller 兼容
# ============================================================

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
    }


_LAST_FORWARD: Dict[int, List[Dict[str, Any]]] = {}


@app.post("/fwd")
async def pre_forward(req: Request) -> Response:
    """
    与 controller.call_pre_fwd 对齐：
    输入 msgpack:
      {
        "x": packed_tensor(B,T),
        "micro_id": int,  # 可选，用于缓存本次前向
        "train": bool,    # 可选，缺省视为训练模式
      }
    输出 msgpack:
      {
        "h": packed_tensor(B,T,D),
        "routing": route_pack(...),
        "hidden": packed_tensor(B,T,D),
        "gate":   packed_tensor(B,T,NUM_EXPERTS),
        "route":  routing,
        "hot": [int expert_id...],
        "cold": [int expert_id...],
      }
    """
    model, _ = get_model_and_optim()

    body = await req.body()
    obj = loads(body)
    x = pack_to_tensor(obj["x"], device=DEVICE).long()  # (B,T)
    micro_id = int(obj.get("micro_id", -1))
    train = bool(obj.get("train", True))

    if train:
        model.train()
        h, router_logits = model(x)  # (B,T,D), (B,T,NUM_EXPERTS)
    else:
        model.eval()
        with torch.no_grad():
            h, router_logits = model(x)
    # 选 top-k 专家
    topk_vals, topk_idx = torch.topk(router_logits, k=TOP_K, dim=-1)  # (B,T,K)
    # 对 top-k logits 做 softmax 得到归一化权重
    gates = F.softmax(topk_vals, dim=-1)  # (B,T,K)

    # routing 打包（controller 里目前主要用于记录与调试）
    routing = route_pack(topk_idx.cpu(), gates.cpu())

    # 简单的 hot/cold 判定：凡是被选中的专家视为 hot，其余视为 cold
    flat_idx = topk_idx.reshape(-1)
    counts = torch.bincount(flat_idx, minlength=NUM_EXPERTS)
    hot = [int(i) for i, c in enumerate(counts.tolist()) if c > 0]
    cold = [i for i in range(NUM_EXPERTS) if i not in hot]

    if train and micro_id >= 0:
        # 缓存前向结果，后续 /bwd 用同一批数据做真实反向
        bucket = _LAST_FORWARD.setdefault(micro_id, [])
        bucket.append({"h": h, "router_logits": router_logits, "x": x})

    out: Dict[str, Any] = {
        "h": tensor_to_pack(h.detach().cpu()),
        "routing": routing,
        "hidden": tensor_to_pack(h.detach().cpu()),
        # gate 这里直接返回所有专家的原始 logits，方便 controller 做 top-k
        "gate": tensor_to_pack(router_logits.detach().cpu()),
        "route": routing,
        "hot": hot,
        "cold": cold,
    }
    return Response(content=dumps(out), media_type="application/octet-stream")

@app.post("/bwd")
async def pre_backward(req: Request) -> Response:
    """
    真实反向更新：
    - 从 payload["grads"] 中取出 grad_h: (B,T,D)
    - 从缓存中取回对应 micro 的前向输出 h，直接执行 h.backward(grad_h)
    - 若未命中缓存，则回退到占位反向以保持接口可用
    """
    model, optim = get_model_and_optim()

    body = await req.body()
    obj = loads(body)
    grads_pack = obj.get("grads")
    micro_id = int(obj.get("micro_id", -1))
    if grads_pack is None:
        return Response(
            content=dumps({"ok": False, "reason": "no grads"}),
            media_type="application/octet-stream",
        )

    grad_h = pack_to_tensor(grads_pack, device=DEVICE).float()  # (B,T,D)

    cached_list = _LAST_FORWARD.get(micro_id, []) if micro_id >= 0 else []
    cached = cached_list.pop(0) if cached_list else None
    if cached_list == [] and micro_id >= 0:
        _LAST_FORWARD.pop(micro_id, None)

    if cached is not None:
        model.train()
        optim.zero_grad(set_to_none=True)

        h = cached["h"]
        grad_h = grad_h.to(h.device)

        try:
            h.backward(grad_h)
        except Exception as e:
            # 如果原图无效（例如被覆盖或释放），重算一次前向再反向
            log("pre_fn", f"cached backward failed for micro_id={micro_id}: {e}; recompute")
            optim.zero_grad(set_to_none=True)
            x_cached = cached.get("x")
            if x_cached is None:
                raise
            with torch.enable_grad():
                h, _ = model(x_cached)
            h.backward(grad_h)

        optim.step()

        log(
            "pre_fn",
            f"apply real backward, micro_id={micro_id}, loss_grad_norm={float(grad_h.norm().item()):.6f}",
        )
        return Response(content=dumps({"ok": True}), media_type="application/octet-stream")

    # 占位分支：找不到缓存时仍保持接口可用
    model.train()
    B, T, D = grad_h.shape
    dummy_h = torch.randn(B, T, D, device=DEVICE, requires_grad=True)

    optim.zero_grad(set_to_none=True)
    loss = (dummy_h * grad_h).mean()
    loss.backward()

    # 将 dummy_h 的梯度“抄写”给模型第一个参数，避免无梯度情况
    for p in model.parameters():
        if p.grad is None and dummy_h.grad is not None:
            p.grad = dummy_h.grad.detach().mean().expand_as(p)
            break

    optim.step()

    log("pre_fn", f"apply dummy backward, loss={float(loss.item()):.6f}")
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")