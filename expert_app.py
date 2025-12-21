# expert_app.py
"""
Serverless Expert Function (makeMoE-free, moe_model-based)

特性：
- ❌ 不再依赖 makeMoE
- ✅ 直接复用 moe_model.build_expert（结构唯一来源）
- ✅ 单 expert / 单职责 / 可独立部署
- ✅ 支持 forward + gradient apply（HTTP + hot/cold 通道）
"""

import os
import asyncio
from typing import Dict, Any, List

import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from comm import CommManager
from utils.logger import log
from moe_config import load_moe_config
from moe_model import build_expert   # ⭐ 关键：直接复用 moe_model

# ----------------------------------------------------------------------
# 基本配置
# ----------------------------------------------------------------------
app = FastAPI()

DEVICE = os.getenv("DEVICE", "cpu")
device = torch.device(DEVICE)

LOGICAL_EID = int(os.getenv("LOGICAL_EID", "0"))

COMM_SIM_DIR = os.getenv("COMM_SIM_DIR", "comm_sim")
COMM = CommManager(COMM_SIM_DIR)

PULL_INTERVAL_MS = int(os.getenv("PULL_INTERVAL_MS", "50"))

# ----------------------------------------------------------------------
# Expert 初始化（完全 makeMoE-free）
# ----------------------------------------------------------------------
def init_expert() -> Dict[str, Any]:
    """
    初始化单个 Expert：
    - 结构来源：moe_model.build_expert
    - 参数与 controller 中 SimpleMoE.experts[eid] 一致
    """
    moe_cfg = load_moe_config()

    d_model = int(getattr(moe_cfg, "d_model", 256))
    hidden_mult = int(os.getenv("EXPERT_HIDDEN_MULT", "2"))   # 与 SimpleMoE 默认一致
    act = os.getenv("EXPERT_ACT", "relu")

    expert = build_expert(
        d_model=d_model,
        hidden_mult=hidden_mult,
        act=act,
    ).to(device)

    lr = float(os.getenv("LR_EXPERT", os.getenv("LR", "1e-3")))
    wd = float(os.getenv("WD_EXPERT", "0.0"))

    optim = torch.optim.AdamW(
        expert.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    log(
        "expert-app",
        f"Init Expert-{LOGICAL_EID}: "
        f"d_model={d_model}, hidden_mult={hidden_mult}, act={act}, "
        f"lr={lr}, device={device}",
    )

    return {
        "model": expert,
        "optim": optim,
    }


_STATE = init_expert()
expert_model: nn.Module = _STATE["model"]
expert_optim: torch.optim.Optimizer = _STATE["optim"]

# ----------------------------------------------------------------------
# 梯度应用逻辑
# ----------------------------------------------------------------------
def apply_grads_to_expert(
    model: nn.Module,
    grad_y: torch.Tensor,
    x: torch.Tensor,
):
    """
    对 Expert 应用梯度：
      y = expert(x)
      y.backward(grad_y)
      optimizer.step()
    """
    model.train()
    expert_optim.zero_grad(set_to_none=True)

    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    y.backward(grad_y)

    expert_optim.step()


# ----------------------------------------------------------------------
# 后台：从 hot / cold 通道异步拉取梯度
# ----------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    async def background_grad_worker():
        """
        模拟 serverless 场景：
        - hot 通道：频繁更新
        - cold 通道：延迟 / 累积更新
        """
        while True:
            try:
                hot_grads = COMM.pull_hot(LOGICAL_EID) or []
                cold_grads = COMM.pull_cold(LOGICAL_EID) or []

                grads: List[torch.Tensor] = []
                xs: List[torch.Tensor] = []

                for pack in hot_grads + cold_grads:
                    if pack is None:
                        continue
                    grad = pack_to_tensor(pack["grad"]).to(device)
                    x = pack_to_tensor(pack["x"]).to(device)
                    grads.append(grad)
                    xs.append(x)

                if grads:
                    grad_cat = torch.cat(grads, dim=0)
                    x_cat = torch.cat(xs, dim=0)
                    apply_grads_to_expert(expert_model, grad_cat, x_cat)

            except Exception as e:
                log("expert-app", f"background_grad_worker error: {e}")

            await asyncio.sleep(PULL_INTERVAL_MS / 1000.0)

    asyncio.create_task(background_grad_worker())


# ----------------------------------------------------------------------
# HTTP 接口：Expert Forward
# ----------------------------------------------------------------------
@app.post("/fwd")
async def expert_forward(req: Request) -> Response:
    """
    Forward 接口
    输入:
      {"x": packed_tensor(N,D)}  或 {"h": ...}
    输出:
      {"y": packed_tensor(N,D)}
    """
    body = await req.body()
    obj = loads(body)

    x_pack = obj.get("x") or obj.get("h") or obj.get("hidden")
    if x_pack is None:
        return Response(
            content=dumps({"ok": False, "reason": "missing x"}),
            media_type="application/octet-stream",
        )

    x = pack_to_tensor(x_pack).to(device)

    expert_model.eval()
    with torch.no_grad():
        y = expert_model(x)

    out = {"y": tensor_to_pack(y.cpu())}
    return Response(content=dumps(out), media_type="application/octet-stream")


# ----------------------------------------------------------------------
# HTTP 接口：同步梯度应用（兼容路径）
# ----------------------------------------------------------------------
@app.post("/grad/apply")
async def expert_apply_grad(req: Request) -> Response:
    """
    同步梯度应用接口
    payload:
      {"x": packed_tensor(N,D), "grad_y": packed_tensor(N,D)}
    """
    body = await req.body()
    obj = loads(body)

    x = pack_to_tensor(obj["x"]).to(device)
    grad_y = pack_to_tensor(obj["grad_y"]).to(device)

    apply_grads_to_expert(expert_model, grad_y, x)
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")


# ----------------------------------------------------------------------
# 可选接口：显式 step（兼容旧路径）
# ----------------------------------------------------------------------
@app.post("/step")
async def expert_step() -> Response:
    expert_optim.step()
    expert_optim.zero_grad(set_to_none=True)
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")
