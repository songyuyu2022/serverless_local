import os
import asyncio
from typing import Dict, Any, List

import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from comm import CommManager
from utils.logger import log


app = FastAPI()

# 强制使用 CPU（本地模拟）
device = "cpu"

LOGICAL_EID = int(os.getenv("LOGICAL_EID", "0"))

COMM_SIM_DIR = os.getenv("COMM_SIM_DIR", "comm_sim")
COMM = CommManager(COMM_SIM_DIR)

GRAD_BATCH_SIZE = int(os.getenv("GRAD_BATCH_SIZE", "4"))
PULL_INTERVAL_MS = int(os.getenv("PULL_INTERVAL_MS", "50"))


class ExpertMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return h


def init_expert() -> Dict[str, Any]:
    dim = int(os.getenv("EMB_DIM", "256"))
    lr = float(os.getenv("LR_EXPERT", os.getenv("LR", "1e-3")))
    weight_decay = float(os.getenv("WD_EXPERT", "0.0"))

    expert = ExpertMLP(dim).to(device)
    optim = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=weight_decay)

    log(
        "expert-app",
        f"ExpertMLP init: dim={dim}, lr={lr}, wd={weight_decay}, "
        f"LOGICAL_EID={LOGICAL_EID}, device={device}",
    )

    return {"model": expert, "optim": optim}


_state = init_expert()
expert_model: ExpertMLP = _state["model"]
expert_optim: torch.optim.Optimizer = _state["optim"]


def apply_grads_to_expert(
    expert_model: nn.Module,
    grad_y: torch.Tensor,
    x: torch.Tensor,
) -> None:
    """
    对 ExpertMLP 应用梯度：
      - y = expert(x)
      - y.backward(grad_y)
    """
    expert_model.train()
    expert_optim.zero_grad(set_to_none=True)

    x_clone = x.detach().clone().requires_grad_(True)
    y = expert_model(x_clone)
    y.backward(grad_y)

    expert_optim.step()


@app.on_event("startup")
async def startup_event():
    async def background_grad_worker():
        """
        从 hot/cold 通道拉取梯度，做异步大 batch 更新。
        梯度包约定：
          {
            "grad": packed_tensor,
            "x": packed_tensor
          }
        """
        while True:
            try:
                hot_grad = COMM.pull_hot(LOGICAL_EID) or []
                cold_grad = COMM.pull_cold(LOGICAL_EID) or []

                grads: List[torch.Tensor] = []
                xs: List[torch.Tensor] = []

                for g_pack in hot_grad:
                    if g_pack is None:
                        continue
                    grad_tensor = pack_to_tensor(g_pack["grad"]).to(device)
                    x_tensor = pack_to_tensor(g_pack["x"]).to(device)
                    grads.append(grad_tensor)
                    xs.append(x_tensor)

                for g_pack in cold_grad:
                    if g_pack is None:
                        continue
                    grad_tensor = pack_to_tensor(g_pack["grad"]).to(device)
                    x_tensor = pack_to_tensor(g_pack["x"]).to(device)
                    grads.append(grad_tensor)
                    xs.append(x_tensor)

                if grads:
                    grad_cat = torch.cat(grads, dim=0)
                    x_cat = torch.cat(xs, dim=0)
                    apply_grads_to_expert(expert_model, grad_cat, x_cat)

            except Exception as e:
                log("expert-app", f"background_grad_worker error: {e}")

            await asyncio.sleep(PULL_INTERVAL_MS / 1000.0)

    asyncio.create_task(background_grad_worker())


@app.post("/fwd")
async def expert_forward(req: Request) -> Response:
    """
    输入:
      {"x": packed_tensor(N,D)}
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


@app.post("/grad/apply")
async def expert_apply_grad(req: Request) -> Response:
    """
    直接通过 HTTP 形式同步应用梯度
    payload:
      {
        "x": packed_tensor(N,D),
        "grad_y": packed_tensor(N,D)
      }
    """
    body = await req.body()
    obj = loads(body)

    x = pack_to_tensor(obj["x"]).to(device)
    grad_y = pack_to_tensor(obj["grad_y"]).to(device)

    apply_grads_to_expert(expert_model, grad_y, x)
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")


@app.post("/step")
async def expert_step() -> Response:
    """
    显式 step 接口（当前实现中可选用）
    """
    expert_optim.step()
    expert_optim.zero_grad(set_to_none=True)
    return Response(content=dumps({"ok": True}), media_type="application/octet-stream")
