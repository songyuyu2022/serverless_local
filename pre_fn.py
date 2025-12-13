"""
Pre-function (Scientific Adaptive Version):
1. 核心机制：Strict Load-aware Tiering (严格负载感知分层)
2. 原理：
   - 计算实时平均负载 (Avg Load = K / N)
   - 只有负载 >= 平均值的专家才被定义为 Hot (享受 Redis 加速)
   - 负载 < 平均值的专家被定义为 Cold (走 OSS 低速通道)
3. 优势：
   - 不修改 Top-K，不影响模型精度
   - 纯观测式分层，完全自适应
"""

import os
import json
import time
import sys
from typing import Dict, Any, List, Optional
from collections import deque, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Request, Response

from moe_config import load_moe_config
from shared import dumps, loads, tensor_to_pack, pack_to_tensor, route_pack
from utils.logger import log

app = FastAPI()

# ------------------------------------------------------------
# 1. 配置加载
# ------------------------------------------------------------
try:
    INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
    FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")
    VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_json(path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[pre_fn] Error loading {path}: {e}")
            return {}

    func_map = _load_json(FUNC_MAP_FILE)
    if isinstance(func_map, dict) and "funcs" in func_map:
        func_map = func_map["funcs"]

    expert_entries = {k: v for k, v in func_map.items() if k.startswith("moe.expert_fwd:")}
    MOE_CONFIG = load_moe_config(expert_entries)

    print(f"[pre_fn] Config Loaded: Experts={MOE_CONFIG.num_experts}, TopK={MOE_CONFIG.top_k}, Device={DEVICE}")

except Exception as e:
    print(f"[pre_fn] CRITICAL STARTUP ERROR: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 2. 模型定义 (Router 参数初始化增加扰动)
# ------------------------------------------------------------

class PreModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_experts):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # [关键优化] 初始化时增加微小扰动，打破完美的均匀分布
        # 这有助于 Router 更快地产生偏好，从而出现冷热分化
        self.router = nn.Linear(emb_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        logits = self.router(h)
        return h, logits

try:
    model = PreModel(
        vocab_size=VOCAB_SIZE,
        emb_dim=MOE_CONFIG.d_model,
        num_experts=MOE_CONFIG.num_experts
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
except Exception as e:
    print(f"[pre_fn] Model init failed: {e}")
    sys.exit(1)

backward_cache: Dict[int, Dict[str, Any]] = {}

# ------------------------------------------------------------
# 3. [论文级实现] 自适应专家热度追踪
# ------------------------------------------------------------

class ExpertHeatTracker:
    def __init__(self, num_experts: int, top_k: int, window_size: int = 50):
        self.history = deque(maxlen=window_size)
        self.num_experts = num_experts
        self.top_k = top_k

        # 严苛阈值系数：只有达到平均负载的 1.0 倍以上才算热
        # 即使是均匀分布，由于随机波动，也总会有约一半在平均线以上，一半在以下
        self.alpha = 1.0

    def update(self, selected_indices_flat: List[int]):
        self.history.append(selected_indices_flat)

    def get_status(self) -> Dict[str, List[int]]:
        # 冷启动保护：前 5 个 batch 默认全热，保证不报错
        if len(self.history) < 5:
            return {"hot": list(range(self.num_experts)), "cold": []}

        counter = Counter()
        total_hits = 0
        for batch_selection in self.history:
            counter.update(batch_selection)
            total_hits += len(batch_selection)

        if total_hits == 0:
            return {"hot": [], "cold": list(range(self.num_experts))}

        # [核心算法]
        # 1. 计算每个专家的实际负载率
        real_load = {}
        for eid in range(self.num_experts):
            real_load[eid] = counter[eid] / total_hits

        # 2. 计算系统平均负载 (Theoretical Average)
        # 例如 4 个专家，Top-2，则每个专家理论上被选中的概率是 2/4 = 0.5 (相对于 batch size)
        # 但这里的 total_hits 是所有选中次数之和，所以平均负载就是 1 / N
        avg_load = 1.0 / max(1, self.num_experts)

        # 3. 设定动态阈值
        threshold = avg_load * self.alpha

        hot_experts = []
        cold_experts = []

        for eid, load in real_load.items():
            # 只有表现优于(或等于)平均值的才是热专家
            # 这保证了总会有一些专家(低于平均的)被归为冷专家
            if load >= threshold:
                hot_experts.append(eid)
            else:
                cold_experts.append(eid)

        return {"hot": hot_experts, "cold": cold_experts}

tracker = ExpertHeatTracker(
    num_experts=MOE_CONFIG.num_experts,
    top_k=MOE_CONFIG.top_k,
    window_size=20 # 使用较短的窗口以捕捉动态变化
)

# ------------------------------------------------------------
# 4. HTTP 接口
# ------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "experts": MOE_CONFIG.num_experts}

@app.post("/fwd")
async def fwd(request: Request):
    try:
        body = await request.body()
        data = loads(body)

        x_pack = data["x"]
        micro_id = data.get("micro_id", 0)
        x = pack_to_tensor(x_pack).to(DEVICE).long()

        # 1. 模型前向 (Router 自由决策)
        model.train()
        h, logits = model(x)

        backward_cache[micro_id] = {"h": h, "x": x}

        # 2. 观测与记录 (Observation)
        # 我们在这里只是"看" Router 选了谁，完全不干预它的选择
        with torch.no_grad():
            topk_indices = torch.topk(logits, k=MOE_CONFIG.top_k, dim=-1)[1]
            selected_flat = topk_indices.view(-1).tolist()

            # 更新统计数据
            tracker.update(selected_flat)
            # 获取当前的分类标签
            status = tracker.get_status()

        # 3. 返回结果 + 标签
        resp = {
            "hidden": tensor_to_pack(h.detach()),
            "gate": tensor_to_pack(logits.detach()),
            "hot": status["hot"],   # 告诉 Controller 谁是热的
            "cold": status["cold"]  # 告诉 Controller 谁是冷的
        }
        return Response(content=dumps(resp), media_type="application/msgpack")

    except Exception as e:
        print(f"[pre_fn] FWD ERROR: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)

@app.post("/bwd")
async def bwd(request: Request):
    try:
        body = await request.body()
        data = loads(body)
        grad_h_pack = data["grads"]
        micro_id = data.get("micro_id", 0)

        if micro_id not in backward_cache:
            return Response(content=dumps({"ok": False}), media_type="application/msgpack")

        cached = backward_cache.pop(micro_id)
        h = cached["h"]
        grad_h = pack_to_tensor(grad_h_pack).to(DEVICE)

        optimizer.zero_grad()
        h.backward(grad_h)
        optimizer.step()

        return Response(content=dumps({"ok": True}), media_type="application/msgpack")

    except Exception as e:
        print(f"[pre_fn] BWD ERROR: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)

@app.post("/step")
async def step_sched(request: Request):
    return Response(content=dumps({"ok": True}), media_type="application/msgpack")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)