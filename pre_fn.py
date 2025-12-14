import os
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

# 保持原有的工具引用
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from moe_config import load_moe_config
from utils.logger import log

# --- 引入 MakeMoE 适配器 ---
# 确保 makeMoE.py 在同一目录下
from makeMoE import MakeMoEAdapter

app = FastAPI()

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 如果是 mac 本地测试，可能需要强制 cpu 或 mps
# DEVICE = "cpu"

def init_pre_model():
    moe_cfg = load_moe_config()

    log("pre-fn", f"Initializing PreStage for model: {moe_cfg.model_name}")

    if moe_cfg.model_name == "make_moe":
        adapter = MakeMoEAdapter(moe_cfg)
        # 获取 MakeMoE 的前半部分 (Embedding -> Block_SA -> Router)
        model = adapter.get_pre_stage().to(DEVICE)
    else:
        # 兼容旧逻辑或报错
        raise ValueError(f"Unknown model: {moe_cfg.model_name}")

    # 简单的优化器，用于演示参数更新
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer

# 初始化模型
pre_model, optimizer = init_pre_model()

@app.post("/fwd")
async def pre_forward(req: Request):
    """
    前向传播接口：
    输入: {"x": packed_indices}
    输出:
      - "hidden": 发给 Expert 的隐层状态
      - "gate": Router 的概率分布 (用于 Controller 分发)
      - "hot"/"cold": 热点专家列表 (此处暂留空，可扩展)
    """
    try:
        body = await req.body()
        obj = loads(body)

        # MakeMoE 的输入是 token indices (LongTensor)
        # 对应的 dataset.py 产生的也是 indices
        x_pack = obj.get("x")
        if x_pack is None:
            return Response(content=dumps({"error": "missing x"}), status_code=400)

        x_mb = pack_to_tensor(x_pack).to(DEVICE, dtype=torch.long)

        pre_model.eval()
        with torch.no_grad():
            # 调用 MakeMoEPreStage
            # 期望返回: {"hidden_states": ..., "router_probs": ...}
            out = pre_model(x_mb)

        hidden = out["hidden_states"]
        router_probs = out["router_probs"]

        # 这里为了兼容 Controller 的逻辑，直接返回 router_probs 作为 gate
        # Controller 会根据 topk 再次处理

        resp_data = {
            "hidden": tensor_to_pack(hidden.cpu()),
            "gate": tensor_to_pack(router_probs.cpu()),
            "hot": [],  # 暂不实现复杂的自适应热点逻辑
            "cold": []
        }
        return Response(content=dumps(resp_data), media_type="application/msgpack")

    except Exception as e:
        log("pre-fn", f"FWD Error: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)

@app.post("/bwd")
async def pre_backward(req: Request):
    """
    反向传播接口：
    接收来自后续节点的梯度信号。
    在 Serverless 拆分架构中，通常 Controller 会协调梯度回传。
    """
    try:
        body = await req.body()
        obj = loads(body)

        # 简单模拟：如果收到梯度，执行 step (实际需要重构计算图)
        if "grads" in obj:
            # 在真实分布式训练中，这里需要根据 Re-computation 或保存的 Graph 进行 backward
            # 这里仅做占位，保证流程跑通
            pass

        return Response(content=dumps({"ok": True}), media_type="application/msgpack")

    except Exception as e:
        log("pre-fn", f"BWD Error: {e}")
        return Response(content=f"Internal Error: {e}", status_code=500)

@app.post("/step")
async def pre_step():
    """显式参数更新接口"""
    try:
        optimizer.step()
        optimizer.zero_grad()
        return Response(content=dumps({"ok": True}), media_type="application/msgpack")
    except Exception as e:
        return Response(content=f"Step Error: {e}", status_code=500)