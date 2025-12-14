import os
import torch
import torch.nn as nn
from fastapi import FastAPI, Request, Response

# 保持原有的工具引用
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from utils.logger import log
from moe_config import load_moe_config

# --- 引入 MakeMoE 适配器 ---
from makeMoE import MakeMoEAdapter

app = FastAPI()

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_post_model():
    moe_cfg = load_moe_config()

    log("post-fn", f"Initializing PostStage for model: {moe_cfg.model_name}")

    if moe_cfg.model_name == "make_moe":
        adapter = MakeMoEAdapter(moe_cfg)
        # 获取 MakeMoE 的后半部分 (LayerNorm -> Head -> Loss)
        model = adapter.get_post_stage().to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {moe_cfg.model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


# 初始化模型
post_model, optimizer = init_post_model()


@app.post("/fwd")
async def post_forward(req: Request):
    """
    前向传播接口：
    输入:
      - "y": 经过 Expert 处理并由 Controller 聚合后的 hidden_states
      - "targets": 标签
      - "train": 是否训练模式
    输出:
      - loss, acc_top1, acc_top5
      - "grads": 对输入 y 的梯度 (用于回传给 Expert)
    """
    try:
        body = await req.body()
        obj = loads(body)

        # 1. 解析输入
        y_pack = obj.get("y")
        targets_pack = obj.get("targets")

        if y_pack is None or targets_pack is None:
            return Response(content=dumps({"error": "missing y or targets"}), status_code=400)

        # 聚合后的 Expert 输出
        combined_x = pack_to_tensor(y_pack).to(DEVICE)
        targets = pack_to_tensor(targets_pack).to(DEVICE, dtype=torch.long)
        train_mode = obj.get("train", True)

        # 2. 准备计算
        if train_mode:
            post_model.train()
            # 关键：为了计算对输入的梯度，必须开启梯度追踪
            combined_x.requires_grad_(True)
        else:
            post_model.eval()

        # 3. 执行 MakeMoE PostStage
        # MakeMoEPostStage 签名: forward(combined_x, residual=None, targets=targets)
        # 注意：此处 residual 暂时设为 None，意味着我们忽略了 MoE 层的残差连接
        # (在严格实现中，Controller 需要传递 residual，或者将其包含在 combined_x 中)
        logits, loss, acc1, acc5 = post_model(combined_x, residual=None, targets=targets)

        resp = {
            "loss": loss.item() if loss is not None else 0.0,
            "acc_top1": acc1,
            "acc_top5": acc5
        }

        # 4. 如果是训练模式，执行反向传播并返回梯度
        if train_mode and loss is not None:
            optimizer.zero_grad()
            loss.backward()

            # 获取对输入 combined_x 的梯度 (即 dLoss / d_ExpertOutput)
            if combined_x.grad is not None:
                x_grad = combined_x.grad
                resp["grads"] = tensor_to_pack(x_grad.cpu())
            else:
                # 某些极端情况没有梯度，返回全0防止 Controller 报错
                resp["grads"] = tensor_to_pack(torch.zeros_like(combined_x).cpu())

            # 更新 PostStage 的参数
            optimizer.step()

        return Response(content=dumps(resp), media_type="application/msgpack")

    except Exception as e:
        log("post-fn", f"FWD Error: {e}")
        # import traceback
        # traceback.print_exc()
        return Response(content=f"Internal Error: {e}", status_code=500)


@app.post("/bwd")
async def post_backward(req: Request):
    """
    反向传播接口：
    PostStage 是最后一环，通常 Controller 在此处不会再调用 bwd，
    除非有更后级的模块。保留接口以防报错。
    """
    return Response(content=dumps({"ok": True}), media_type="application/msgpack")


@app.post("/step")
async def post_step():
    """PostStage 的 step 通常在 fwd 训练时直接做了，这里留空或作为同步点"""
    return Response(content=dumps({"ok": True}), media_type="application/msgpack")