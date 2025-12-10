# scheduler_lgbm.py
"""
基于 LightGBM + 启发式的通用实例调度器。

- 统一对外接口：LGBMScheduler.select_instances(...)
- 特征设计（实例侧 + 请求侧）：
    * func_type_id:   编码 "moe.pre_fwd" / "moe.post_fwd" / "moe.expert_fwd" 等
    * logical_id:     逻辑 expert id 或 0
    * cpu_cores
    * memory_mb
    * is_cuda        (device == "cuda")
    * runtime        (如 "python3.10", 通过哈希到 [0,1])
    * libs_count     (依赖数量)
    * rtt_ms
    * cold_start_ms
    * price_cents_s
    * tokens         (请求 token 数)
    * emb_dim        (embedding 维度)

- 若 LightGBM 模型存在（train_lightgbm.py 训练生成），则用模型预测“cost”；
- 若模型不存在或无法加载，则退回到启发式：
    * CUDA > CPU
    * rtt_ms 越低越好
    * cold_start_ms 越低越好
    * price_cents_s 越低越好
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from utils.logger import log

try:
    import lightgbm as lgb
except Exception:
    lgb = None
    log("lgbm-sched", "LightGBM not available, will use pure heuristic scheduler.")


def _hash_str_to_float(s: str) -> float:
    """
    把字符串 hash 到 [0,1]，用于 runtime 等分类特征。:contentReference[oaicite:6]{index=6}
    """
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def encode_func_type(func_type: str) -> int:
    """
    把函数类型编码成一个离散 id，作为特征的一部分。

    这里统一使用“逻辑函数名”(不带冒号后缀)：
      - moe.pre_fwd           -> 0
      - moe.post_fwd          -> 1
      - moe.expert_fwd        -> 2
      - moe.expert_apply_grad -> 3
      - other                 -> 4
    """
    base = func_type.split(":", 1)[0]
    if base == "moe.pre_fwd":
        return 0
    if base == "moe.post_fwd":
        return 1
    if base == "moe.expert_fwd":
        return 2
    if base == "moe.expert_apply_grad":
        return 3
    return 4


def build_feature_vector(
    func_type: str,
    logical_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> np.ndarray:
    """
    根据实例信息 + 请求信息构造一个特征向量（float32）。:contentReference[oaicite:7]{index=7}

    inst: 来自 instances.json 中的一条记录：
      {
        "id": "fn_pre_py_torch_cpu_1",
        "url": "http://127.0.0.1:8001",
        "runtime": "python3.10",
        "cpu_cores": 2,
        "memory_mb": 1024,
        "device": "cpu" / "cuda",
        "libs": ["torch", "fastapi", ...],
        "rtt_ms": 5.0,
        "cold_start_ms": 100.0,
        "price_cents_s": 0.01,
      }

    req: controller 在调用时传入，例如：
      {
        "tokens": int,
        "emb_dim": int,
      }
    """
    meta = inst.get("meta", inst)  # 兼容两种写法

    func_type_id = encode_func_type(func_type)
    logical_id_f = float(logical_id)

    cpu_cores = float(meta.get("cpu_cores", 1.0))
    memory_mb = float(meta.get("memory_mb", 1024.0))

    device = str(meta.get("device", "cpu")).lower()
    is_cuda = 1.0 if "cuda" in device or "gpu" in device else 0.0

    runtime = str(meta.get("runtime", "python3.10"))
    runtime_h = _hash_str_to_float(runtime)

    libs = meta.get("libs", [])
    libs_count = float(len(libs)) if isinstance(libs, (list, tuple)) else 0.0

    rtt_ms = float(meta.get("rtt_ms", 5.0))
    cold_start_ms = float(meta.get("cold_start_ms", 100.0))
    price_cents_s = float(meta.get("price_cents_s", 0.0))

    tokens = float(req.get("tokens", 0))
    emb_dim = float(req.get("emb_dim", 0))

    feat = np.array(
        [
            float(func_type_id),
            logical_id_f,
            cpu_cores,
            memory_mb,
            is_cuda,
            runtime_h,
            libs_count,
            rtt_ms,
            cold_start_ms,
            price_cents_s,
            tokens,
            emb_dim,
        ],
        dtype=np.float32,
    )
    return feat


class LGBMScheduler:
    """
    通用 LightGBM 调度器封装。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or os.getenv("LGBM_MODEL_PATH", "lgbm_sched.txt")
        self.model: Optional["lgb.Booster"] = None
        self._load_model_if_exists()

    # --------- 模型加载 / 查询 ---------

    def _load_model_if_exists(self) -> None:
        if lgb is None:
            self.model = None
            return
        if not os.path.exists(self.model_path):
            log("lgbm-sched", f"model file {self.model_path} not found, use heuristic.")
            self.model = None
            return
        try:
            self.model = lgb.Booster(model_file=self.model_path)
            log("lgbm-sched", f"Loaded LightGBM model from {self.model_path}")
        except Exception as e:
            log("lgbm-sched", f"Failed to load LightGBM model: {e}")
            self.model = None

    def has_model(self) -> bool:
        return self.model is not None

    # --------- 启发式 cost 计算 ---------

    def _heuristic_cost(self, inst: Dict[str, Any], req: Dict[str, Any]) -> float:
        """
        没有 LGBM 模型时，用一个简单的启发式 cost：:contentReference[oaicite:8]{index=8}

        cost 越小越好：
          - CUDA 实例给予一个负偏置（更“便宜”）
          - rtt_ms 越小越好
          - cold_start_ms 越小越好
          - price_cents_s 越小越好
        """
        meta = inst.get("meta", inst)

        device = str(meta.get("device", "cpu")).lower()
        is_cuda = 1.0 if "cuda" in device or "gpu" in device else 0.0

        rtt_ms = float(meta.get("rtt_ms", 5.0))
        cold_start_ms = float(meta.get("cold_start_ms", 100.0))
        price_cents_s = float(meta.get("price_cents_s", 0.0))

        cost = (
            rtt_ms
            + 0.1 * cold_start_ms
            + 10.0 * price_cents_s
            - 5.0 * is_cuda
        )
        return float(cost)

    # --------- 核心选择逻辑 ---------

    def select_instances(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
        top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        在给定的实例列表中选择 top_k 个实例。

        返回:
          chosen_instances: List[实例对象]
          scores: List[float]，越小越好（预测 cost 或启发式 cost）
        """
        if not instances:
            raise RuntimeError("LGBMScheduler.select_instances: instances 为空")

        top_k = max(1, min(top_k, len(instances)))

        if self.has_model():
            # 使用 LightGBM 模型预测 cost
            feats = [
                build_feature_vector(func_type, logical_id, inst, req)
                for inst in instances
            ]
            X = np.stack(feats, axis=0)  # (N,F)
            pred = self.model.predict(X)  # (N,)
            scores = np.asarray(pred, dtype=np.float32)
        else:
            # 使用启发式 cost
            scores = np.array(
                [
                    self._heuristic_cost(inst, req)
                    for inst in instances
                ],
                dtype=np.float32,
            )

        order = np.argsort(scores)
        chosen_idx = order[:top_k]
        chosen = [instances[i] for i in chosen_idx]
        chosen_scores = [float(scores[i]) for i in chosen_idx]

        log(
            "lgbm-sched",
            f"func={func_type} id={logical_id}, "
            f"candidates={len(instances)}, top_k={top_k}, "
            f"chosen={[(inst.get('id'), chosen_scores[i]) for i, inst in enumerate(chosen)]}",
        )

        return chosen, chosen_scores


# 一个全局默认调度器实例，HybridScheduler / controller 直接使用
DEFAULT_LGBM_SCHED = LGBMScheduler()


def record_lgb_training_sample(
    log_path: str,
    func_type: str,
    logical_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
    latency_ms: float,
) -> None:
    """
    把一次调用的 (feature, label=latency_ms) 追加到本地日志文件中，供离线训练 LightGBM 使用。

    train_lightgbm.py 可以从这个日志中读取样本，然后训练出一个新的模型文件，
    再由 LGBMScheduler 加载使用。
    """
    feat = build_feature_vector(func_type, logical_id, inst, req)
    label = float(latency_ms)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    try:
        with open(log_path, "ab") as f:
            # 简单用 numpy 保存：前面是 label，后面是特征
            arr = np.concatenate([[label], feat.astype(np.float32)], axis=0)
            f.write(arr.tobytes())
    except Exception as e:
        log("lgbm-sched", f"record_lgb_training_sample error: {e}")
