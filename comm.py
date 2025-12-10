# comm.py
"""
本地通信模拟层：
- 使用本地目录 + 文件来模拟 Redis (hot) 和 OSS (cold)
- 适用于本地多进程 FastAPI 服务之间共享梯度 / 中间结果
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from shared import dumps, loads  # 复用你已有的 msgpack 序列化工具
from utils.logger import log


class LocalKVStore:
    """
    用目录 + 文件模拟一个简单 KV 存储：
    - set(key, value_bytes): 把 value 写到 base_dir/key
    - get(key): 读 base_dir/key 的内容
    """

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        # key 允许有子目录，例如 "hot/0"
        return self.base.joinpath(*key.split("/"))

    def set(self, key: str, value: bytes) -> None:
        with self._lock:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(value)
            tmp.replace(p)

    def get(self, key: str, delete: bool = False) -> Optional[bytes]:
        with self._lock:
            p = self._path(key)
            if not p.exists():
                return None
            data = p.read_bytes()
            if delete:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            return data

    def list_keys(self, prefix: str = "") -> Dict[str, Path]:
        """
        列出所有以 prefix 开头的 key -> path
        仅用于调试。
        """
        res: Dict[str, Path] = {}
        base_len = len(self.base.as_posix()) + 1
        for f in self.base.rglob("*"):
            if not f.is_file():
                continue
            rel = f.as_posix()[base_len:]
            if prefix and not rel.startswith(prefix):
                continue
            res[rel] = f
        return res


class CommManager:
    """
    通信管理器：
    - send_hot / pull_hot: 用 LocalKVStore 模拟 Redis 热通道
    - send_cold / pull_cold: 用 LocalKVStore 模拟 OSS 冷通道

    默认使用 ./comm_sim 作为根目录：
    - ./comm_sim/hot
    - ./comm_sim/cold
    """

    def __init__(self, base_dir: Optional[str] = None):
        base_dir = base_dir or os.getenv("COMM_SIM_DIR", "comm_sim")
        base = Path(base_dir)
        hot_dir = base / "hot"
        cold_dir = base / "cold"

        self._hot = LocalKVStore(str(hot_dir))
        self._cold = LocalKVStore(str(cold_dir))

        log("comm", f"CommManager initialized, base_dir={base_dir}")

    # -------- hot 通道（模拟 Redis） --------

    def _hot_key(self, expert_id: str) -> str:
        # 每个 expert 一个最新的 key，你要更细也可以包含 step
        return f"{expert_id}.bin"

    def send_hot(self, expert_id: str, obj: Dict[str, Any]) -> None:
        """
        把梯度等对象通过 hot 通道发送：
        - expert_id: 逻辑专家 id（'0','1',...）
        - obj: 任意可 msgpack 序列化的对象（例如 expert_grads dict）
        """
        key = self._hot_key(expert_id)
        data = dumps(obj)  # bytes
        self._hot.set(key, data)
        log("comm", f"send_hot: expert_id={expert_id}, bytes={len(data)}")

    def pull_hot(self, expert_id: str, delete: bool = True) -> Optional[Dict[str, Any]]:
        """
        从 hot 通道拉取最近一次发送的对象；
        - delete=True 表示读取后删除（一次性消费）
        """
        key = self._hot_key(expert_id)
        data = self._hot.get(key, delete=delete)
        if data is None:
            return None
        obj = loads(data)
        log("comm", f"pull_hot: expert_id={expert_id}, bytes={len(data)}")
        return obj

    # -------- cold 通道（模拟 OSS） --------

    def _cold_key(self, expert_id: str) -> str:
        # 可以额外加上 step，比如 {expert_id}_stepxxxx.bin，这里先简单实现
        return f"{expert_id}.bin"

    def send_cold(self, expert_id: str, obj: Dict[str, Any]) -> None:
        """
        把梯度等对象通过 cold 通道发送（模拟 OSS）；
        实际仍然写本地文件，但你可以认为是“慢通道”。
        """
        key = self._cold_key(expert_id)
        data = dumps(obj)
        self._cold.set(key, data)
        log("comm", f"send_cold: expert_id={expert_id}, bytes={len(data)}")

    def pull_cold(self, expert_id: str, delete: bool = True) -> Optional[Dict[str, Any]]:
        """
        从 cold 通道拉取最近一次发送的对象；
        对于真正的 OSS，通常不会删除文件；但在模拟环境下可以根据需要删除。
        """
        key = self._cold_key(expert_id)
        data = self._cold.get(key, delete=delete)
        if data is None:
            return None
        obj = loads(data)
        log("comm", f"pull_cold: expert_id={expert_id}, bytes={len(data)}")
        return obj
