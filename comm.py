# comm.py
"""
本地通信模拟层：
- 使用本地目录 + 文件来模拟 Redis (hot) 和 OSS (cold)
- 适用于本地多进程 FastAPI 服务之间共享梯度 / 中间结果
- [Updated] 支持多文件并发写入与原子抢占读取，适配并行微批次训练
"""

import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from shared import dumps, loads  # 复用你已有的 msgpack 序列化工具
from utils.logger import log


class LocalKVStore:
    """
    用目录 + 文件模拟一个简单 KV 存储：
    - set(key, value_bytes): 把 value 写到 base_dir/key
    - get(key): 读 base_dir/key 的内容
    - list_keys(prefix): 列出匹配的 key
    """

    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        # key 允许有子目录，例如 "hot/0"
        return self.base.joinpath(*key.split("/"))

    def set(self, key: str, value: bytes) -> None:
        """
        写入数据。使用临时文件+重命名保证写入的原子性。
        """
        with self._lock:
            p = self._path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            # 使用临时文件写入，避免读到半截数据
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(value)
            # 原子替换 (POSIX compliant)
            tmp.replace(p)

    def get(self, key: str, delete: bool = False) -> Optional[bytes]:
        """
        读取数据。
        如果 delete=True，尝试使用重命名(rename)来原子性地“抢占”文件，
        防止多进程竞争导致的数据重复读取。
        """
        p = self._path(key)

        if not delete:
            # 普通读取，不删除
            if not p.exists():
                return None
            try:
                return p.read_bytes()
            except FileNotFoundError:
                return None

        # 删除模式 (一次性消费)
        # 策略：尝试将 file.bin 重命名为 file.bin.lock
        # 只有重命名成功的进程才有资格读取并删除它
        lock_p = p.with_suffix(p.suffix + ".lock")

        try:
            p.rename(lock_p)
        except OSError:
            # 重命名失败，说明文件不存在，或者被别人抢先了
            return None

        # 抢占成功，读取数据
        try:
            data = lock_p.read_bytes()
        except Exception as e:
            log("comm", f"Error reading locked file {lock_p}: {e}")
            data = None
        finally:
            # 无论读取成功与否，都要删除锁文件
            try:
                lock_p.unlink()
            except FileNotFoundError:
                pass

        return data

    def list_keys(self, prefix: str = "") -> List[str]:
        """
        列出所有以 prefix 开头的 key (relative path)。
        """
        res = []
        base_len = len(self.base.as_posix()) + 1
        # 遍历所有文件
        for f in self.base.rglob("*"):
            if not f.is_file():
                continue
            # 忽略临时文件和锁文件
            if f.name.endswith(".tmp") or f.name.endswith(".lock"):
                continue

            rel = f.as_posix()[base_len:]
            if prefix and not rel.startswith(prefix):
                continue
            res.append(rel)
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

    # -------- 通用辅助方法 --------

    def _generate_unique_key(self, expert_id: str) -> str:
        """生成带 UUID 的唯一文件名，防止并行写入冲突"""
        # 格式: {expert_id}_{uuid}.bin
        # 加下划线是为了方便 list_keys 时区分 "1" 和 "10"
        return f"{expert_id}_{uuid.uuid4().hex}.bin"

    def _pull_any(self, store: LocalKVStore, expert_id: str, delete: bool) -> Optional[Dict[str, Any]]:
        """从 store 中拉取属于 expert_id 的任意一个文件"""
        # 搜索前缀："{expert_id}_"
        prefix = f"{expert_id}_"

        # 1. 获取候选文件列表
        keys = store.list_keys(prefix)

        # 为了防止某种饥饿，可以简单排序
        keys.sort()

        # 2. 尝试读取（Competing Consumer）
        # 可能有多个 expert 实例同时在抢这些文件，所以需要循环尝试
        for key in keys:
            data = store.get(key, delete=delete)
            if data is not None:
                obj = loads(data)
                # log("comm", f"pull success: expert_id={expert_id}, key={key}, bytes={len(data)}")
                return obj

        # 3. 兼容旧逻辑：尝试读取不带 UUID 的固定文件名 "{expert_id}.bin"
        legacy_key = f"{expert_id}.bin"
        data = store.get(legacy_key, delete=delete)
        if data is not None:
            return loads(data)

        return None

    # -------- hot 通道（模拟 Redis） --------

    def send_hot(self, expert_id: str, obj: Dict[str, Any]) -> None:
        key = self._generate_unique_key(expert_id)
        data = dumps(obj)
        self._hot.set(key, data)
        log("comm", f"send_hot: expert_id={expert_id}, key={key}, bytes={len(data)}")

    def pull_hot(self, expert_id: str, delete: bool = True) -> Optional[Dict[str, Any]]:
        return self._pull_any(self._hot, expert_id, delete)

    # -------- cold 通道（模拟 OSS） --------

    def send_cold(self, expert_id: str, obj: Dict[str, Any]) -> None:
        key = self._generate_unique_key(expert_id)
        data = dumps(obj)
        self._cold.set(key, data)
        log("comm", f"send_cold: expert_id={expert_id}, key={key}, bytes={len(data)}")

    def pull_cold(self, expert_id: str, delete: bool = True) -> Optional[Dict[str, Any]]:
        return self._pull_any(self._cold, expert_id, delete)