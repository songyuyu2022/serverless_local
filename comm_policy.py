# comm_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import random
import time

Mode = Literal["hot", "cold", "http", "local", "fallback"]

@dataclass
class CommResult:
    mode: Mode
    latency_ms: float
    bytes_sent: int
    ok: bool
    reason: str

class CommPolicy:
    """
    模拟不同通信通道的延迟与失败：
    - hot: 低延迟、小概率失败
    - cold: 高延迟、较稳定
    - http: 中等延迟、稳定
    - local: 几乎 0（用于“同机/同进程”模拟）
    - fallback: 兜底
    """
    def __init__(
        self,
        hot_ms: Tuple[float, float] = (1.0, 3.0),
        cold_ms: Tuple[float, float] = (15.0, 40.0),
        http_ms: Tuple[float, float] = (5.0, 12.0),
        hot_fail_p: float = 0.05,
        http_fail_p: float = 0.01,
        cold_fail_p: float = 0.005,
        enable_local: bool = False,
        rng_seed: int = 1234,
    ):
        self.hot_ms = hot_ms
        self.cold_ms = cold_ms
        self.http_ms = http_ms
        self.hot_fail_p = hot_fail_p
        self.http_fail_p = http_fail_p
        self.cold_fail_p = cold_fail_p
        self.enable_local = enable_local
        self.rng = random.Random(rng_seed)

    def _sample_ms(self, lo_hi: Tuple[float, float]) -> float:
        lo, hi = lo_hi
        return lo + (hi - lo) * self.rng.random()

    def send(
        self,
        prefer: Mode,
        bytes_sent: int,
        deadline_ms: float,
        elapsed_ms: float,
        allow_fallback: bool = True,
    ) -> CommResult:
        """
        deadline_ms: 本 step 的 budget
        elapsed_ms: 已经消耗的时间（用于触发切换）
        """
        # deadline 紧张时强制走更快的通道
        slack = deadline_ms - elapsed_ms
        if slack < 8.0 and self.enable_local:
            return CommResult("local", 0.2, bytes_sent, True, "tight_deadline->local")
        if slack < 8.0 and prefer in ("cold",):
            prefer = "http"

        def attempt(mode: Mode) -> CommResult:
            if mode == "local":
                return CommResult("local", 0.2, bytes_sent, True, "local")
            if mode == "hot":
                lat = self._sample_ms(self.hot_ms)
                ok = self.rng.random() > self.hot_fail_p
                return CommResult("hot", lat, bytes_sent, ok, "hot")
            if mode == "cold":
                lat = self._sample_ms(self.cold_ms)
                ok = self.rng.random() > self.cold_fail_p
                return CommResult("cold", lat, bytes_sent, ok, "cold")
            if mode == "http":
                lat = self._sample_ms(self.http_ms)
                ok = self.rng.random() > self.http_fail_p
                return CommResult("http", lat, bytes_sent, ok, "http")
            return CommResult("fallback", 0.0, bytes_sent, False, "fallback")

        r = attempt(prefer)
        if r.ok:
            return r

        if not allow_fallback:
            return CommResult("fallback", r.latency_ms, bytes_sent, False, f"{r.reason}->no_fallback")

        # fallback 次序：hot/cold -> http -> local(若开)
        if prefer != "http":
            r2 = attempt("http")
            if r2.ok:
                r2.reason = f"{r.reason}->http"
                return r2
        if self.enable_local:
            r3 = attempt("local")
            r3.reason = f"{r.reason}->local"
            return r3

        return CommResult("fallback", r.latency_ms, bytes_sent, False, f"{r.reason}->fail")
