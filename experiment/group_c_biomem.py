"""Group C — 4D-BioMem 全量系统。

直接实例化 core/ 的算法组件（不启动 HTTP 服务），与 A/B 组在同一进程内
公平对比。使用 SynapticPruningEngine（权重衰减 + 剪枝）+ DualPathwayRetriever
（双通路检索）+ MockLLMAuditor（风险检测）+ MockEmbedder。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.memory_cell import MemoryCell, SynapticPruningEngine
from core.retrieval import DualPathwayRetriever, RetrievalResult
from experiment.memory_system import EMBED_DIM, MemorySystem


def _embed(text: str) -> list[float]:
    """同步向量化（复用 MockEmbedder 的 4-gram hash 算法，不依赖 api.main）。"""
    import hashlib
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    grams = [text[i:i + 4] for i in range(max(1, len(text) - 3))]
    if not grams:
        grams = [text or "_"]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
    h2 = int(hashlib.md5((text or "_").encode()).hexdigest(), 16)
    vec[(h2 // 7) % EMBED_DIM] += 2.0
    n = float(np.linalg.norm(vec))
    return (vec / n).tolist() if n > 0 else vec.tolist()


def _audit(content: str) -> tuple[bool, float, dict[str, Any]]:
    """同步风险审计（等价于 MockLLMAuditor，内联以避免依赖 api.main）。"""
    is_risk = any(kw in content for kw in ("过敏", "青霉素", "密码", "secret", "API Key",
                                            "哮喘", "身份证", "社保卡", "账号"))
    text = content.lower()
    if is_risk:
        intensity = 10.0
    elif any(s in text for s in ("项目", "bug", "重试", "架构", "部署", "监控", "审计",
                                   "容器", "集成", "迁移", "测试")):
        intensity = 7.0
    elif any(s in content for s in ("吃了", "今天", "中午", "天气")):
        intensity = 2.0
    else:
        intensity = 2.0
    tags: dict[str, Any] = {}
    if is_risk:
        tags["type"] = "medical" if "药" in content else "secret"
    elif any(k in content for k in ("项目", "bug", "修复")):
        tags["type"] = "tech"
        import re
        m = re.search(r"项目\s*([A-Za-z0-9_\-]+)", content)
        if m:
            tags["project"] = m.group(1)
    elif any(k in content for k in ("吃了", "今天", "天气")):
        tags["type"] = "casual"
    else:
        tags["type"] = "general"
    return is_risk, intensity, tags


class GroupC(MemorySystem):
    def __init__(self, lambda_: float = 0.05, theta_prune: float = 0.5,
                 tau: float = 1.0) -> None:
        self._lambda = lambda_
        self._theta = theta_prune
        self._tau = tau
        self._now = datetime(2024, 1, 1, tzinfo=timezone.utc)  # 模拟时钟起始
        self._vectors: dict[str, list[float]] = {}
        self._create_engine()

    def _create_engine(self) -> None:
        self._engine = SynapticPruningEngine(
            lambda_=self._lambda, theta_prune=self._theta, start_time=self._now,
        )
        self._retriever = DualPathwayRetriever(
            engine=self._engine,
            vector_lookup=lambda cid: (
                np.asarray(self._vectors[cid], dtype=np.float32)
                if cid in self._vectors else None
            ),
            tau=self._tau, sim_floor=0.3,
        )

    def add(self, content: str, category: str = "") -> None:
        is_risk, intensity, tags = _audit(content)
        cell = MemoryCell(
            content=content,
            user_id="bench",
            agent_id="group-c",
            is_risk=is_risk,
            base_intensity=intensity,
            access_count=1,
            created_at=self._now,
            last_accessed_at=self._now,
            task_tags=tags,
        )
        self._engine.register(cell)
        vec = _embed(content)
        self._vectors[cell.id] = vec

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        _, _, tags = _audit(query)
        qv = np.asarray(_embed(query), dtype=np.float32)
        result: RetrievalResult = self._retriever.retrieve(query_tags=tags, query_vector=qv)
        return [h.cell.content for h in result.hits[:top_k]]

    def get_all_contents(self) -> list[str]:
        return [c.content for c in self._engine.active_cells()]

    def storage_count(self) -> int:
        return len(self._engine)

    def step_prune(self, simulate_days: float = 0.0) -> tuple[int, int]:
        from datetime import timedelta
        self._engine.tick(timedelta(days=simulate_days))
        pruned = self._engine.run_pruning()
        # 清除已剪枝记忆的向量
        pruned_ids = {p.id for p in pruned}
        for pid in pruned_ids:
            self._vectors.pop(pid, None)
        # 引擎已扫了全量，但 step_prune 接口约定返回 (scanned, pruned)
        return (len(pruned) + len(self._engine), len(pruned))

    def reset(self) -> None:
        self._vectors.clear()
        self._now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self._create_engine()
