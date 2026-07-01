"""Group A — Vanilla RAG。

纯向量检索基线：所有记忆作为点存入内存数组，无剪枝、无衰减、无风险锁定。
检索时计算查询向量与所有点的余弦距离，返回 Top-K。
存储线性增长 O(N)。"""

from __future__ import annotations

import numpy as np

from experiment.memory_system import EMBED_DIM, MemorySystem, VectorFunc


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class GroupA(MemorySystem):
    def __init__(self, embedder: VectorFunc | None = None) -> None:
        self._embed = embedder or self._default_embed
        self._items: list[tuple[str, str, list[float]]] = []  # (content, category, vector)

    @staticmethod
    def _default_embed(text: str) -> list[float]:
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

    def add(self, content: str, category: str) -> None:
        self._items.append((content, category, self._embed(content)))

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self._items:
            return []
        qv = np.asarray(self._embed(query), dtype=np.float32)
        scored = [(_cosine(qv, vec), content) for content, _, vec in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def get_all_contents(self) -> list[str]:
        return [c for c, _, _ in self._items]

    def storage_count(self) -> int:
        return len(self._items)

    def step_prune(self, simulate_days: float = 0.0) -> tuple[int, int]:
        return (0, 0)

    def reset(self) -> None:
        self._items.clear()
