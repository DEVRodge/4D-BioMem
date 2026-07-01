"""Group B — FIFO 滑动窗口 + 摘要压缩。

简化版 SOTA 基线，模拟 Mem0/MemGPT 的上下文压缩机制：
  - 保持最近 N 条原始记忆（滑动窗口，FIFO）。
  - 超出窗口时，最旧的 batch（N//5 条）被压缩为一条"摘要"条目
    （截断拼接 + [SUMMARY] 前缀，重新嵌入）。
  - 检索时同时搜索窗口与摘要。
  - 无权重衰减、无风险锁定、无显式剪枝。
  - 存储有界于 N + 摘要数。"""

from __future__ import annotations

from collections import deque

import numpy as np

from experiment.memory_system import EMBED_DIM, MemorySystem, VectorFunc

BATCH_FRACTION = 5      # 每 N//5 条旧记忆压缩为 1 条摘要


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class GroupB(MemorySystem):
    def __init__(self, window_size: int = 100, embedder: VectorFunc | None = None) -> None:
        self._size = window_size
        self._embed = embedder or self._default_embed
        self._batch = max(1, window_size // BATCH_FRACTION)
        self._window: deque[str] = deque()          # 最新原始记忆（FIFO）
        self._summaries: list[tuple[str, list[float]]] = []  # (摘要文本, 向量)

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

    def add(self, content: str, category: str = "") -> None:
        self._window.append(content)
        if len(self._window) > self._size:
            self._summarize_oldest_batch()

    def _summarize_oldest_batch(self) -> None:
        """弹出最旧的 batch 条记忆，压缩为一条摘要。"""
        items: list[str] = []
        for _ in range(self._batch):
            if not self._window:
                break
            items.append(self._window.popleft())
        if not items:
            return
        # 截断拼接 => 模拟 LLM 摘要（保留关键词片段，但丢失精确表述）
        truncated = "; ".join(it[:80] for it in items)
        summary_text = f"[SUMMARY] {truncated}"
        vec = self._embed(summary_text)
        self._summaries.append((summary_text, vec))

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        qv = np.asarray(self._embed(query), dtype=np.float32)
        scored: list[tuple[float, str]] = []
        # 搜索窗口记忆
        for content in self._window:
            vec = self._embed(content)
            sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
            scored.append((sim, content))
        # 搜索摘要
        for text, vec in self._summaries:
            sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
            scored.append((sim, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def get_all_contents(self) -> list[str]:
        return list(self._window) + [s[0] for s in self._summaries]

    def storage_count(self) -> int:
        return len(self._window) + len(self._summaries)

    def step_prune(self, simulate_days: float = 0.0) -> tuple[int, int]:
        return (0, 0)

    def reset(self) -> None:
        self._window.clear()
        self._summaries.clear()
