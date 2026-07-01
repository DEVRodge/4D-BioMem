"""MemorySystem 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

EMBED_DIM = 128
VectorFunc = Callable[[str], list[float]]


class MemorySystem(ABC):
    """所有记忆系统的统一接口。

    三组在同一个标注语料上运行，通过此接口计算指标，确保对比公平。
    """

    @abstractmethod
    def add(self, content: str, category: str) -> None:
        """灌入一条记忆。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """根据查询返回 top-K 条内容字符串（按相关性降序）。"""

    @abstractmethod
    def get_all_contents(self) -> list[str]:
        """返回当前所有存活的记忆内容列表。"""

    @abstractmethod
    def storage_count(self) -> int:
        """当前存储条目数。"""

    @abstractmethod
    def step_prune(self, simulate_days: float = 0.0) -> tuple[int, int]:
        """运行一轮剪枝。返回 (scanned, pruned)。无剪枝机制则 no-op 返回 (0,0)。"""

    @abstractmethod
    def reset(self) -> None:
        """清空所有记忆（重新运行时用）。"""

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def label(self) -> str:
        """简短标签，用于报告表格列头。"""
        return {
            "GroupA": "A (RAG)",
            "GroupB": "B (SW+Sum)",
            "GroupC": "C (4D-BioMem)",
        }.get(type(self).__name__, type(self).__name__)
