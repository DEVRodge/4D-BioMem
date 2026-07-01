"""4D-BioMem 对照组实验包。

A (Vanilla RAG) / B (FIFO+摘要) / C (4D-BioMem) 横向对比。
"""

from .group_a_rag import GroupA
from .group_b_window import GroupB
from .group_c_biomem import GroupC
from .memory_system import MemorySystem

__all__ = ["MemorySystem", "GroupA", "GroupB", "GroupC"]
