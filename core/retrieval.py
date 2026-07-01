"""4D-BioMem 里程碑 3：双通路检索层（Dual-Pathway Awakening）。

实现 SPEC §3 的双通路记忆唤醒架构，定量描述 P3 的认知决策边界：

  潜意识反射链（被动，硬检索）
      - 风险轨（R=1）：永久强制注入，作为隐式上下文常驻 Prompt（安全底线）。
      - 反射轨（R=0）：任务特征（F 轴）+ 时间流（T 轴）硬匹配，权重排序 Top-K_h。
      毫秒级，零向量计算。hard_confidence = 反射轨命中数。

  显意识搜索链（主动，软检索）
      - 高维语义向量（V 轴）余弦相似度软匹配 × 突触权重对数加成，Top-K_s。
      高算力，仅在被激活时调用。

  激活阈值 τ（Activation Threshold）
      - soft_activated = (hard_confidence < τ)
      - τ 定量刻画"潜意识够用则不调用显意识"的算力调配边界。

检索命中即"唤醒"：C_i += 1、刷新 last_accessed_at、重算权重（突触强化闭环），
把检索行为反馈回 M1 的权重演化与 M2 的持久化层。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Optional

import numpy as np

from core.memory_cell import MemoryCell, SynapticPruningEngine

VectorLookup = Callable[[str], Optional[np.ndarray]]

PATHWAY_REFLEX_RISK = "reflex_risk"
PATHWAY_REFLEX_TASK = "reflex_task"
PATHWAY_SOFT = "soft"


@dataclass
class RetrievalHit:
    """单条检索命中。"""

    cell: MemoryCell
    score: float
    pathway: str  # reflex_risk | reflex_task | soft | 合并标签 reflex_task+soft
    detail: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """双通路检索结果。"""

    hits: list[RetrievalHit]
    hard_confidence: float
    soft_activated: bool
    hard_hits_count: int   # 硬检索命中数（含风险+反射，合并去重前）
    soft_hits_count: int   # 软检索命中数（合并去重前）

    def cell_ids(self) -> set[str]:
        return {h.cell.id for h in self.hits}

    def pathways_of(self, cell_id: str) -> str:
        for h in self.hits:
            if h.cell.id == cell_id:
                return h.pathway
        return ""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tags_overlap(cell_tags: dict, query_tags: dict) -> bool:
    """F 轴硬过滤：query 的某个 (key, value) 在 cell 的 task_tags 中同时一致。"""
    if not cell_tags or not query_tags:
        return False
    for k, v in query_tags.items():
        if cell_tags.get(k) == v:
            return True
    return False


class DualPathwayRetriever:
    """双通路检索器。

    与存储层解耦：只依赖 engine（记忆池 + 权重 + 突触强化）与一个
    vector_lookup 回调（cell_id -> 向量）。生产环境 vector_lookup 接
    DBManager.get_vector；测试环境接内存字典。

    Parameters
    ----------
    engine : SynapticPruningEngine
        提供活动记忆池、权重计算（engine.lambda_ / engine.now()）、突触强化 access()。
    vector_lookup : Callable[[str], np.ndarray | None]
        cell_id -> 高维向量；返回 None 表示该记忆暂无向量（软检索跳过）。
    K_h : int
        潜意识反射链返回的非风险 Top-K（按权重排序）。
    K_s : int
        显意识搜索链返回的 Top-K（按相似度×权重加成排序）。
    tau : float
        激活阈值：hard_confidence < τ 时升级软检索。
    time_window : timedelta
        反射轨的时间流窗口：last_accessed_at ∈ [now-window, now] 才算反射命中。
    sim_floor : float
        软检索相似度下限，低于此值不召回。
    """

    def __init__(
        self,
        engine: SynapticPruningEngine,
        vector_lookup: VectorLookup,
        K_h: int = 5,
        K_s: int = 5,
        tau: float = 1.0,
        time_window: timedelta = timedelta(days=30),
        sim_floor: float = 0.3,
    ) -> None:
        self.engine = engine
        self.vector_lookup = vector_lookup
        self.K_h = K_h
        self.K_s = K_s
        self.tau = tau
        self.time_window = time_window
        self.sim_floor = sim_floor

    # ---- 潜意识反射链（硬检索）---------------------------------------------

    def hard_retrieve(self, query_tags: dict) -> tuple[list[RetrievalHit], float]:
        """风险强制常驻 + 任务/时间硬匹配 → 权重 Top-K_h。

        Returns
        -------
        hits : list[RetrievalHit]
            含 reflex_risk 与 reflex_task 两类命中。
        hard_confidence : float
            反射轨（非风险）命中数，作为 τ 判定输入。风险轨不计入置信度
            （它是无条件常驻，不反映"潜意识是否够用"）。
        """
        now = self.engine.now()
        lam = self.engine.lambda_
        hits: list[RetrievalHit] = []

        # 1) 风险轨：所有 R=1 记忆强制注入
        for cell in self.engine.active_cells():
            if cell.is_risk:
                w = cell.compute_weight(now, lam)
                hits.append(RetrievalHit(
                    cell=cell, score=w, pathway=PATHWAY_REFLEX_RISK,
                    detail={"reason": "risk_force_inject"},
                ))

        # 2) 反射轨：任务特征 ∩ + 时间流窗口
        window_start = now - self.time_window
        reflex_candidates: list[tuple[float, MemoryCell]] = []
        for cell in self.engine.active_cells():
            if cell.is_risk:
                continue
            if not _tags_overlap(cell.task_tags, query_tags):
                continue
            if cell.last_accessed_at < window_start:
                continue
            w = cell.compute_weight(now, lam)
            reflex_candidates.append((w, cell))

        reflex_candidates.sort(key=lambda x: x[0], reverse=True)
        reflex_hits: list[RetrievalHit] = []
        for w, cell in reflex_candidates[: self.K_h]:
            reflex_hits.append(RetrievalHit(
                cell=cell, score=w, pathway=PATHWAY_REFLEX_TASK,
                detail={"reason": "task+time_hard_match", "weight": w},
            ))

        hard_confidence = float(len(reflex_hits))
        return hits + reflex_hits, hard_confidence

    # ---- 显意识搜索链（软检索）---------------------------------------------

    def soft_retrieve(self, query_vector) -> list[RetrievalHit]:
        """高维语义相似度软匹配 × 突触权重对数加成 → Top-K_s。

        风险记忆已在硬检索强制注入，此处跳过（其权重 ∞ 会扭曲 score 排序）。
        """
        now = self.engine.now()
        lam = self.engine.lambda_
        qv = np.asarray(query_vector, dtype=np.float32)

        scored: list[tuple[float, float, MemoryCell]] = []  # (score, sim, cell)
        for cell in self.engine.active_cells():
            if cell.is_risk:
                continue
            vec = self.vector_lookup(cell.id)
            if vec is None:
                continue
            sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
            if sim < self.sim_floor:
                continue
            w = cell.compute_weight(now, lam)
            # 相似度为主，突触权重为对数加成：log(1+w) ∈ [log(1.5), log(1+∞))
            score = sim * math.log(1.0 + w)
            scored.append((score, sim, cell))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, sim, cell in scored[: self.K_s]:
            hits.append(RetrievalHit(
                cell=cell, score=score, pathway=PATHWAY_SOFT,
                detail={"sim": sim, "weight": cell.compute_weight(now, lam)},
            ))
        return hits

    # ---- 双通路编排 --------------------------------------------------------

    def retrieve(self, query_tags: dict, query_vector) -> RetrievalResult:
        """双通路检索主入口：硬检索 → τ 判定 → 可能升级软检索 → 合并去重 → 突触强化。"""
        hard_hits, hard_confidence = self.hard_retrieve(query_tags)
        soft_activated = hard_confidence < self.tau
        soft_hits: list[RetrievalHit] = []
        if soft_activated:
            soft_hits = self.soft_retrieve(query_vector)

        # 合并去重：同一 cell 可能同时被 reflex_task 与 soft 命中
        best_by_id: dict[str, RetrievalHit] = {}
        pathways_by_id: dict[str, set[str]] = defaultdict(set)
        for h in hard_hits + soft_hits:
            cid = h.cell.id
            pathways_by_id[cid].add(h.pathway)
            if cid not in best_by_id or h.score > best_by_id[cid].score:
                best_by_id[cid] = h

        # 多通路命中的，合并 pathway 标签
        for cid, hit in best_by_id.items():
            if len(pathways_by_id[cid]) > 1:
                hit.pathway = "+".join(sorted(pathways_by_id[cid]))

        # 突触强化：所有命中记忆 C_i += 1，刷新 last_accessed，重算权重
        for hit in best_by_id.values():
            self.engine.access(hit.cell.id)

        return RetrievalResult(
            hits=list(best_by_id.values()),
            hard_confidence=hard_confidence,
            soft_activated=soft_activated,
            hard_hits_count=len(hard_hits),
            soft_hits_count=len(soft_hits),
        )


__all__ = [
    "DualPathwayRetriever",
    "RetrievalHit",
    "RetrievalResult",
    "PATHWAY_REFLEX_RISK",
    "PATHWAY_REFLEX_TASK",
    "PATHWAY_SOFT",
]
