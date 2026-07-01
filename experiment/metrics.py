"""四项定量指标计算。"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiment.corpus import CORPUS, GROUND_TRUTH_QUERIES
from experiment.memory_system import MemorySystem


@dataclass
class MetricsResult:
    risk_recall: float
    precision_at_k: float
    storage_checkpoints: list[tuple[int, int]] = field(default_factory=list)
    context_noise_ratio: float = 0.0
    prune_stats: tuple[int, int, int] = (0, 0, 0)


def compute(c: MemorySystem, checkpoints: list[int]) -> MetricsResult:
    """计算四项指标。

    1. Risk Recall: 风险类别 GT 查询能否检索回风险记忆（而非仅库中存在）。
    2. Precision@K: GT 查询 top-K 中含预期关键词的比例。
    3. Storage Convergence: runner 外部填充，此处不计算。
    4. Context Noise Ratio: top-5 中闲聊噪声占比（无用干扰 Token）。
    """
    risk_content_set = {it.content for it in CORPUS if it.category == "risk"}
    casual_content_set = {it.content for it in CORPUS if it.category == "casual"}

    risk_gt_queries = [g for g in GROUND_TRUTH_QUERIES if g.expected_category == "risk"]

    risk_recall = 100.0
    if risk_gt_queries:
        risk_found = 0
        for gq in risk_gt_queries:
            results = c.retrieve(gq.query, top_k=gq.top_k)
            if any(r in risk_content_set for r in results):
                risk_found += 1
        risk_recall = risk_found / len(risk_gt_queries) * 100.0

    pk_hits = 0
    noise_ratios: list[float] = []
    for gq in GROUND_TRUTH_QUERIES:
        results = c.retrieve(gq.query, top_k=gq.top_k)
        result_text = " ".join(results)
        if any(kw in result_text for kw in gq.expected_keywords):
            pk_hits += 1

        top5 = c.retrieve(gq.query, top_k=5)
        if top5:
            noisy = sum(1 for r in top5 if r in casual_content_set)
            noise_ratios.append(noisy / len(top5) * 100.0)
        else:
            noise_ratios.append(100.0)

    n_gt = len(GROUND_TRUTH_QUERIES) or 1
    precision_at_k = pk_hits / n_gt * 100.0
    context_noise_ratio = sum(noise_ratios) / len(noise_ratios) if noise_ratios else 0.0

    return MetricsResult(
        risk_recall=round(risk_recall, 1),
        precision_at_k=round(precision_at_k, 1),
        storage_checkpoints=[],
        context_noise_ratio=round(context_noise_ratio, 1),
    )
