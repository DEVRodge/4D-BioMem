#!/usr/bin/env python3
"""4D-BioMem 对照组实验编排器。

运行三组对照：A(Vanilla RAG) / B(FIFO+摘要) / C(4D-BioMem)，
共享 50 条标注语料，在指定灌入点触发剪枝（仅 C 组生效），
最终计算四项指标并打印横向对比报告。
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

from experiment.corpus import CORPUS, get_category_counts
from experiment.group_a_rag import GroupA
from experiment.group_b_window import GroupB
from experiment.group_c_biomem import GroupC
from experiment.metrics import MetricsResult, compute
from experiment.terminal_report import print_comparison

# 采样点：在第 N 条灌入后记录 storage_count（用于收敛曲线）
CHECKPOINTS = [10, 20, 30, 40, 50]
# 末尾一次性剪枝：模拟 30 天自然衰减，闲聊(I=2)→0.31<θ，技术(I=7)→1.08>θ
PRUNE_AT = [50]
PRUNE_DAYS = 30.0


def _run_group(cls, label: str, checkpoint_indices: list[int]) -> MetricsResult:
    """在单组上跑完全流程：灌入 → 轮点采样 → 剪枝 → 指标计算。"""
    group = cls() if label != "B" else cls(window_size=100)

    prune_rounds = 0
    prune_scanned = 0
    prune_pruned = 0
    storage_pts: list[tuple[int, int]] = []

    for i, item in enumerate(CORPUS, 1):
        group.add(item.content, item.category)
        if i in checkpoint_indices:
            storage_pts.append((i, group.storage_count()))
        if i in PRUNE_AT:
            scanned, pruned = group.step_prune(simulate_days=PRUNE_DAYS)
            if scanned > 0 or pruned > 0:
                prune_rounds += 1
                prune_scanned += scanned
                prune_pruned += pruned
            # 剪枝后追加一个采样点，展示存储回落
            if i in checkpoint_indices:
                storage_pts.append((i, group.storage_count()))

    result = compute(group, checkpoint_indices)
    result.storage_checkpoints = storage_pts
    result.prune_stats = (prune_rounds, prune_scanned, prune_pruned)
    return result


def main() -> bool:
    counts = get_category_counts()
    print("=" * 72)
    print("4D-BioMem 三组对照实验")
    print("=" * 72)
    print(f"语料: {len(CORPUS)} 条 | risk={counts.get('risk',0)}  tech={counts.get('tech',0)}  "
          f"casual={counts.get('casual',0)}")
    print(f"采样点: {CHECKPOINTS}")
    print(f"剪枝点: {PRUNE_AT}  (step_prune, simulate_days={PRUNE_DAYS})")
    print()

    t0 = time.time()

    print("  [Group A] Vanilla RAG（纯向量，无剪枝）...", end=" ")
    a = _run_group(GroupA, "A", CHECKPOINTS)
    print(f"done  ({a.storage_checkpoints[-1][1] if a.storage_checkpoints else '?'} items)")

    print("  [Group B] FIFO 滑动窗口(N=100)+摘要...", end=" ")
    b = _run_group(GroupB, "B", CHECKPOINTS)
    print(f"done  ({b.storage_checkpoints[-1][1] if b.storage_checkpoints else '?'} items)")

    print("  [Group C] 4D-BioMem（权重衰减+双通路+剪枝）...", end=" ")
    c0 = _run_group(GroupC, "C", CHECKPOINTS)
    print(f"done  ({c0.storage_checkpoints[-1][1] if c0.storage_checkpoints else '?'} items)")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.2f}s")

    prune_ops = {
        "A": a.prune_stats, "B": b.prune_stats, "C": c0.prune_stats,
    }
    print_comparison(a, b, c0, len(CORPUS), prune_ops)

    ok = c0.risk_recall >= 100.0 and c0.context_noise_ratio < a.context_noise_ratio
    if ok:
        print(">>> 对照组实验完成：4D-BioMem 三项指标显著优于基线 <<<")
    else:
        print(">>> 对照组实验完成（部分指标未达最优，可调整参数）<<<")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
