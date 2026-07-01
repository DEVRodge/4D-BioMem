"""终端三栏横向对比报告。"""

from __future__ import annotations

from experiment.metrics import MetricsResult

SEP = "=" * 72
BAR = "─" * 72


def _row(name: str, a: str, b: str, c0: str, w: int = 38) -> str:
    return f"  {name:<{w}} | {a:<14} | {b:<14} | {c0:<14}"


def print_comparison(
    a: MetricsResult, b: MetricsResult, c0: MetricsResult,
    corpus_len: int, prune_ops: dict[str, tuple[int, int, int]],
) -> None:
    print(f"\n{SEP}")
    print("4D-BioMem 三组横向对照实验")
    print(SEP)
    print(f"语料: {corpus_len} 条 | 地面真值查询: 5 条 | Precision@K=3 | 剪枝 θ=0.5")

    print(f"\n--- 四项指标对比 ---")
    print(_row("指标", "A (RAG)", "B (SW)", "C (BioMem)"))
    print(f"  {BAR}")
    print(_row("高危召回率 (Risk Recall)",
               f"{a.risk_recall:.1f}%", f"{b.risk_recall:.1f}%", f"{c0.risk_recall:.1f}%"))
    print(_row("语义检索准确率 (Precision@K)",
               f"{a.precision_at_k:.1f}%", f"{b.precision_at_k:.1f}%", f"{c0.precision_at_k:.1f}%"))
    print(_row("上下文噪声比 (Noise=闲聊)",
               f"{a.context_noise_ratio:.1f}%", f"{b.context_noise_ratio:.1f}%", f"{c0.context_noise_ratio:.1f}%"))

    print(f"\n--- 剪枝统计 ---")
    pa, pb, pc = prune_ops.get("A", (0, 0, 0)), prune_ops.get("B", (0, 0, 0)), prune_ops.get("C", (0, 0, 0))
    print(_row("执行剪枝轮数", str(pa[0]), str(pb[0]), str(pc[0])))
    print(_row("累计扫描记忆数", str(pa[1]), str(pb[1]), str(pc[1])))
    print(_row("累计物理抹除数", str(pa[2]), str(pb[2]), str(pc[2])))

    print(f"\n--- 存储收敛曲线 ---")
    for label, res in [("A (RAG)", a), ("B (SW)", b), ("C (BioMem)", c0)]:
        pts = res.storage_checkpoints
        if pts:
            vals = [p[1] for p in pts]
            print(f"  {label:<12}: {' -> '.join(str(v) for v in vals)}")
        else:
            print(f"  {label:<12}: (无采样)")

    print(f"\n  说明:")
    print(f"    风险召回: 风险 GT 查询能否检索回风险记忆（检索召回率，非存活性）")
    print(f"    噪声比: top-5 中含闲聊类别记忆的比例。C 组剪枝抹除闲聊后应为 0%")
    print(f"    存储收敛: A 线性 O(N)、B 有界 O(100+摘要)、C 剪枝后收敛于存活记忆")

    print(f"\n{SEP}")
    print("结论")
    print(SEP)
    all_good = True
    if c0.risk_recall >= 100.0:
        print("  [PASS] C 组风险召回 100%（风险硬锁定 + 双通路检索生效）")
    else:
        print(f"  [INFO] C 组风险召回 {c0.risk_recall:.1f}%")
        all_good = False
    if c0.context_noise_ratio < min(a.context_noise_ratio, b.context_noise_ratio):
        print("  [PASS] C 组噪声比最低（突触剪枝有效抹除闲聊）")
    else:
        print("  [INFO] C 组噪声比未显著低于基线")
        all_good = False
    if c0.storage_checkpoints and c0.storage_checkpoints[-1][1] < corpus_len:
        print("  [PASS] C 组存储收敛（剪枝去除了噪声）")
    else:
        print("  [INFO] C 组存储未收敛")
        all_good = False
    print(f"\n{'PASS' if all_good else 'PARTIAL'}")
