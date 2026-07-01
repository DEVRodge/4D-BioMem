"""4D-BioMem 里程碑 1 实验脚本：核心突触剪枝算法层验证。

场景设计（对齐 SPEC 第三部分 §3.2 评测数据集的三类线索）：
  - 闲聊  (casual)  : "今天中午吃了酸菜鱼"          R=0, I=2, 写入后永不再被检索
  - 技术  (tech)    : "项目 Alpha 的 Bug 修复方案"  R=0, I=5, 前 50 天每 5 天高频激活一次，后 50 天休眠
  - 过敏史(allergy) : "我对青霉素过敏"              R=1, I=10, 风险硬锁定，永不剪枝

加速衰减：λ=0.05/天（半衰期约 14 天），θ_prune=0.5。100 个模拟日在秒级真实时间内跑完。

预期新陈代谢：
  - 闲聊   : 约 day 20 跌破 θ → day 30 剪枝扫描时被物理抹除
  - 技术   : 高频激活累积 ln(1+C) 频次奖励，即便休眠 50 天仍 > θ → 存活
  - 过敏史 : R=1 → W=∞ → 永久存活，100% 召回
"""

from __future__ import annotations

import math
from datetime import timedelta

from core.memory_cell import (
    RISK_LOCKED_WEIGHT,
    MemoryCell,
    SynapticPruningEngine,
)

INF = RISK_LOCKED_WEIGHT


def fmt(w: float) -> str:
    return "INF" if w == INF else f"{w:.4f}"


def snapshot(engine: SynapticPruningEngine, day: int, cells: list[tuple[str, MemoryCell]]) -> None:
    """打印某一天的权重快照（只读，不触发剪枝）。已剪枝的记忆从 prune_log 回溯。"""
    print(f"\n=== Day {day:>3} 权重快照 ===")
    print(f"{'记忆':<10} {'R':<3} {'I':<4} {'C':<4} {'权重':>10}   {'状态'}")
    print("-" * 58)
    living_ids = {c.id for c in engine.active_cells()}
    for label, cell in cells:
        if cell.id in living_ids:
            w = cell.compute_weight(engine.now(), engine.lambda_)
            if cell.is_risk:
                status = "[LOCK]  风险锁定"
            elif w >= engine.theta_prune:
                status = "[OK]    存活"
            else:
                status = "[WARN]  濒死(<θ)"
            r_disp = int(cell.is_risk)
            print(f"{label:<10} {r_disp:<3} {cell.base_intensity:<4} {cell.access_count:<4} {fmt(w):>10}   {status}")
        else:
            rec = next(p for p in engine.prune_log if p.cell_id == cell.id)
            print(f"{label:<10} {'-':<3} {'-':<4} {rec.access_count:<4} {'(已抹除)':>10}   [PRUNED] 终末权重 {fmt(rec.final_weight)}")


def pruning_sweep(engine: SynapticPruningEngine, day: int, labels: dict[str, str]) -> None:
    """运行剪枝守护作业并打印结果。"""
    print(f"\n--- Day {day} 剪枝守护作业 ---")
    before = len(engine)
    pruned = engine.run_pruning()
    after = len(engine)
    print(f"  扫描 {before} 条 -> 保留 {after} 条 -> 物理抹除 {len(pruned)} 条")
    for p in pruned:
        label = labels.get(p.id, p.id[:8])
        print(f"  [PRUNED] {label:<8} content={p.content!r}")
        print(f"           终末权重={fmt(p.current_weight)}  C={p.access_count}  I={p.base_intensity}")


def assert_verdict(
    engine: SynapticPruningEngine,
    casual: MemoryCell,
    tech: MemoryCell,
    allergy: MemoryCell,
) -> bool:
    print("\n" + "=" * 64)
    print("实验断言验证")
    print("=" * 64)
    living = {c.id for c in engine.active_cells()}

    allergy_w = engine.get(allergy.id).current_weight
    tech_w = engine.get(tech.id).compute_weight(engine.now(), engine.lambda_)
    casual_rec = next(p for p in engine.prune_log if p.cell_id == casual.id)

    checks = [
        ("过敏史(风险锁定) 永久存活，未被剪枝", allergy.id in living),
        ("过敏史权重 == ∞", allergy_w == INF),
        ("技术方案(高频激活) 存活", tech.id in living),
        ("技术方案 day100 权重 > θ_prune (频次奖励生效)", tech_w > engine.theta_prune),
        ("闲聊(低频衰减) 已被物理剪枝", casual.id not in living),
        ("闲聊终末权重 < θ_prune", casual_rec.final_weight < engine.theta_prune),
        ("剪枝为不可逆物理抹除 (prune_log 留档)", casual_rec.cell_id == casual.id),
    ]

    all_pass = True
    for desc, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {desc}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(">>> 全部断言通过：三类记忆新陈代谢完全符合 SPEC 预期 <<<")
    else:
        print(">>> 存在失败断言，需要修正 <<<")
    return all_pass


def main() -> bool:
    print("=" * 64)
    print("4D-BioMem 里程碑 1 实验：核心突触剪枝算法层")
    print("=" * 64)
    print("超参：λ=0.05/天  θ_prune=0.5  (半衰期 ≈ 13.86 天)")
    print("场景：100 个模拟日；技术记忆前 50 天每 5 天激活，后 50 天休眠")

    engine = SynapticPruningEngine(lambda_=0.05, theta_prune=0.5)

    # Day 0：写入三条记忆
    casual = MemoryCell(
        content="今天中午吃了酸菜鱼",
        base_intensity=2.0,
        is_risk=False,
        task_tags={"type": "casual"},
    )
    tech = MemoryCell(
        content="项目 Alpha 的 Bug 修复方案：采用重试队列 + 幂等键",
        base_intensity=5.0,
        is_risk=False,
        task_tags={"type": "tech", "project": "Alpha"},
    )
    allergy = MemoryCell(
        content="我对青霉素过敏",
        base_intensity=10.0,
        is_risk=True,
        task_tags={"type": "medical"},
    )
    for c in (casual, tech, allergy):
        engine.register(c)

    cells = [("闲聊", casual), ("技术", tech), ("过敏史", allergy)]
    labels = {casual.id: "闲聊", tech.id: "技术", allergy.id: "过敏史"}

    snapshot(engine, 0, cells)

    # 模拟 100 天
    for day in range(1, 101):
        engine.tick(timedelta(days=1))
        # 技术记忆：前 50 天每 5 天激活一次（高频线索）
        if day % 5 == 0 and day <= 50:
            engine.access(tech.id)
        # 闲聊 & 过敏史：永不主动检索

        # 只读快照
        if day in (10, 20, 30, 50, 75, 100):
            snapshot(engine, day, cells)

        # 剪枝守护作业
        if day == 30:
            pruning_sweep(engine, day, labels)
        if day == 100:
            pruning_sweep(engine, day, labels)

    return assert_verdict(engine, casual, tech, allergy)


if __name__ == "__main__":
    ok = main()
    # 退出码供 CI 判定
    raise SystemExit(0 if ok else 1)
