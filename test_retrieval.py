"""4D-BioMem 里程碑 3 实验脚本：双通路检索层验证。

五场景覆盖 SPEC §3 双通路唤醒架构的全部关键行为：
  S1 风险强制常驻 + 软检索升级   : 医药查询无任务标签 → 风险常驻 + 软检索打捞文档
  S2 硬反射够用，软检索被跳过     : Alpha 查询命中反射轨，τ=1 不升级（省算力）
  S3 τ 激活边界翻转              : 同 S2 查询但 τ=2，软检索被激活
  S4 突触强化闭环                : 检索命中后 C_i += 1，未命中记忆不变
  S5 时间窗硬过滤                : 陈旧记忆逃过反射轨，但被软检索按语义打捞

向量设计（128 维，正交基）：basis(0)..basis(4) 两两正交；
v_doc = normalize(basis(0) + 0.3·basis(4))，与 v_medical=basis(0) 余弦≈0.958。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from core.memory_cell import MemoryCell, SynapticPruningEngine
from core.retrieval import (
    PATHWAY_REFLEX_RISK,
    PATHWAY_REFLEX_TASK,
    PATHWAY_SOFT,
    DualPathwayRetriever,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
DIM = 128


# ---- 向量构造 -------------------------------------------------------------

def basis(idx: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[idx] = 1.0
    return v


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


V_MEDICAL = basis(0)                                   # allergy (R=1)
V_DOC = _normalize(basis(0) + 0.3 * basis(4))          # doc, sim≈0.958 与 medical
V_ALPHA = basis(1)                                     # tech_alpha
V_BETA = basis(2)                                      # tech_beta
V_CASUAL = basis(3)                                    # casual

Q_MEDICAL = basis(0)
Q_ALPHA = basis(1)


# ---- 池构造 ----------------------------------------------------------------

def build_pool(stale_alpha: bool = False):
    """构造 5 条记忆的引擎池。stale_alpha=True 时 tech_alpha 的 last_accessed 落到 60 天前。"""
    engine = SynapticPruningEngine(lambda_=0.05, theta_prune=0.5, start_time=T0)

    cells: dict[str, MemoryCell] = {}
    vectors: dict[str, np.ndarray] = {}

    def mk(cid, content, intensity, is_risk, tags, vec, access_count=1, last_accessed=None):
        la = last_accessed if last_accessed is not None else T0
        cell = MemoryCell(
            content=content,
            base_intensity=float(intensity),
            is_risk=is_risk,
            access_count=access_count,
            created_at=T0,
            last_accessed_at=la,
            id=cid,
            task_tags=tags,
        )
        engine.load_cell(cell)  # 按 sim_clock=T0 重算权重，保留 last_accessed_at
        cells[cid] = cell
        vectors[cid] = vec

    mk("allergy", "我对青霉素过敏", 10, True, {"type": "medical"}, V_MEDICAL)
    mk("doc", "青霉素用药禁忌文档", 5, False, {"type": "doc"}, V_DOC)
    mk("tech_alpha", "项目 Alpha Bug 修复方案", 5, False,
       {"project": "Alpha", "type": "tech"}, V_ALPHA, access_count=5,
       last_accessed=(T0 - timedelta(days=60)) if stale_alpha else None)
    mk("tech_beta", "项目 Beta 部署方案", 5, False,
       {"project": "Beta", "type": "tech"}, V_BETA)
    mk("casual", "今天吃了酸菜鱼", 2, False, {"type": "casual"}, V_CASUAL)

    lookup = lambda cid: vectors.get(cid)
    return engine, cells, vectors, lookup


def _hit_summary(res) -> str:
    lines = []
    for h in sorted(res.hits, key=lambda x: x.cell.id):
        lines.append(f"    {h.cell.id:<12} pathway={h.pathway:<20} score={h.score:.4f}")
    lines.append(f"    hard_confidence={res.hard_confidence}  soft_activated={res.soft_activated}")
    return "\n".join(lines)


# ---- 场景 ------------------------------------------------------------------

def scenario_s1() -> bool:
    """S1 风险强制常驻 + 软检索升级：医药查询无任务标签。"""
    print("\n=== S1 风险强制常驻 + 软检索升级 ===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    res = r.retrieve(query_tags={}, query_vector=Q_MEDICAL)
    print(_hit_summary(res))

    ids = res.cell_ids()
    checks = [
        ("allergy 经风险轨强制注入", "allergy" in ids),
        ("doc 经软检索打捞", "doc" in ids),
        ("soft_activated=True（hard_confidence=0 < τ=1）", res.soft_activated is True),
        ("tech_alpha 不在结果（无任务/语义匹配）", "tech_alpha" not in ids),
        ("casual 不在结果", "casual" not in ids),
        ("allergy pathway == reflex_risk",
         res.pathways_of("allergy") == PATHWAY_REFLEX_RISK),
        ("doc pathway == soft", res.pathways_of("doc") == PATHWAY_SOFT),
    ]
    return _report(checks)


def scenario_s2() -> bool:
    """S2 硬反射够用，软检索被跳过：Alpha 查询，τ=1。"""
    print("\n=== S2 硬反射够用 -> 软检索跳过（τ=1）===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    res = r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA)
    print(_hit_summary(res))

    ids = res.cell_ids()
    checks = [
        ("allergy 风险常驻", "allergy" in ids),
        ("tech_alpha 反射轨命中", "tech_alpha" in ids),
        ("soft_activated=False（hard_confidence=1 ≥ τ=1）", res.soft_activated is False),
        ("tech_alpha pathway == reflex_task", res.pathways_of("tech_alpha") == PATHWAY_REFLEX_TASK),
        ("tech_beta 不在结果（任务标签不匹配）", "tech_beta" not in ids),
        ("doc 不在结果", "doc" not in ids),
        ("casual 不在结果", "casual" not in ids),
    ]
    return _report(checks)


def scenario_s3() -> bool:
    """S3 τ 激活边界翻转：同 S2 查询但 τ=2，软检索被激活。"""
    print("\n=== S3 τ 激活边界翻转（τ=2）===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=2.0, sim_floor=0.3)
    res = r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA)
    print(_hit_summary(res))

    checks = [
        ("soft_activated=True（hard_confidence=1 < τ=2）", res.soft_activated is True),
        ("tech_alpha 命中（reflex+soft 合并）", "tech_alpha" in res.cell_ids()),
        ("tech_alpha pathway 含 soft", "soft" in res.pathways_of("tech_alpha")),
        ("tech_alpha pathway 含 reflex_task", "reflex_task" in res.pathways_of("tech_alpha")),
    ]
    return _report(checks)


def scenario_s4() -> bool:
    """S4 突触强化闭环：检索命中后 C_i += 1，未命中记忆不变。"""
    print("\n=== S4 突触强化闭环（access 反馈）===")
    engine, cells, _, lookup = build_pool()
    before = {cid: c.access_count for cid, c in cells.items()}
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA)
    after = {cid: c.access_count for cid, c in cells.items()}
    print(f"    access_count before: {before}")
    print(f"    access_count after : {after}")

    checks = [
        ("tech_alpha C: 5 -> 6（命中并被唤醒）", after["tech_alpha"] == before["tech_alpha"] + 1),
        ("allergy C: 1 -> 2（风险轨命中也唤醒）", after["allergy"] == before["allergy"] + 1),
        ("casual C 不变（未命中）", after["casual"] == before["casual"]),
        ("tech_beta C 不变（未命中）", after["tech_beta"] == before["tech_beta"]),
        ("doc C 不变（未命中）", after["doc"] == before["doc"]),
    ]
    return _report(checks)


def scenario_s5() -> bool:
    """S5 时间窗硬过滤：陈旧 tech_alpha 逃过反射轨，但被软检索按语义打捞。"""
    print("\n=== S5 时间窗硬过滤（陈旧记忆）===")
    engine, cells, _, lookup = build_pool(stale_alpha=True)
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0,
                             time_window=timedelta(days=30), sim_floor=0.3)
    res = r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA)
    print(_hit_summary(res))

    checks = [
        ("tech_alpha 不在反射轨（陈旧超窗）",
         res.pathways_of("tech_alpha") != PATHWAY_REFLEX_TASK),
        ("tech_alpha 经软检索打捞", "soft" in res.pathways_of("tech_alpha")),
        ("soft_activated=True（反射轨空 → confidence=0 < τ）", res.soft_activated is True),
        ("allergy 仍风险常驻", "allergy" in res.cell_ids()),
    ]
    return _report(checks)


def scenario_s6() -> bool:
    """S6 风险常驻不挤占非风险查询排序：技术查询先给技术命中，风险仍随结果返回。"""
    print("\n=== S6 风险常驻不挤占非风险查询排序 ===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    res = r.retrieve(query_tags={"project": "Alpha", "type": "tech"}, query_vector=Q_ALPHA)
    ordered_ids = [h.cell.id for h in res.hits]
    print(_hit_summary(res))
    print(f"    ordered_ids={ordered_ids}")

    checks = [
        ("tech_alpha 是技术查询的首位结果", ordered_ids[0] == "tech_alpha"),
        ("allergy 风险记忆仍在结果中常驻", "allergy" in ordered_ids),
    ]
    return _report(checks)


def scenario_s7() -> bool:
    """S7 风险类查询仍优先风险记忆：风险轨不能因排序优化被降级。"""
    print("\n=== S7 风险类查询仍优先风险记忆 ===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    res = r.retrieve(query_tags={"type": "medical"}, query_vector=Q_MEDICAL)
    ordered_ids = [h.cell.id for h in res.hits]
    print(_hit_summary(res))
    print(f"    ordered_ids={ordered_ids}")

    checks = [
        ("allergy 是风险查询的首位结果", ordered_ids[0] == "allergy"),
        ("doc 仍可经软检索进入结果", "doc" in ordered_ids),
    ]
    return _report(checks)


def scenario_s8() -> bool:
    """S8 实体 boost：语义分数接近时，实体匹配项优先。"""
    print("\n=== S8 实体 boost 优先实体匹配项 ===")
    engine = SynapticPruningEngine(lambda_=0.05, theta_prune=0.5, start_time=T0)
    vectors: dict[str, np.ndarray] = {}

    def mk(cid, content, entities):
        cell = MemoryCell(
            content=content,
            base_intensity=5.0,
            is_risk=False,
            access_count=1,
            created_at=T0,
            last_accessed_at=T0,
            id=cid,
            task_tags={"type": "tech"},
            entities=entities,
        )
        engine.load_cell(cell)
        vectors[cid] = Q_ALPHA

    mk("plain", "项目 Alpha 的普通技术记录", [])
    mk("entity", "项目 Alpha 的实体匹配技术记录",
       [{"name": "Alpha", "type": "project", "role": "subject"}])

    r = DualPathwayRetriever(
        engine,
        vector_lookup=lambda cid: vectors.get(cid),
        K_h=0,
        K_s=5,
        tau=1.0,
        sim_floor=0.3,
    )
    res = r.retrieve(
        query_tags={},
        query_vector=Q_ALPHA,
        query_entities=[{"name": "Alpha", "type": "project"}],
    )
    ordered_ids = [h.cell.id for h in res.hits]
    print(_hit_summary(res))
    print(f"    ordered_ids={ordered_ids}")

    checks = [
        ("entity 因实体 boost 排在 plain 前", ordered_ids[0] == "entity"),
    ]
    return _report(checks)


def scenario_s9() -> bool:
    """S9 force_soft：硬反射够用时也可显式执行软检索补充精确语义命中。"""
    print("\n=== S9 force_soft 强制软检索补充精确命中 ===")
    engine, cells, _, lookup = build_pool()
    r = DualPathwayRetriever(engine, lookup, K_h=5, K_s=5, tau=1.0, sim_floor=0.3)
    res = r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA, force_soft=True)
    print(_hit_summary(res))

    checks = [
        ("soft_activated=True（force_soft 覆盖 τ）", res.soft_activated is True),
        ("tech_alpha pathway 含 soft", "soft" in res.pathways_of("tech_alpha")),
    ]
    return _report(checks)


def scenario_s10() -> bool:
    """S10 多通路融合：force_soft 时用软语义分数压过泛化高权重硬命中。"""
    print("\n=== S10 force_soft 使用软语义分数排序 ===")
    engine = SynapticPruningEngine(lambda_=0.05, theta_prune=0.5, start_time=T0)
    vectors: dict[str, np.ndarray] = {}

    def mk(cid, content, access_count, vec):
        cell = MemoryCell(
            content=content,
            base_intensity=5.0,
            is_risk=False,
            access_count=access_count,
            created_at=T0,
            last_accessed_at=T0,
            id=cid,
            task_tags={"project": "Alpha", "type": "tech"},
        )
        engine.load_cell(cell)
        vectors[cid] = vec

    mk("generic", "项目 Alpha 的泛化高频记录", 20, V_BETA)
    mk("exact", "项目 Alpha 的精确语义记录", 1, Q_ALPHA)

    r = DualPathwayRetriever(
        engine,
        vector_lookup=lambda cid: vectors.get(cid),
        K_h=5,
        K_s=5,
        tau=1.0,
        sim_floor=0.3,
    )
    res = r.retrieve(query_tags={"project": "Alpha"}, query_vector=Q_ALPHA, force_soft=True)
    ordered_ids = [h.cell.id for h in res.hits]
    print(_hit_summary(res))
    print(f"    ordered_ids={ordered_ids}")

    checks = [
        ("exact 软语义命中排在 generic 高权重硬命中前", ordered_ids[0] == "exact"),
        ("exact pathway 含 soft", "soft" in res.pathways_of("exact")),
    ]
    return _report(checks)


def scenario_s11() -> bool:
    """S11 lexical boost：低频精确短语可压过高频但不相关的同项目记忆。"""
    print("\n=== S11 lexical boost 提升低频精确短语 ===")
    engine = SynapticPruningEngine(lambda_=0.05, theta_prune=0.5, start_time=T0)
    vectors: dict[str, np.ndarray] = {}

    def mk(cid, content, access_count):
        cell = MemoryCell(
            content=content,
            base_intensity=7.0,
            is_risk=False,
            access_count=access_count,
            created_at=T0,
            last_accessed_at=T0,
            id=cid,
            task_tags={"project": "Alpha", "type": "tech"},
        )
        engine.load_cell(cell)
        vectors[cid] = Q_ALPHA

    mk("generic", "项目 Alpha 的数据备份策略是每日全量", 20)
    mk("exact", "项目 Alpha 的安全审计修复了 SQL 注入", 1)

    r = DualPathwayRetriever(
        engine,
        vector_lookup=lambda cid: vectors.get(cid),
        K_h=5,
        K_s=5,
        tau=1.0,
        sim_floor=0.3,
    )
    res = r.retrieve(
        query_tags={"project": "Alpha"},
        query_vector=Q_ALPHA,
        force_soft=True,
        query_text="项目 Alpha 安全审计 SQL 注入怎么修的",
    )
    ordered_ids = [h.cell.id for h in res.hits]
    print(_hit_summary(res))
    print(f"    ordered_ids={ordered_ids}")

    checks = [
        ("exact 因 lexical boost 排在 generic 前", ordered_ids[0] == "exact"),
    ]
    return _report(checks)


def _report(checks: list[tuple[str, bool]]) -> bool:
    ok_all = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            ok_all = False
    return ok_all


def main() -> bool:
    print("=" * 64)
    print("4D-BioMem 里程碑 3 实验：双通路检索层")
    print("=" * 64)

    scenarios = [
        ("S1 风险常驻+软检索升级", scenario_s1),
        ("S2 硬反射够用跳过软检索", scenario_s2),
        ("S3 τ边界翻转", scenario_s3),
        ("S4 突触强化闭环", scenario_s4),
        ("S5 时间窗硬过滤", scenario_s5),
        ("S6 风险不挤占非风险排序", scenario_s6),
        ("S7 风险查询优先风险记忆", scenario_s7),
        ("S8 实体 boost 排序", scenario_s8),
        ("S9 force_soft 覆盖 τ", scenario_s9),
        ("S10 force_soft 使用软语义排序", scenario_s10),
        ("S11 lexical boost 精确短语", scenario_s11),
    ]
    results = []
    for name, fn in scenarios:
        ok = fn()
        results.append((name, ok))

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_pass = all(ok for _, ok in results)
    print()
    if all_pass:
        print(">>> 双通路检索层全部场景通过 <<<")
    else:
        print(">>> 存在失败场景，需修正 <<<")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
