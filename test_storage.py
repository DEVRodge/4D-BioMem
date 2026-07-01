"""4D-BioMem 里程碑 2 实验脚本：双轨制存储层持久化与 Dual-Delete 验证。

对两种向量后端（NumpyVectorStore 默认 / ChromaVectorStore 可选）各跑一遍
四阶段流程，确认解耦后两条路径都自洽：
  Phase 1 — 双轨写入 3 条记忆（含 128 维伪造向量），关闭连接。
  Phase 2 — 重新打开数据库，验证元数据与向量从硬盘完整加载（round-trip）。
  Phase 3 — 载入剪枝引擎，推进 30 模拟日，运行剪枝，对死亡记忆执行 Dual-Delete。
  Phase 4 — 校验被抹除记忆在 SQLite 与向量库同时消失，存活记忆双端俱在。

剪枝场景（λ=0.05/天，θ=0.5，Δt=30 天）：
  casual  (I=1,  C=1,  R=0) : 1·ln2·e^-1.5  = 0.1546 < 0.5  -> 剪枝
  tech    (I=8,  C=10, R=0) : 8·ln11·e^-1.5 = 4.276  > 0.5   -> 存活（频次奖励）
  allergy (I=10, R=1)       : INF                       -> 存活（风险锁定）
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

from core.memory_cell import MemoryCell, SynapticPruningEngine
from storage.db_manager import DBManager

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
LAMBDA = 0.05
THETA = 0.5
DIM = 128


def make_cell(cell_id: str, content: str, intensity: float, is_risk: bool, access_count: int = 1) -> MemoryCell:
    cell = MemoryCell(
        content=content,
        base_intensity=intensity,
        is_risk=is_risk,
        access_count=access_count,
        created_at=T0,
        last_accessed_at=T0,
        id=cell_id,
        task_tags={"type": cell_id.split("-")[0]},
    )
    cell.current_weight = cell.compute_weight(T0, LAMBDA)
    return cell


def fake_vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(DIM).astype(np.float32)


def run_persistence_test(prefer_chroma: bool) -> bool:
    backend = "chroma" if prefer_chroma else "numpy"
    print(f"\n{'#' * 64}")
    print(f"# 后端：{backend}（prefer_chroma={prefer_chroma}）")
    print(f"{'#' * 64}")

    tmp = tempfile.mkdtemp(prefix=f"biomem_{backend}_")
    db_path = os.path.join(tmp, "biomem.db")
    vec_path = os.path.join(tmp, "vec_store")
    print(f"临时数据目录: {tmp}")

    seed_cells = [
        (make_cell("casual-1", "今天中午吃了酸菜鱼", 1.0, False, 1), fake_vector(1)),
        (make_cell("tech-1", "项目 Alpha Bug 修复方案：重试队列 + 幂等键", 8.0, False, 10), fake_vector(2)),
        (make_cell("allergy-1", "我对青霉素过敏", 10.0, True, 1), fake_vector(3)),
    ]

    # ---- Phase 1: 双轨写入 --------------------------------------------------
    print("\n=== Phase 1: 双轨写入（save_memory）===")
    db = DBManager(db_path=db_path, vector_path=vec_path, prefer_chroma=prefer_chroma)
    print(f"向量后端实例: {db.vector_backend}")
    for cell, vec in seed_cells:
        db.save_memory(cell, vec)
    print(f"SQLite 行数 = {db.count_sqlite()}    向量库数 = {db.count_vectors()}")
    assert db.count_sqlite() == 3 and db.count_vectors() == 3, f"[{backend}] 写入后双端计数应为 3"
    db.close()
    print("-> 已关闭数据库连接（数据落盘）")

    # ---- Phase 2: 重开，验证持久化 -----------------------------------------
    print("\n=== Phase 2: 重开数据库 -> 验证硬盘加载 ===")
    db2 = DBManager(db_path=db_path, vector_path=vec_path, prefer_chroma=prefer_chroma)
    print(f"重开后向量后端实例: {db2.vector_backend}")
    loaded = db2.load_all_active_cells()
    print(f"load_all_active_cells() 载入: {len(loaded)} 条")
    assert len(loaded) == 3, f"[{backend}] 重开应载入 3 条"

    loaded_by_id = {c.id: c for c in loaded}
    for cell, vec in seed_cells:
        lc = loaded_by_id[cell.id]
        assert lc.content == cell.content, f"[{backend}] content mismatch {cell.id}"
        assert lc.is_risk == cell.is_risk
        assert lc.base_intensity == cell.base_intensity
        assert lc.access_count == cell.access_count
        assert lc.task_tags == cell.task_tags
        assert lc.created_at == cell.created_at
        assert lc.last_accessed_at == cell.last_accessed_at
        lv = db2.get_vector(cell.id)
        assert lv is not None, f"[{backend}] vector missing {cell.id}"
        assert np.allclose(np.asarray(lv, dtype=np.float32), vec, atol=1e-6), f"[{backend}] vector drift {cell.id}"
    print(f"  [{backend}] 元数据 + 向量 round-trip 全部一致")

    # ---- Phase 3: 载入引擎，剪枝，Dual-Delete ------------------------------
    print("\n=== Phase 3: 载入引擎 -> 剪枝 -> Dual-Delete ===")
    engine = SynapticPruningEngine(lambda_=LAMBDA, theta_prune=THETA, start_time=T0)
    engine.tick(timedelta(days=30))
    for cell in loaded:
        engine.load_cell(cell)
    pruned = engine.run_pruning()
    pruned_ids = [p.id for p in pruned]
    print(f"剪枝守护作业: 死亡 {len(pruned)} 条 -> {pruned_ids}")
    for p in pruned:
        db2.delete_memory(p.id)
        print(f"  Dual-Delete {p.id:<12} 终末权重={p.current_weight:.4f}")

    # ---- Phase 4: 双端一致性校验 ------------------------------------------
    print("\n=== Phase 4: 双端一致性校验 ===")
    expected_pruned = {"casual-1"}
    expected_survivors = {"tech-1", "allergy-1"}

    checks = []
    for pid in expected_pruned:
        checks.append((f"{pid} 已从 SQLite 抹除", not db2.sqlite_has(pid)))
        checks.append((f"{pid} 已从向量库抹除", not db2.vector_has(pid)))
    for sid in expected_survivors:
        checks.append((f"{sid} 仍在 SQLite", db2.sqlite_has(sid)))
        checks.append((f"{sid} 仍在向量库", db2.vector_has(sid)))
    checks.append(("SQLite 剩余行数 == 2", db2.count_sqlite() == 2))
    checks.append(("向量库剩余数 == 2", db2.count_vectors() == 2))
    checks.append(("剪枝结果 == {casual-1}", set(pruned_ids) == expected_pruned))

    all_pass = True
    for desc, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] [{backend}] {desc}")
        if not ok:
            all_pass = False

    db2.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return all_pass


def main() -> bool:
    print("=" * 64)
    print("4D-BioMem 里程碑 2 实验：双轨制存储层（双后端验证）")
    print("=" * 64)

    results = {}
    # 默认后端：numpy（轻量）
    results["numpy"] = run_persistence_test(prefer_chroma=False)
    # 可选后端：chroma（若已安装）
    try:
        import chromadb  # noqa: F401
        results["chroma"] = run_persistence_test(prefer_chroma=True)
    except ImportError:
        print("\n[skip] chromadb 未安装，跳过 chroma 后端测试（默认 numpy 已覆盖）")
        results["chroma"] = True  # 不计为失败

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for backend, ok in results.items():
        print(f"  [{backend}] {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    print()
    if all_pass:
        print(">>> 存储层持久化 + Dual-Delete（双后端）全部通过 <<<")
    else:
        print(">>> 存在失败项，需修正 <<<")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
