"""4D-BioMem API 端到端验证脚本。

在后台线程启动 FastAPI/uvicorn 服务（独立临时库），用 httpx 走完整闭环：
  1. 异步录入 3 条记忆（闲聊 / 技术 / 过敏史）—— 立即返回 queued
  2. 轮询 /list 等待后台 LLM 审计 + 写盘完成
  3. 校验 MockLLMAuditor：过敏史 is_risk=True，技术记忆带 project 标签
  4. 高频检索技术查询（top_k=2）—— 强化 tech + allergy，闲聊不进 top_k 不被强化
  5. 校验 tech 的 access_count 与 current_weight 上升，闲聊不变
  6. 调用 /prune 模拟 30 天衰减 —— 闲聊被物理抹除，技术 + 过敏史存活
  7. 校验 /list 仅剩 2 条
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time
from typing import Any

import httpx
import uvicorn

from api.main import create_app

BASE_USER = "user-e2e"
MEMORIES = [
    ("今天中午吃了酸菜鱼", "casual"),
    ("项目 Alpha 的 Bug 修复方案：采用重试队列和幂等键", "tech"),
    ("我对青霉素过敏，开药时要注意", "allergy"),
]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(port: int, db_path: str, vec_path: str, holder: dict) -> None:
    app = create_app(db_path=db_path, vector_path=vec_path, prefer_chroma=False, seed=False)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    holder["server"] = server
    server.run()


def _wait_health(base: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server did not become healthy")


def _find(items: list[dict], tag: str) -> dict:
    for it in items:
        if it["task_tags"].get("type") == tag:
            return it
    raise AssertionError(f"未找到 type={tag} 的记忆")


def main() -> bool:
    print("=" * 64)
    print("4D-BioMem API 端到端验证")
    print("=" * 64)

    tmp = tempfile.mkdtemp(prefix="biomem_api_")
    db_path = os.path.join(tmp, "biomem.db")
    vec_path = os.path.join(tmp, "vec_store")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"临时库: {tmp}  端口: {port}")

    holder: dict[str, Any] = {}
    t = threading.Thread(target=_start_server, args=(port, db_path, vec_path, holder), daemon=True)
    t.start()
    _wait_health(base)

    checks: list[tuple[str, bool]] = []

    try:
        # ---- Phase 1: 异步录入 --------------------------------------------
        print("\n=== Phase 1: POST /v1/memory/add（异步录入）===")
        with httpx.Client(base_url=base, timeout=5.0) as cli:
            for content, _tag in MEMORIES:
                r = cli.post("/v1/memory/add", json={"user_id": BASE_USER, "content": content})
                r.raise_for_status()
                body = r.json()
                print(f"  add {content[:24]:<26} -> {body['status']} (req={body['request_id'][:8]})")
                checks.append((f"add 立即返回 queued: {content[:12]}", body["status"] == "queued"))

            # ---- Phase 2: 轮询直到 3 条全部写盘 ----------------------------
            print("\n=== Phase 2: 轮询 /list 等待后台审计写盘 ===")
            items: list[dict] = []
            for _ in range(50):
                r = cli.get(f"/v1/memory/list", params={"user_id": BASE_USER})
                r.raise_for_status()
                items = r.json()["items"]
                if len(items) == 3:
                    break
                time.sleep(0.1)
            print(f"  载入 {len(items)} 条记忆")
            checks.append(("3 条记忆全部异步写盘", len(items) == 3))

            # ---- Phase 3: 校验 MockLLMAuditor ------------------------------
            print("\n=== Phase 3: 校验 LLM 审计结果 ===")
            casual = _find(items, "casual")
            tech = _find(items, "tech")
            allergy = _find(items, "medical")
            print(f"  casual : is_risk={casual['is_risk']} I={casual['base_intensity']} tags={casual['task_tags']}")
            print(f"  tech   : is_risk={tech['is_risk']} I={tech['base_intensity']} tags={tech['task_tags']}")
            print(f"  allergy: is_risk={allergy['is_risk']} I={allergy['base_intensity']} tags={allergy['task_tags']}")
            checks.append(("过敏史识别为 is_risk=True（关键词命中）", allergy["is_risk"] is True))
            checks.append(("闲聊识别为 is_risk=False", casual["is_risk"] is False))
            checks.append(("技术记忆提取 project=Alpha 标签", tech["task_tags"].get("project") == "Alpha"))
            checks.append(("技术记忆 type=tech", tech["task_tags"].get("type") == "tech"))

            # ---- Phase 4: 高频检索（top_k=2 只强化 tech+allergy，闲聊不进 top）---
            print("\n=== Phase 4: 高频检索技术查询（top_k=2，5 次）===")
            tech_init_c = tech["access_count"]
            tech_init_w = float(tech["current_weight"])
            casual_init_c = casual["access_count"]
            print(f"  检索前: tech C={tech_init_c} w={tech_init_w} | casual C={casual_init_c}")

            retrieve_hits_all_have_allergy = True
            for i in range(5):
                r = cli.post("/v1/memory/retrieve", json={
                    "user_id": BASE_USER, "query": "项目 Alpha Bug 修复", "top_k": 2,
                })
                r.raise_for_status()
                body = r.json()
                hit_ids = {h["id"] for h in body["hits"]}
                if not any(h["is_risk"] for h in body["hits"]):
                    retrieve_hits_all_have_allergy = False
                if i == 0:
                    print(f"  首次检索 pathways(A/B): {body['pathways']}")
                    for h in body["hits"]:
                        print(f"    hit: {h['content'][:24]:<26} pathways={h['pathways']} score={h['score']}")
            checks.append(("每次检索都强制返回风险记忆（allergy）", retrieve_hits_all_have_allergy))

            # ---- Phase 4b: 标签化技术查询不应被风险常驻挤占首位 ------------
            print("\n=== Phase 4b: 技术查询排序（风险常驻不挤占首位）===")
            r = cli.post("/v1/memory/retrieve", json={
                "user_id": BASE_USER,
                "query": "项目 Alpha Bug 修复",
                "top_k": 3,
                "query_tags": {"project": "Alpha", "type": "tech"},
            })
            r.raise_for_status()
            tagged_body = r.json()
            tagged_hits = tagged_body["hits"]
            for h in tagged_hits:
                print(f"    hit: {h['content'][:24]:<26} tags={h['task_tags']} pathways={h['pathways']}")
            checks.append(("标签化技术查询首位是 tech 记忆", tagged_hits[0]["task_tags"].get("type") == "tech"))
            checks.append(("风险记忆仍随结果返回", any(h["is_risk"] for h in tagged_hits)))

            # ---- Phase 5: 校验 tech 强化、闲聊不变 -------------------------
            print("\n=== Phase 5: 校验突触强化（access_count + 权重上升）===")
            r = cli.get("/v1/memory/list", params={"user_id": BASE_USER})
            r.raise_for_status()
            items2 = r.json()["items"]
            tech2 = _find(items2, "tech")
            casual2 = _find(items2, "casual")
            allergy2 = _find(items2, "medical")
            tech_now_c = tech2["access_count"]
            tech_now_w = float(tech2["current_weight"])
            casual_now_c = casual2["access_count"]
            print(f"  检索后: tech C={tech_now_c} w={tech_now_w} | casual C={casual_now_c}")
            checks.append(("tech access_count 上升（高频检索强化）", tech_now_c > tech_init_c))
            checks.append(("tech current_weight 上升（频次奖励 ln(1+C)）", tech_now_w > tech_init_w))
            checks.append(("casual 未进 top_k 故 access_count 不变", casual_now_c == casual_init_c))

            # ---- Phase 6: 主动新陈代谢（模拟 30 天衰减）--------------------
            print("\n=== Phase 6: POST /v1/memory/prune（模拟 30 天衰减）===")
            r = cli.post("/v1/memory/prune", json={
                "user_id": BASE_USER, "lambda_": 0.05, "theta_prune": 0.5, "simulate_days": 30,
            })
            r.raise_for_status()
            pbody = r.json()
            print(f"  scanned={pbody['scanned']} pruned={pbody['pruned']} survivors={pbody['survivors']}")
            for p in pbody["pruned_items"]:
                print(f"  [PRUNED] {p['content'][:24]:<26} final_weight={p['final_weight']} C={p['access_count']}")
            checks.append(("剪枝数 == 1（仅闲聊）", pbody["pruned"] == 1))

            # ---- Phase 7: 校验闲聊物理抹除，技术+过敏史存活 ----------------
            print("\n=== Phase 7: 校验最终记忆池 ===")
            r = cli.get("/v1/memory/list", params={"user_id": BASE_USER})
            r.raise_for_status()
            final_items = r.json()["items"]
            final_types = {it["task_tags"].get("type") for it in final_items}
            print(f"  剩余 {len(final_items)} 条: types={final_types}")
            checks.append(("闲聊(casual)被物理抹除", "casual" not in final_types))
            checks.append(("技术(tech)存活", "tech" in final_types))
            checks.append(("过敏史(medical)存活（风险锁定）", "medical" in final_types))
            checks.append(("最终记忆数 == 2", len(final_items) == 2))

            # ---- health 计数校验 ------------------------------------------
            r = cli.get("/health")
            r.raise_for_status()
            h = r.json()
            print(f"\n  /health: sqlite_count={h['sqlite_count']} vector_count={h['vector_count']}")
            checks.append(("/health sqlite_count == 2", h["sqlite_count"] == 2))
            checks.append(("/health vector_count == 2", h["vector_count"] == 2))

    finally:
        server = holder.get("server")
        if server is not None:
            server.should_exit = True
        t.join(timeout=5.0)
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 64)
    print("断言汇总")
    print("=" * 64)
    all_pass = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            all_pass = False
    print()
    if all_pass:
        print(">>> API 端到端闭环全部通过 <<<")
    else:
        print(">>> 存在失败项，需修正 <<<")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
