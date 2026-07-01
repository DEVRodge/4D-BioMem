"""4D-BioMem 里程碑 4：自动化科学评测闭环。

模拟用户与 Hermes Agent 进行 30 轮高密度对话（硬编码语料：技术讨论 / 日常闲聊 /
安全隐私），通过 integrations/hermes_tools.py 的 remember_fact / recall_memory
工具灌入与检索记忆，中途与末尾各触发一次 /v1/memory/prune 加速衰减，最终计算
SPEC 第三部分约定的三项硬核指标：

  1. 高危召回率 Risk Recall       : 安全底线 / 机密隐私是否 100% 仍在。
  2. 上下文噪声抑制比             : 闲聊噪声被物理剪枝的比例。
  3. 存储空间收敛度               : 剪枝前后 Pickle / SQLite 文件体积变化曲线。

运行：python3 run_benchmark.py
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
from integrations import hermes_tools

# ---------------------------------------------------------------------------
# 30 轮对话语料：tech=技术讨论(高频高价值) / casual=闲聊噪声 / risk=安全隐私
# ---------------------------------------------------------------------------
CORPUS: list[dict[str, str]] = [
    {"cat": "tech",   "content": "项目 Alpha 的 Bug 修复方案采用重试队列加幂等键"},     # 1
    {"cat": "casual", "content": "今天天气挺好，适合散步"},                          # 2
    {"cat": "casual", "content": "中午吃了牛肉面，味道一般"},                        # 3
    {"cat": "tech",   "content": "项目 Alpha 的部署架构用 k8s 加双活"},              # 4
    {"cat": "risk",   "content": "我对青霉素过敏，开药千万别用青霉素类"},            # 5  风险
    {"cat": "casual", "content": "昨晚看了一部电影，挺无聊的"},                      # 6
    {"cat": "tech",   "content": "项目 Alpha 的 Bug 根因是并发竞态"},                # 7
    {"cat": "casual", "content": "早上喝了杯美式咖啡"},                              # 8
    {"cat": "casual", "content": "周末想去爬山"},                                    # 9
    {"cat": "tech",   "content": "项目 Alpha 的监控用 prometheus 加告警"},           # 10
    {"cat": "tech",   "content": "项目 Alpha 的 Bug 修复后压测通过"},                # 11
    {"cat": "casual", "content": "今天地铁有点挤"},                                  # 12
    {"cat": "risk",   "content": "服务器 root 密码是 Alpha-Bug-2024，务必保密"},     # 13 风险
    {"cat": "casual", "content": "晚饭吃了酸菜鱼"},                                  # 14
    {"cat": "casual", "content": "最近睡眠不太好"},                                  # 15
    {"cat": "tech",   "content": "项目 Alpha 的回滚方案是灰度发布"},                 # 16
    {"cat": "casual", "content": "今天有点累"},                                      # 17
    {"cat": "casual", "content": "中午和同事吃了火锅"},                              # 18
    {"cat": "tech",   "content": "项目 Alpha 的数据备份策略是每日全量"},             # 19
    {"cat": "casual", "content": "下班路上堵车了"},                                  # 20
    {"cat": "casual", "content": "早上吃了油条和豆浆"},                              # 21
    {"cat": "tech",   "content": "项目 Alpha 的安全审计修复了 SQL 注入"},            # 22
    {"cat": "casual", "content": "今天风挺大"},                                      # 23
    {"cat": "casual", "content": "晚上打了会儿游戏"},                                # 24
    {"cat": "tech",   "content": "项目 Alpha 的性能优化用上了缓存"},                 # 25
    {"cat": "casual", "content": "中午吃了炒饭"},                                    # 26
    {"cat": "casual", "content": "今天心情不错"},                                    # 27
    {"cat": "tech",   "content": "项目 Alpha 的文档已更新到 v2 版本"},               # 28
    {"cat": "casual", "content": "晚上看了会儿书"},                                  # 29
    {"cat": "casual", "content": "今天下班早点"},                                    # 30
]

USER_ID = "bench-user"
LAMBDA = 0.05
THETA = 0.5


# ---------------------------------------------------------------------------
# 服务启动
# ---------------------------------------------------------------------------


def _pick_port(prefer: int = 8000) -> int:
    try:
        s = socket.socket()
        s.bind(("127.0.0.1", prefer))
        s.close()
        return prefer
    except OSError:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p


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
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server did not become healthy")


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _list_items(cli: httpx.Client) -> list[dict]:
    r = cli.get("/v1/memory/list", params={"user_id": USER_ID})
    r.raise_for_status()
    return r.json()["items"]


def _flush_stable(cli: httpx.Client, stable_cycles: int = 4, interval: float = 0.15,
                  timeout: float = 20.0) -> int:
    """轮询 /list 直到计数连续 stable_cycles 次不变（异步写入已追平）。"""
    deadline = time.time() + timeout
    last = -1
    stable = 0
    while time.time() < deadline:
        n = len(_list_items(cli))
        if n == last:
            stable += 1
            if stable >= stable_cycles:
                return n
        else:
            stable = 0
            last = n
        time.sleep(interval)
    return last


def _prune(cli: httpx.Client, simulate_days: float) -> dict:
    r = cli.post("/v1/memory/prune", json={
        "user_id": USER_ID, "lambda_": LAMBDA, "theta_prune": THETA,
        "simulate_days": simulate_days,
    })
    r.raise_for_status()
    return r.json()


def _bar(value: float, max_value: float, width: int = 24) -> str:
    if max_value <= 0:
        return "." * width
    filled = int(round(width * min(value, max_value) / max_value))
    return "#" * filled + "." * (width - filled)


def _categorize(items: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    surviving = {it["content"] for it in items}
    counts = {"risk": 0, "tech": 0, "casual": 0}
    totals = {"risk": 0, "tech": 0, "casual": 0}
    for entry in CORPUS:
        totals[entry["cat"]] += 1
        if entry["content"] in surviving:
            counts[entry["cat"]] += 1
    return counts, totals


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> bool:
    print("=" * 72)
    print("4D-BioMem 科学评测闭环（30 轮高密度对话）")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="biomem_bench_")
    db_path = os.path.join(tmp, "biomem.db")
    vec_path = os.path.join(tmp, "vec_store")
    pickle_path = vec_path + ".pkl"
    port = _pick_port(8000)
    base = f"http://127.0.0.1:{port}"
    n_risk = sum(1 for c in CORPUS if c["cat"] == "risk")
    n_tech = sum(1 for c in CORPUS if c["cat"] == "tech")
    n_casual = sum(1 for c in CORPUS if c["cat"] == "casual")
    print(f"临时库: {tmp}")
    print(f"服务端口: {port}  (hermes_tools 默认 8000，已 configure 到实际端口)")
    print(f"语料: {len(CORPUS)} 条 | risk={n_risk} tech={n_tech} casual={n_casual}")

    holder: dict[str, Any] = {}
    t = threading.Thread(target=_start_server, args=(port, db_path, vec_path, holder), daemon=True)
    t.start()
    _wait_health(base)
    hermes_tools.configure(base_url=base, default_user=USER_ID)

    checkpoints: list[tuple[str, int, int, int]] = []  # (label, rows, pickle_bytes, sqlite_bytes)

    def record(label: str, rows: int) -> None:
        checkpoints.append((label, rows, _file_size(pickle_path), _file_size(db_path)))

    ok = True
    try:
        with httpx.Client(base_url=base, timeout=10.0) as cli:
            record("baseline (0)", 0)

            # ---- 30 轮对话 --------------------------------------------------
            print("\n--- 30 轮对话推进（每轮 remember_fact，每 5 轮 recall_memory 强化）---")
            for i, entry in enumerate(CORPUS, 1):
                cat = entry["cat"]
                content = entry["content"]
                res = hermes_tools.remember_fact(content)
                status = res.get("status", "??")
                tag = {"risk": "RISK", "tech": "TECH", "casual": "CASY"}[cat]
                print(f"  [Round {i:02d}] {tag} | {content[:30]:<32} | {status}")

                # 每 5 轮显意识回忆（强化技术记忆，模拟高频线索）
                if i % 5 == 0 and i < 30:
                    _flush_stable(cli)  # 先等异步写入追平，recall 才能看到全部记忆
                    rc = hermes_tools.recall_memory("项目 Alpha Bug 修复", top_k=2)
                    nhit = len(rc.get("hits", [])) if isinstance(rc, dict) else 0
                    print(f"           ↳ recall_memory(top_k=2) -> {nhit} hits (突触强化)")

                if i == 10:
                    n = _flush_stable(cli)
                    record("after round 10 (pre-prune)", n)

                # 第 15 轮：中途剪枝（SPEC：10~20 轮之间，加速衰减 25 天）
                if i == 15:
                    n = _flush_stable(cli)
                    record("after round 15 (pre mid-prune)", n)
                    p = _prune(cli, simulate_days=25)
                    print(f"           ✂ mid-prune(simulate=25d): scanned={p['scanned']} "
                          f"pruned={p['pruned']} survivors={p['survivors']}")
                    record("after round 15 (post mid-prune)", p["survivors"])

                if i == 20:
                    n = _flush_stable(cli)
                    record("after round 20", n)

            # 全部 30 轮灌完
            n30 = _flush_stable(cli)
            record("after round 30 (pre final-prune)", n30)

            # ---- 末尾剪枝：彻底清除残余噪声 ---------------------------------
            p = _prune(cli, simulate_days=30)
            print(f"\n--- 末尾剪枝 (simulate=30d): scanned={p['scanned']} "
                  f"pruned={p['pruned']} survivors={p['survivors']} ---")
            record("after final-prune", p["survivors"])

            # ---- 指标计算 ---------------------------------------------------
            _flush_stable(cli)  # 确保状态一致
            items = _list_items(cli)
            counts, totals = _categorize(items)
            print(f"\n--- 最终记忆池 ---")
            print(f"  剩余 {len(items)} 条 | risk={counts['risk']} tech={counts['tech']} casual={counts['casual']}")
            print(f"  (语料总数 risk={totals['risk']} tech={totals['tech']} casual={totals['casual']})")

            risk_recall = counts["risk"] / totals["risk"] if totals["risk"] else 0.0
            noise_suppression = 1.0 - (counts["casual"] / totals["casual"]) if totals["casual"] else 0.0

            pre = next((c for c in checkpoints if "pre final-prune" in c[0]), None)
            post = next((c for c in checkpoints if c[0] == "after final-prune"), None)
            pickle_pre = pre[2] if pre else 0
            pickle_post = post[2] if post else 0
            sqlite_pre = pre[3] if pre else 0
            sqlite_post = post[3] if post else 0
            pickle_reduction = (1.0 - pickle_post / pickle_pre) if pickle_pre else 0.0

            # ---- 检查点曲线 -------------------------------------------------
            print(f"\n--- 存储空间收敛曲线 ---")
            print(f"  {'检查点':<38} {'行数':>5} {'Pickle':>9} {'SQLite':>9}")
            max_pickle = max(c[2] for c in checkpoints) or 1
            for label, rows, pk, sq in checkpoints:
                bar = _bar(pk, max_pickle)
                print(f"  {label:<38} {rows:>5} {pk:>7}B {sq:>7}B  [{bar}]")

            # ---- 三指标 -----------------------------------------------------
            print(f"\n{'=' * 72}")
            print(f"科学评测指标汇总")
            print(f"{'=' * 72}")
            metrics = [
                ("高危召回率 Risk Recall",
                 f"{counts['risk']}/{totals['risk']} = {risk_recall*100:.1f}%",
                 risk_recall >= 1.0,
                 "安全/隐私记忆 100% 存活"),
                ("上下文噪声抑制比",
                 f"{totals['casual']-counts['casual']}/{totals['casual']} = {noise_suppression*100:.1f}%",
                 noise_suppression >= 0.9,
                 "闲聊噪声被物理剪枝"),
                ("存储空间收敛度 (Pickle 缩减)",
                 f"{pickle_pre}B -> {pickle_post}B  缩减 {pickle_reduction*100:.1f}%",
                 pickle_reduction > 0.3,
                 "剪枝后体积回落 O(N)->O(1)"),
            ]
            for name, val, passed, note in metrics:
                mark = "PASS" if passed else "FAIL"
                print(f"  [{mark}] {name:<28} {val:<30} {note}")
                if not passed:
                    ok = False

            tech_survival = counts["tech"] / totals["tech"] if totals["tech"] else 0.0
            print(f"  [INFO] 技术记忆存活率            {counts['tech']}/{totals['tech']} = {tech_survival*100:.1f}%  "
                  f"(高频线索应保留)")

    finally:
        server = holder.get("server")
        if server is not None:
            server.should_exit = True
        t.join(timeout=5.0)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if ok:
        print(">>> 科学评测闭环通过：4D-BioMem 三项硬核指标全部达标 <<<")
    else:
        print(">>> 存在未达标指标，需修正 <<<")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
