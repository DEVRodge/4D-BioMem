"""4D-BioMem API 服务层：FastAPI 异步记忆服务（生产版）。

把 M1（剪枝引擎）+ M2（双轨存储）+ M3（双通路检索）缝合为 HTTP 服务：

  POST /v1/memory/add       异步录入：立即返回 queued，后台 LLM 审计 → 组装 MemoryCell → save_memory
  POST /v1/memory/retrieve  双通路唤醒：A 硬过滤(近期+风险+标签+agent) + B 软匹配(向量+实体boost) → 去重融合 → access_count += 1
  POST /v1/memory/synthesize 跨记忆合成：检索 Top-K → LLM 综合回答（Mock 模式返回拼接摘要）
  POST /v1/memory/prune     主动新陈代谢：扫描非风险记忆，物理抹除 W_i(t) < θ_prune 的死亡记忆
  GET  /v1/memory/list      列出某用户全部记忆（调试用）
  GET  /health              健康检查

LLM/Embedding 默认用 Mock（无需 API Key）；设置 LLM_BACKEND=openai + OPENAI_API_KEY 启用 OpenAI。
所有 /v1/ 路由受 X-API-Key 头部保护（配置 API_KEY 启用，空则开放）。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from core.retrieval import _tags_overlap
from core.llm_auditor import OpenAIEmbedder, OpenAILLMAuditor, MockEmbedder, MockLLMAuditor
from core.memory_cell import RISK_LOCKED_WEIGHT, MemoryCell, SynapticPruningEngine
from storage.db_manager import DBManager

# ---------------------------------------------------------------------------
# 超参（可被 config.py / 环境变量覆盖）
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA = settings.lambda_
DEFAULT_THETA = settings.theta_prune
DEFAULT_TAU = settings.tau


# ---------------------------------------------------------------------------
# API Key 鉴权
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str | None = Header(None)):
    """FastAPI 依赖：校验 X-API-Key 请求头。未配置 API_KEY 时放行。"""
    if not settings.api_key:
        return True
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# ---------------------------------------------------------------------------
# 应用状态
# ---------------------------------------------------------------------------


class AppState:
    """单例式应用状态：DB 管理器 + 审计器 + 嵌入器 + 写入队列。"""

    def __init__(self, db_path: str, vector_path: str, prefer_chroma: bool = False,
                 seed: bool = True) -> None:
        self.db_path = db_path
        self.vector_path = vector_path
        self.prefer_chroma = prefer_chroma
        self.seed = seed
        self.db: DBManager | None = None
        # 根据配置选择 OpenAI 或 Mock 后端
        if settings.use_openai:
            self.auditor = OpenAILLMAuditor()
            self.embedder = OpenAIEmbedder()
        else:
            self.auditor = MockLLMAuditor()
            self.embedder = MockEmbedder()
        self.write_queue: asyncio.Queue[tuple[str, str, str, str, dict | None]] | None = None
        # Queue: (request_id, user_id, agent_id, content, task_tags)
        self._worker_task: asyncio.Task | None = None
        self._stop = False
        # 剪枝计数（监控指标用）
        self.pruned_total = 0
        self.pruned_today = 0
        self._prune_date = datetime.now(tz=timezone.utc).date()

    def startup(self) -> None:
        self.db = DBManager(
            db_path=self.db_path, vector_path=self.vector_path, prefer_chroma=self.prefer_chroma
        )
        self.write_queue = asyncio.Queue()
        if self.seed:
            self.seed_if_empty()

    def record_pruned(self, n: int) -> None:
        """记录一次剪枝抹除的数量（按 UTC 日期归零 pruned_today）。"""
        today = datetime.now(tz=timezone.utc).date()
        if self._prune_date != today:
            self._prune_date = today
            self.pruned_today = 0
        self.pruned_today += n
        self.pruned_total += n

    def seed_if_empty(self) -> None:
        """库为空时灌入演示数据，让看板首次打开就有内容。"""
        if self.db is None or self.db.count_sqlite() > 0:
            return
        now = datetime.now(tz=timezone.utc)
        # (content, is_risk, intensity, access_count, days_ago_last_access, tags, entities)
        samples = [
            ("我对青霉素过敏，开药务必避开青霉素类", True, 10.0, 1, 0, {"type": "medical"},
             [{"name": "青霉素", "type": "tech", "role": "tool"}]),
            ("服务器 root 密码是 Alpha-Bug-2024，务必保密", True, 10.0, 1, 5, {"type": "secret"},
             [{"name": "Alpha-Bug-2024", "type": "version", "role": "reference"}]),
            ("项目 Alpha 的 Bug 修复方案采用重试队列加幂等键", False, 7.0, 4, 1, {"type": "tech", "project": "Alpha"},
             [{"name": "Alpha", "type": "project", "role": "subject"}]),
            ("项目 Alpha 的部署架构用 k8s 加双活", False, 7.0, 2, 15, {"type": "tech", "project": "Alpha"},
             [{"name": "Alpha", "type": "project", "role": "subject"}, {"name": "k8s", "type": "tech", "role": "tool"}]),
            ("项目 Alpha 的监控用 prometheus 加告警", False, 7.0, 6, 30, {"type": "tech", "project": "Alpha"},
             [{"name": "Alpha", "type": "project", "role": "subject"}, {"name": "prometheus", "type": "tech", "role": "tool"}]),
            ("今天中午吃了酸菜鱼", False, 2.0, 1, 2, {"type": "casual"}, []),
            ("昨晚看了一部电影，挺无聊的", False, 2.0, 1, 18, {"type": "casual"}, []),
            ("周末想去爬山", False, 2.0, 1, 35, {"type": "casual"}, []),
        ]
        for content, is_risk, intensity, access_count, days_ago, tags, entities in samples:
            last = now - timedelta(days=days_ago)
            vec = self.embedder._compute(content)
            cell = MemoryCell(
                content=content,
                user_id="demo",
                agent_id="biomem-api",
                is_risk=is_risk,
                base_intensity=intensity,
                access_count=access_count,
                created_at=last,
                last_accessed_at=last,
                task_tags=tags,
                entities=entities,
                id=str(uuid.uuid4()),
            )
            cell.current_weight = cell.compute_weight(now, DEFAULT_LAMBDA)
            self.db.save_memory(cell, vec)

    def start_worker(self) -> None:
        self._stop = False
        self._worker_task = asyncio.create_task(self._ingestion_worker())

    async def shutdown(self) -> None:
        self._stop = True
        if self.write_queue is not None:
            await self.write_queue.put(("__stop__", "", "", "", None))
        if self._worker_task is not None:
            await self._worker_task
        if self.db is not None:
            self.db.close()

    async def _ingestion_worker(self) -> None:
        """后台消费者：从队列取 (req_id, user_id, agent_id, content, user_tags)，跑审计 → save_memory。"""
        assert self.db is not None and self.write_queue is not None
        while not self._stop:
            req_id, user_id, agent_id, content, user_tags = await self.write_queue.get()
            if req_id == "__stop__":
                break
            try:
                audit = await self.auditor.audit(content)
                # 合并标签：调用方显式传入的 task_tags 优先于自动提取
                final_tags = {**audit["task_tags"], **(user_tags or {})}
                entities = audit.get("entities", [])
                vector = await self.embedder.embed(content)
                now = datetime.now(tz=timezone.utc)
                cell = MemoryCell(
                    content=content,
                    user_id=user_id,
                    agent_id=agent_id or "biomem-api",
                    is_risk=audit["is_risk"],
                    base_intensity=audit["base_intensity"],
                    access_count=1,
                    created_at=now,
                    last_accessed_at=now,
                    task_tags=final_tags,
                    entities=entities,
                    id=str(uuid.uuid4()),
                )
                cell.current_weight = cell.compute_weight(now, DEFAULT_LAMBDA)
                self.db.save_memory(cell, vector)
            except Exception as exc:  # noqa: BLE001
                # 后台任务不可把异常抛回 HTTP 调用方；记录到 stderr
                import sys
                print(f"[ingestion_worker] req={req_id} 失败: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    user_id: str
    content: str
    agent_id: str | None = None  # 来源 agent，不传则默认 biomem-api
    task_tags: dict | None = None


class RetrieveRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    query_tags: dict | None = None  # F 轴过滤：指定则通路A仅返回标签匹配的非风险记忆
    query_entities: list[dict] | None = None  # 实体检索：通路B对命中实体做 score boost
    agent_id: str | None = None  # 指定则只检索该 agent 的记忆（不传则不过滤）


class SynthesizeRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    query_tags: dict | None = None
    query_entities: list[dict] | None = None
    agent_id: str | None = None


class PruneRequest(BaseModel):
    user_id: str | None = None  # None = 全库；指定则仅该用户
    lambda_: float = DEFAULT_LAMBDA
    theta_prune: float = DEFAULT_THETA
    simulate_days: float = 0.0  # 模拟额外流逝的天数（加速衰减）


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def create_app(
    db_path: str | None = None,
    vector_path: str | None = None,
    prefer_chroma: bool | None = None,
    seed: bool | None = None,
) -> FastAPI:
    # 优先用显式参数，否则从 config 环境变量读取
    db_path = db_path or settings.db_path
    vector_path = vector_path or settings.vector_path
    if prefer_chroma is None:
        prefer_chroma = settings.prefer_chroma
    if seed is None:
        seed = settings.seed

    state = AppState(db_path, vector_path, prefer_chroma, seed=seed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.startup()
        state.start_worker()
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(title="4D-BioMem API", version="1.2.0", lifespan=lifespan)
    app.state.state = state
    _register_routes(app, state)
    # 前端看板静态资源挂载在 /dashboard（访问 /dashboard 即打开 index.html）
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
    return app


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI, state: AppState) -> None:

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "auth": bool(settings.api_key),
            "llm_backend": "openai" if settings.use_openai else "mock",
            "db_backend": state.db.vector_backend if state.db else None,
            "sqlite_count": state.db.count_sqlite() if state.db else 0,
            "vector_count": state.db.count_vectors() if state.db else 0,
        }

    @app.post("/v1/memory/add", dependencies=[Depends(verify_api_key)])
    async def add_memory(req: AddRequest) -> dict:
        """异步录入：立即返回 queued，后台审计 + 写盘。"""
        if state.db is None or state.write_queue is None:
            raise HTTPException(503, "service not ready")
        request_id = str(uuid.uuid4())
        await state.write_queue.put((request_id, req.user_id, req.agent_id or "biomem-api", req.content, req.task_tags))
        return {"status": "queued", "message": "Memory ingestion started.", "request_id": request_id}

    @app.get("/v1/memory/list", dependencies=[Depends(verify_api_key)])
    async def list_memory(user_id: str) -> dict:
        """列出某用户全部记忆（调试用）。"""
        if state.db is None:
            raise HTTPException(503, "service not ready")
        cells = state.db.load_cells_by_user(user_id)
        now = datetime.now(tz=timezone.utc)
        items = []
        for c in cells:
            w = c.compute_weight(now, DEFAULT_LAMBDA)
            items.append({
                "id": c.id,
                "content": c.content,
                "agent_id": c.agent_id,
                "is_risk": c.is_risk,
                "base_intensity": c.base_intensity,
                "access_count": c.access_count,
                "task_tags": c.task_tags,
                "entities": c.entities,
                "last_accessed_at": c.last_accessed_at.isoformat(),
                "current_weight": "INF" if w == RISK_LOCKED_WEIGHT else round(w, 4),
            })
        return {"user_id": user_id, "count": len(items), "items": items}

    @app.post("/v1/memory/retrieve", dependencies=[Depends(verify_api_key)])
    async def retrieve_memory(req: RetrieveRequest) -> dict:
        """双通路唤醒：A 硬过滤 + B 软匹配 → 去重融合 → access_count += 1。"""
        if state.db is None:
            raise HTTPException(503, "service not ready")
        hits, pathway_a, pathway_b = await _retrieve_hits(
            state, req.user_id, req.query, req.top_k,
            query_tags=req.query_tags, query_entities=req.query_entities,
            agent_id=req.agent_id,
        )
        return {
            "user_id": req.user_id,
            "query": req.query,
            "hits": hits,
            "pathways": {"A": len(pathway_a), "B": len(pathway_b)},
        }

    @app.post("/v1/memory/synthesize", dependencies=[Depends(verify_api_key)])
    async def synthesize_memory(req: SynthesizeRequest) -> dict:
        """跨记忆合成：检索 Top-K → LLM 综合回答（Mock 模式返回拼接摘要）。"""
        if state.db is None:
            raise HTTPException(503, "service not ready")
        hits, _, _ = await _retrieve_hits(
            state, req.user_id, req.question, req.top_k,
            query_tags=req.query_tags, query_entities=req.query_entities,
            agent_id=req.agent_id,
        )
        contexts = [h["content"] for h in hits]

        if settings.use_openai and settings.openai_api_key:
            # OpenAI 路径：LLM 综合回答
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            ctx_block = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            prompt = (
                f"Based on the following memory fragments, answer the user's question concisely.\n\n"
                f"Question: {req.question}\n\n"
                f"Memory fragments:\n{ctx_block}\n\n"
                f"Answer:"
            ) if contexts else f"No relevant memories found for: {req.question}"
            try:
                resp = await client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                answer = resp.choices[0].message.content or "No answer generated."
                mode = "openai"
            except Exception as exc:
                answer = f"Synthesis failed: {exc}"
                mode = "error"
        else:
            # Mock 路径：拼接摘要
            if contexts:
                answer = f"Found {len(contexts)} relevant memories:\n" + "\n".join(
                    f"- {c}" for c in contexts
                )
            else:
                answer = "No relevant memories found."
            mode = "mock"

        return {
            "question": req.question,
            "answer": answer,
            "hits": hits,
            "synthesis_mode": mode,
        }

    @app.post("/v1/memory/prune", dependencies=[Depends(verify_api_key)])
    async def prune_memory(req: PruneRequest) -> dict:
        """主动新陈代谢：扫描非风险记忆，物理抹除 W_i(t) < θ_prune 的死亡记忆。

        simulate_days > 0 时把评估时钟往前推，模拟"加速衰减"。
        """
        if state.db is None:
            raise HTTPException(503, "service not ready")
        db = state.db
        from datetime import timedelta as _td
        evaluation_time = datetime.now(tz=timezone.utc) + _td(days=req.simulate_days)
        cells = db.load_all_active_cells()
        # 引擎时钟 = 评估时刻；load_cell 用该时钟重算权重（delta 含 simulate_days 衰减）
        engine = SynapticPruningEngine(
            lambda_=req.lambda_, theta_prune=req.theta_prune, start_time=evaluation_time
        )
        for c in cells:
            engine.load_cell(c)
        pruned = engine.run_pruning()
        pruned_items = []
        for p in pruned:
            db.delete_memory(p.id)
            pruned_items.append({
                "id": p.id,
                "content": p.content,
                "final_weight": "INF" if p.current_weight == RISK_LOCKED_WEIGHT else round(p.current_weight, 4),
                "access_count": p.access_count,
            })
        state.record_pruned(len(pruned))
        return {
            "scanned": len(cells),
            "pruned": len(pruned),
            "survivors": len(cells) - len(pruned),
            "pruned_items": pruned_items,
            "simulated_extra_days": req.simulate_days,
        }

    # ---- 监控接口（看板用）-----------------------------------------------

    @app.get("/v1/monitor/cells", dependencies=[Depends(verify_api_key)])
    async def monitor_cells(user_id: str | None = None) -> dict:
        """返回全部记忆细胞（含实时权重、频次、距上次唤醒秒数、风险状态）。"""
        if state.db is None:
            raise HTTPException(503, "service not ready")
        db = state.db
        now = datetime.now(tz=timezone.utc)
        cells = db.load_cells_by_user(user_id) if user_id else db.load_all_active_cells()
        items = []
        for c in cells:
            w = c.compute_weight(now, DEFAULT_LAMBDA)
            is_inf = w == RISK_LOCKED_WEIGHT
            items.append({
                "id": c.id,
                "content": c.content,
                "agent_id": c.agent_id,
                "is_risk": c.is_risk,
                "weight": "INF" if is_inf else round(w, 4),
                "weight_sort": float("inf") if is_inf else w,
                "access_count": c.access_count,
                "base_intensity": c.base_intensity,
                "task_tags": c.task_tags,
                "entities": c.entities,
                "seconds_since_last_access": max(0.0, (now - c.last_accessed_at).total_seconds()),
                "last_accessed_at": c.last_accessed_at.isoformat(),
                "_sort": (1 if c.is_risk else 0, float("inf") if is_inf else w),
            })
        items.sort(key=lambda x: x["_sort"], reverse=True)
        for x in items:
            x.pop("_sort", None)
        return {
            "count": len(items),
            "items": items,
            "theta_prune": DEFAULT_THETA,
            "lambda": DEFAULT_LAMBDA,
        }

    @app.post("/v1/monitor/system_status", dependencies=[Depends(verify_api_key)])
    async def monitor_system_status() -> dict:
        """返回整体系统指标：有效记忆数、风险锁定率、今日剪枝数等。"""
        if state.db is None:
            raise HTTPException(503, "service not ready")
        db = state.db
        cells = db.load_all_active_cells()
        risk_count = sum(1 for c in cells if c.is_risk)
        active = len(cells)
        return {
            "active_count": active,
            "risk_count": risk_count,
            "risk_lock_rate": round(risk_count / active, 4) if active else 0.0,
            "pruned_today": state.pruned_today,
            "pruned_total": state.pruned_total,
            "theta_prune": DEFAULT_THETA,
            "lambda": DEFAULT_LAMBDA,
            "vector_backend": db.vector_backend,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


async def _retrieve_hits(
    state: AppState, user_id: str, query: str, top_k: int,
    query_tags: dict | None = None,
    query_entities: list[dict] | None = None,
    agent_id: str | None = None,
) -> tuple[list[dict], dict[str, MemoryCell], dict[str, float]]:
    """共享检索逻辑：双通路唤醒 → 去重融合 → 突触强化 → 返回命中列表。

    Returns
    -------
    (fused_hits, pathway_a, pathway_b)
    """
    db = state.db
    now = datetime.now(tz=timezone.utc)
    cells = db.load_cells_by_user(user_id)
    if not cells:
        return [], {}, {}

    # agent_id 预过滤
    if agent_id:
        cells = [c for c in cells if c.agent_id == agent_id]

    query_vec = await state.embedder.embed(query)
    qv = np.asarray(query_vec, dtype=np.float32)

    # ---- 通路 A：潜意识反射（硬过滤）----------------------------------
    recent = sorted(cells, key=lambda c: c.last_accessed_at, reverse=True)[:5]
    risk_cells = [c for c in cells if c.is_risk]
    pathway_a: dict[str, MemoryCell] = {}
    for c in recent:
        if query_tags and not c.is_risk:
            if not _tags_overlap(c.task_tags, query_tags):
                continue
        pathway_a[c.id] = c
    for c in risk_cells:
        pathway_a[c.id] = c
    a_scores = {cid: c.compute_weight(now, DEFAULT_LAMBDA) for cid, c in pathway_a.items()}

    # ---- 通路 B：显意识回忆（软匹配 + 实体 boost）-----------------------
    pathway_b: dict[str, float] = {}
    for c in cells:
        if c.is_risk:
            continue
        vec = db.get_vector(c.id)
        if vec is None:
            continue
        sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
        if sim < 0.3:
            continue
        w = c.compute_weight(now, DEFAULT_LAMBDA)
        # 实体重叠 boost
        entity_boost = 1.0
        if query_entities and c.entities:
            n_overlap = 0
            for qe in query_entities:
                for ce in c.entities:
                    if (qe.get("name", "").lower() == ce.get("name", "").lower()
                            and qe.get("type") == ce.get("type")):
                        n_overlap += 1
                        break
            if n_overlap > 0:
                entity_boost = min(1.25 ** n_overlap, 2.0)  # cap at 2x
        score = sim * math.log(1.0 + w) * entity_boost
        pathway_b[c.id] = score

    # ---- 去重融合 -----------------------------------------------------
    all_ids = set(pathway_a) | set(pathway_b)
    fused: list[dict] = []
    for cid in all_ids:
        cell = next(c for c in cells if c.id == cid)
        in_a = cid in pathway_a
        in_b = cid in pathway_b
        a_score = a_scores.get(cid, 0.0) if in_a else 0.0
        b_score = pathway_b.get(cid, 0.0) if in_b else 0.0
        if cell.is_risk:
            sort_key = (True, True, float("inf"))
            display = "INF"
        elif in_b:
            sort_key = (False, True, b_score)
            display = round(b_score, 4)
        else:
            sort_key = (False, False, a_score)
            display = round(a_score, 4)
        pathways = []
        if in_a:
            pathways.append("reflex_risk" if cell.is_risk else "reflex_recent")
        if in_b:
            pathways.append("soft")
        fused.append({
            "id": cid,
            "content": cell.content,
            "agent_id": cell.agent_id,
            "is_risk": cell.is_risk,
            "access_count": cell.access_count,
            "task_tags": cell.task_tags,
            "entities": cell.entities,
            "score": display,
            "pathways": pathways,
            "_sort": sort_key,
        })
    fused.sort(key=lambda x: x["_sort"], reverse=True)
    fused = fused[:top_k]
    for x in fused:
        x.pop("_sort", None)

    # ---- 突触强化：access_count += 1 + 时间戳更新（持久化）-------------
    for item in fused:
        cell = next(c for c in cells if c.id == item["id"])
        cell.access_count += 1
        cell.last_accessed_at = now
        cell.current_weight = cell.compute_weight(now, DEFAULT_LAMBDA)
        db.update_cell(cell)

    return fused, pathway_a, pathway_b


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# 默认应用实例（供 uvicorn 直接加载）
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8765, reload=False)
