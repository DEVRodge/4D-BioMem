"""4D-BioMem 里程碑 2：双轨制存储层。

缝合 SQLite（元数据，T/F/R 轴 + 突触权重）与本地向量库（V 轴），
提供 Dual-Write / Dual-Delete 联合接口，供突触剪枝守护进程调用。

设计要点：
  - 元数据（含 task_tags JSON 串、时间戳、权重）落 SQLite，单文件 biomem.db。
  - 高维向量落本地向量库：优先 chromadb 持久化集合；环境无 chromadb 时
    自动降级为 NumpyVectorStore（pickle 落盘的单文件字典）。
  - cell.id 即向量库主键，vector_id 字段对齐为 cell.id，保证 Dual-Delete 干净。
  - save_memory 做跨存储事务补偿：向量库写入失败时回滚 SQLite 行。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol

from core.memory_cell import RISK_LOCKED_WEIGHT, MemoryCell

# ---------------------------------------------------------------------------
# 向量存储抽象
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """本地向量库的统一接口（V 轴载体）。"""

    def add(self, cell_id: str, vector, metadata: dict | None = None) -> None: ...
    def delete(self, cell_id: str) -> None: ...
    def get(self, cell_id: str):
        """返回向量，不存在则返回 None。"""
    ...
    def has(self, cell_id: str) -> bool: ...
    def count(self) -> int: ...
    def close(self) -> None: ...


def _to_list(vector) -> list[float]:
    """chromadb 接受 list[float]；numpy / tuple 一律归一化为 list。"""
    if vector is None:
        return []
    if hasattr(vector, "tolist"):  # numpy / torch
        return [float(x) for x in vector.tolist()]
    return [float(x) for x in vector]


def _clean_required_text(value: str | None, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


# ---------------------------------------------------------------------------
# ChromaDB 实现（首选）
# ---------------------------------------------------------------------------


class ChromaVectorStore:
    """基于 chromadb.PersistentClient 的本地持久化向量库。"""

    def __init__(self, path: str, collection_name: str = "memory_vectors") -> None:
        import chromadb
        from chromadb.config import Settings

        os.makedirs(path, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(collection_name)

    def add(self, cell_id: str, vector, metadata: dict | None = None) -> None:
        meta = {k: v for k, v in (metadata or {}).items() if isinstance(v, (str, int, float, bool))}
        # upsert：同一 id 重复写入时覆盖
        self._col.upsert(ids=[cell_id], embeddings=[_to_list(vector)], metadatas=[meta or None])

    def delete(self, cell_id: str) -> None:
        # chromadb delete 对不存在的 id 不报错
        self._col.delete(ids=[cell_id])

    def get(self, cell_id: str):
        res = self._col.get(ids=[cell_id], include=["embeddings"])
        embs = res.get("embeddings")
        if embs is None or len(embs) == 0:
            return None
        return embs[0]

    def has(self, cell_id: str) -> bool:
        res = self._col.get(ids=[cell_id])
        ids = res.get("ids")
        return ids is not None and len(ids) > 0

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        # PersistentClient 无显式 close；触发落盘由 chromadb 自行管理
        pass


# ---------------------------------------------------------------------------
# Numpy 降级实现
# ---------------------------------------------------------------------------


class NumpyVectorStore:
    """纯 numpy + pickle 的极简本地向量库（chromadb 不可用时的降级方案）。

    单文件持久化：{cell_id: {"vector": list[float], "metadata": dict}}。
    """

    def __init__(self, path: str) -> None:
        import pickle

        self.path = path
        self._pickle = pickle
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._data = pickle.load(f)
        else:
            self._data: dict[str, dict] = {}

    def _flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            self._pickle.dump(self._data, f)

    def add(self, cell_id: str, vector, metadata: dict | None = None) -> None:
        self._data[cell_id] = {"vector": _to_list(vector), "metadata": metadata or {}}
        self._flush()

    def delete(self, cell_id: str) -> None:
        self._data.pop(cell_id, None)
        self._flush()

    def get(self, cell_id: str):
        entry = self._data.get(cell_id)
        return entry["vector"] if entry else None

    def has(self, cell_id: str) -> bool:
        return cell_id in self._data

    def count(self) -> int:
        return len(self._data)

    def close(self) -> None:
        self._flush()


def _make_vector_store(path: str, prefer_chroma: bool) -> VectorStore:
    if prefer_chroma:
        try:
            return ChromaVectorStore(path)
        except Exception as exc:  # noqa: BLE001 — 降级是公开行为
            print(f"[db_manager] chromadb 不可用，降级 NumpyVectorStore: {exc}")
    return NumpyVectorStore(path + ".pkl")


# ---------------------------------------------------------------------------
# SQLite 元数据
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_cells (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL,
    content          TEXT NOT NULL,
    vector_id        TEXT NOT NULL,
    task_tags        TEXT,                 -- JSON 字符串
    entities         TEXT DEFAULT '[]',    -- JSON 实体列表
    is_risk          INTEGER NOT NULL,     -- 0 / 1
    base_intensity   REAL NOT NULL,        -- I_i ∈ [1, 10]
    access_count     INTEGER NOT NULL,     -- C_i
    created_at       TEXT NOT NULL,        -- ISO 8601
    last_accessed_at TEXT NOT NULL,        -- ISO 8601
    current_weight   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL,
    content          TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    task_tags        TEXT,
    created_at       TEXT NOT NULL,
    occurred_at      TEXT NOT NULL,
    archived         INTEGER NOT NULL DEFAULT 0,
    archive_cell_id  TEXT,
    source           TEXT,
    source_id        TEXT,
    connector_id     TEXT
);

CREATE TABLE IF NOT EXISTS memory_connectors (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    connector_type   TEXT NOT NULL,
    config           TEXT,
    enabled          INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS connector_runs (
    id               TEXT PRIMARY KEY,
    connector_id     TEXT NOT NULL,
    status           TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL,
    imported_count   INTEGER NOT NULL DEFAULT 0,
    skipped_count    INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    details          TEXT
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO memory_cells
    (id, user_id, agent_id, content, vector_id, task_tags, entities,
     is_risk, base_intensity, access_count,
     created_at, last_accessed_at, current_weight)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_EVENT_SQL = """
INSERT INTO memory_events
    (id, user_id, agent_id, content, event_type, task_tags,
     created_at, occurred_at, archived, archive_cell_id, source, source_id, connector_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# 双轨管理器
# ---------------------------------------------------------------------------


class DBManager:
    """SQLite + 向量库的双轨制存储管理器。

    Parameters
    ----------
    db_path : str
        SQLite 数据库文件路径，如 "biomem.db"。
    vector_path : str
        向量库路径。chroma 模式下为目录；numpy 降级模式下自动追加 .pkl。
    prefer_chroma : bool
        默认 False 用轻量 NumpyVectorStore（零额外依赖，单 pickle 文件）；
        True 时启用 chromadb（需已安装，作为可选重后端）。
    """

    def __init__(
        self,
        db_path: str = "biomem.db",
        vector_path: str = "vector_store",
        prefer_chroma: bool = False,
    ) -> None:
        self.db_path = db_path
        self.vector_path = vector_path
        # check_same_thread=False：FastAPI 后台任务与主线程共享同一连接；
        # 用 RLock 串行化所有写读，避免 SQLite 并发异常
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # 默认走轻量 NumpyVectorStore（零额外依赖）；显式 prefer_chroma=True 才启用 chromadb
        self.vector_store: VectorStore = _make_vector_store(vector_path, prefer_chroma)
        self._migrate()
        self._closed = False

    def _migrate(self) -> None:
        """为新列做 ALTER TABLE（SQLite 的 CREATE TABLE IF NOT EXISTS 不会追加列）。"""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_cells)")}
        if "entities" not in columns:
            self._conn.execute("ALTER TABLE memory_cells ADD COLUMN entities TEXT DEFAULT '[]'")
            self._conn.commit()

        # v1.6: 每日片段摄取表。对已存在库做幂等创建。
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_events (
            id               TEXT PRIMARY KEY,
            user_id          TEXT NOT NULL,
            agent_id         TEXT NOT NULL,
            content          TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            task_tags        TEXT,
            created_at       TEXT NOT NULL,
            occurred_at      TEXT NOT NULL,
            archived         INTEGER NOT NULL DEFAULT 0,
            archive_cell_id  TEXT,
            source           TEXT,
            source_id        TEXT,
            connector_id     TEXT
        );

        CREATE TABLE IF NOT EXISTS memory_connectors (
            id               TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            connector_type   TEXT NOT NULL,
            config           TEXT,
            enabled          INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS connector_runs (
            id               TEXT PRIMARY KEY,
            connector_id     TEXT NOT NULL,
            status           TEXT NOT NULL,
            started_at       TEXT NOT NULL,
            finished_at      TEXT NOT NULL,
            imported_count   INTEGER NOT NULL DEFAULT 0,
            skipped_count    INTEGER NOT NULL DEFAULT 0,
            error            TEXT,
            details          TEXT
        );
        """)
        event_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_events)")}
        for column in ("source", "source_id", "connector_id"):
            if column not in event_columns:
                self._conn.execute(f"ALTER TABLE memory_events ADD COLUMN {column} TEXT")
        self._deduplicate_connector_event_identities()
        self._conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_events_connector_source
        ON memory_events (connector_id, source, source_id)
        WHERE connector_id IS NOT NULL AND source IS NOT NULL AND source_id IS NOT NULL
        """)
        self._conn.commit()

    def _deduplicate_connector_event_identities(self) -> None:
        """让早期重复连接器身份在建唯一索引前变为可索引状态，不删除事件内容。"""
        duplicate_groups = self._conn.execute(
            """
            SELECT connector_id, source, source_id, COUNT(*) AS n
            FROM memory_events
            WHERE connector_id IS NOT NULL AND source IS NOT NULL AND source_id IS NOT NULL
            GROUP BY connector_id, source, source_id
            HAVING n > 1
            """
        ).fetchall()
        for group in duplicate_groups:
            rows = self._conn.execute(
                """SELECT id, source_id
                   FROM memory_events
                   WHERE connector_id = ? AND source = ? AND source_id = ?
                   ORDER BY created_at ASC, id ASC""",
                (group["connector_id"], group["source"], group["source_id"]),
            ).fetchall()
            for row in rows[1:]:
                self._conn.execute(
                    "UPDATE memory_events SET source_id = ? WHERE id = ?",
                    (f"{row['source_id']}#duplicate:{row['id']}", row["id"]),
                )

    # ---- Dual-Write --------------------------------------------------------

    def save_memory(self, cell: MemoryCell, vector) -> None:
        """联合写入：元数据落 SQLite，向量落向量库。

        跨存储事务补偿：若向量库写入失败，回滚刚写入的 SQLite 行，保证双轨一致。
        """
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        # vector_id 对齐为 cell.id，保证 Dual-Delete 主键统一
        cell.vector_id = cell.id
        tags_json = json.dumps(cell.task_tags, ensure_ascii=False)
        entities_json = json.dumps(cell.entities, ensure_ascii=False)
        row = (
            cell.id,
            cell.user_id,
            cell.agent_id,
            cell.content,
            cell.vector_id,
            tags_json,
            entities_json,
            int(cell.is_risk),
            float(cell.base_intensity),
            int(cell.access_count),
            cell.created_at.isoformat(),
            cell.last_accessed_at.isoformat(),
            float(cell.current_weight),
        )
        with self._lock:
            self._conn.execute(_INSERT_SQL, row)
            self._conn.commit()
        try:
            self.vector_store.add(
                cell.id,
                vector,
                metadata={"content": cell.content, "is_risk": bool(cell.is_risk)},
            )
        except Exception:
            # 回滚 SQLite，避免出现"有元数据无向量"的孤儿行
            with self._lock:
                self._conn.execute("DELETE FROM memory_cells WHERE id = ?", (cell.id,))
                self._conn.commit()
            raise

    # ---- Dual-Read ---------------------------------------------------------

    def load_all_active_cells(self) -> list[MemoryCell]:
        """从 SQLite 读出全部记忆，反序列化为 MemoryCell 列表。

        剪枝守护进程启动时调用：磁盘上现存的都是"未被判死刑"的有效记忆。
        """
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memory_cells").fetchall()
        return [self._row_to_cell(r) for r in rows]

    def _row_to_cell(self, r: sqlite3.Row) -> MemoryCell:
        """sqlite3.Row -> MemoryCell（共享反序列化逻辑）。"""
        tags = json.loads(r["task_tags"]) if r["task_tags"] else {}
        entities = json.loads(r["entities"]) if "entities" in r.keys() and r["entities"] else []
        return MemoryCell(
            content=r["content"],
            user_id=r["user_id"],
            agent_id=r["agent_id"],
            vector_id=r["vector_id"],
            task_tags=tags,
            entities=entities,
            is_risk=bool(r["is_risk"]),
            base_intensity=float(r["base_intensity"]),
            access_count=int(r["access_count"]),
            created_at=datetime.fromisoformat(r["created_at"]),
            last_accessed_at=datetime.fromisoformat(r["last_accessed_at"]),
            current_weight=float(r["current_weight"]),
            id=r["id"],
        )

    def load_cells_by_user(self, user_id: str) -> list[MemoryCell]:
        """载入某用户全部记忆（按 last_accessed_at 降序）。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_cells WHERE user_id = ? ORDER BY last_accessed_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_cell(r) for r in rows]

    def get_cell(self, cell_id: str) -> MemoryCell | None:
        """读取单条记忆。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_cells WHERE id = ?", (cell_id,)
            ).fetchone()
        return self._row_to_cell(row) if row else None

    def update_cell(self, cell: MemoryCell) -> None:
        """把内存中被强化的 cell（access_count / last_accessed / weight）回写 SQLite。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            self._conn.execute(
                """UPDATE memory_cells
                   SET access_count = ?, last_accessed_at = ?, current_weight = ?,
                       base_intensity = ?, is_risk = ?, task_tags = ?, entities = ?
                   WHERE id = ?""",
                (
                    int(cell.access_count),
                    cell.last_accessed_at.isoformat(),
                    float(cell.current_weight),
                    float(cell.base_intensity),
                    int(cell.is_risk),
                    json.dumps(cell.task_tags, ensure_ascii=False),
                    json.dumps(cell.entities, ensure_ascii=False),
                    cell.id,
                ),
            )
            self._conn.commit()

    # ---- Daily Event Fragments -------------------------------------------

    def save_event(
        self,
        *,
        user_id: str,
        agent_id: str = "biomem-api",
        content: str,
        event_type: str = "generic",
        task_tags: dict | None = None,
        occurred_at: datetime | None = None,
        event_id: str | None = None,
        source: str | None = None,
        source_id: str | None = None,
        connector_id: str | None = None,
    ) -> dict[str, Any]:
        """保存一条每日片段事件。

        事件只进入 SQLite，不写向量库；归档后才生成长期 MemoryCell。
        """
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        if any(value is not None for value in (connector_id, source, source_id)):
            connector_id = _clean_required_text(connector_id, "connector_id")
            source = _clean_required_text(source, "source")
            source_id = _clean_required_text(source_id, "source_id")
        now = datetime.now(tz=timezone.utc)
        occurred = occurred_at or now
        row = (
            event_id or str(uuid.uuid4()),
            user_id,
            agent_id,
            content,
            event_type or "generic",
            json.dumps(task_tags or {}, ensure_ascii=False),
            now.isoformat(),
            occurred.isoformat(),
            0,
            None,
            source,
            source_id,
            connector_id,
        )
        with self._lock:
            if connector_id and source and source_id:
                existing = self._conn.execute(
                    """SELECT * FROM memory_events
                       WHERE connector_id = ? AND source = ? AND source_id = ?""",
                    (connector_id, source, source_id),
                ).fetchone()
                if existing is not None:
                    event = self._event_row_to_dict(existing)
                    event["duplicate"] = True
                    return event
            try:
                self._conn.execute(_INSERT_EVENT_SQL, row)
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                if connector_id and source and source_id:
                    existing = self._conn.execute(
                        """SELECT * FROM memory_events
                           WHERE connector_id = ? AND source = ? AND source_id = ?""",
                        (connector_id, source, source_id),
                    ).fetchone()
                    if existing is not None:
                        event = self._event_row_to_dict(existing)
                        event["duplicate"] = True
                        return event
                raise
            inserted = self._conn.execute("SELECT * FROM memory_events WHERE id = ?", (row[0],)).fetchone()
        event = self._event_row_to_dict(inserted)
        event["duplicate"] = False
        return event

    def _get_event_row(self, event_id: str) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM memory_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"memory event not found: {event_id}")
        return row

    def _get_event_by_source_identity(
        self, connector_id: str, source: str, source_id: str
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM memory_events
                   WHERE connector_id = ? AND source = ? AND source_id = ?""",
                (connector_id, source, source_id),
            ).fetchone()

    def _event_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "agent_id": row["agent_id"],
            "content": row["content"],
            "event_type": row["event_type"],
            "task_tags": json.loads(row["task_tags"]) if row["task_tags"] else {},
            "created_at": row["created_at"],
            "occurred_at": row["occurred_at"],
            "archived": bool(row["archived"]),
            "archive_cell_id": row["archive_cell_id"],
            "source": row["source"] if "source" in row.keys() else None,
            "source_id": row["source_id"] if "source_id" in row.keys() else None,
            "connector_id": row["connector_id"] if "connector_id" in row.keys() else None,
        }

    def list_events(
        self,
        *,
        user_id: str | None = None,
        date: str | None = None,
        archived: bool | None = None,
        agent_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """列出每日片段事件，按发生时间升序。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        clauses = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if date:
            clauses.append("substr(occurred_at, 1, 10) = ?")
            params.append(date)
        if archived is not None:
            clauses.append("archived = ?")
            params.append(1 if archived else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory_events {where} ORDER BY occurred_at ASC, created_at ASC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._event_row_to_dict(row) for row in rows]

    def list_unarchived_event_groups(
        self,
        *,
        today: str | None = None,
        include_today: bool = False,
    ) -> list[dict[str, Any]]:
        """列出待归档事件分组，默认只返回今天之前的未归档片段。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        clauses = ["archived = 0"]
        params: list[Any] = []
        if today and not include_today:
            clauses.append("substr(occurred_at, 1, 10) < ?")
            params.append(today)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT user_id, agent_id, substr(occurred_at, 1, 10) AS date, COUNT(*) AS event_count
                FROM memory_events
                WHERE {where}
                GROUP BY user_id, agent_id, date
                ORDER BY date ASC, user_id ASC, agent_id ASC
                """,
                tuple(params),
            ).fetchall()
        return [
            {
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "date": row["date"],
                "event_count": int(row["event_count"]),
            }
            for row in rows
        ]

    # ---- Connectors ------------------------------------------------------

    def register_connector(
        self,
        *,
        connector_id: str | None = None,
        name: str,
        connector_type: str,
        config: dict | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """注册或更新一个连接器配置。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        cid = _clean_required_text(connector_id, "connector_id") if connector_id is not None else str(uuid.uuid4())
        name = _clean_required_text(name, "name")
        connector_type = _clean_required_text(connector_type, "connector_type")
        now = datetime.now(tz=timezone.utc).isoformat()
        config_json = json.dumps(config or {}, ensure_ascii=False)
        with self._lock:
            existing = self._conn.execute(
                "SELECT created_at FROM memory_connectors WHERE id = ?", (cid,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            self._conn.execute(
                """INSERT OR REPLACE INTO memory_connectors
                   (id, name, connector_type, config, enabled, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cid, name, connector_type, config_json, int(enabled), created_at, now),
            )
            self._conn.commit()
        return self.get_connector(cid)

    def get_connector(self, connector_id: str) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_connectors WHERE id = ?", (connector_id,)
            ).fetchone()
        return self._connector_row_to_dict(row) if row else None

    def _connector_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "connector_type": row["connector_type"],
            "config": json.loads(row["config"]) if row["config"] else {},
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_connectors(self, *, enabled: bool | None = None) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        clauses = []
        params: list[Any] = []
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(1 if enabled else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memory_connectors {where} ORDER BY created_at ASC, id ASC",
                tuple(params),
            ).fetchall()
        return [self._connector_row_to_dict(row) for row in rows]

    def record_connector_run(
        self,
        *,
        connector_id: str,
        status: str,
        imported_count: int = 0,
        skipped_count: int = 0,
        error: str | None = None,
        details: dict | None = None,
        run_id: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> dict[str, Any]:
        """记录一次连接器同步结果。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        connector_id = _clean_required_text(connector_id, "connector_id")
        started = started_at or datetime.now(tz=timezone.utc)
        finished = finished_at or datetime.now(tz=timezone.utc)
        rid = run_id or str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO connector_runs
                   (id, connector_id, status, started_at, finished_at,
                    imported_count, skipped_count, error, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rid,
                    connector_id,
                    status,
                    started.isoformat(),
                    finished.isoformat(),
                    int(imported_count),
                    int(skipped_count),
                    error,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
            self._conn.commit()
        return self._connector_run_row_to_dict(self._get_connector_run_row(rid))

    def _get_connector_run_row(self, run_id: str) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM connector_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"connector run not found: {run_id}")
        return row

    def _connector_run_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "connector_id": row["connector_id"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "imported_count": int(row["imported_count"]),
            "skipped_count": int(row["skipped_count"]),
            "error": row["error"],
            "details": json.loads(row["details"]) if row["details"] else {},
        }

    def list_connector_runs(
        self,
        *,
        connector_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        clauses = []
        params: list[Any] = []
        if connector_id:
            clauses.append("connector_id = ?")
            params.append(connector_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM connector_runs {where} ORDER BY started_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._connector_run_row_to_dict(row) for row in rows]

    def mark_events_archived(self, event_ids: list[str], archive_cell_id: str) -> None:
        """把片段标记为已归档，并关联生成的长期记忆 cell。"""
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        if not event_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memory_events SET archived = 1, archive_cell_id = ? WHERE id = ?",
                [(archive_cell_id, event_id) for event_id in event_ids],
            )
            self._conn.commit()

    def get_vector(self, cell_id: str):
        """从向量库取回原始向量（用于校验或重建）。"""
        return self.vector_store.get(cell_id)

    # ---- Dual-Delete -------------------------------------------------------

    def delete_memory(self, cell_id: str) -> bool:
        """联合物理抹除：同时从 SQLite 与向量库删除该 id。

        返回 SQLite 是否确有行被删除（向量库删除是幂等的 best-effort）。
        """
        if self._closed:
            raise RuntimeError("DBManager 已关闭")
        with self._lock:
            cur = self._conn.execute("DELETE FROM memory_cells WHERE id = ?", (cell_id,))
            self._conn.commit()
            sqlite_deleted = cur.rowcount > 0
        self.vector_store.delete(cell_id)
        return sqlite_deleted

    # ---- 校验辅助 ----------------------------------------------------------

    def sqlite_has(self, cell_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM memory_cells WHERE id = ?", (cell_id,)
            ).fetchone()
        return row is not None

    def vector_has(self, cell_id: str) -> bool:
        return self.vector_store.has(cell_id)

    def count_sqlite(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM memory_cells").fetchone()[0]

    def count_vectors(self) -> int:
        return self.vector_store.count()

    @property
    def vector_backend(self) -> str:
        return type(self.vector_store).__name__

    # ---- 生命周期 ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self.vector_store.close()
        self._conn.close()
        self._closed = True

    def __enter__(self) -> "DBManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "DBManager",
    "VectorStore",
    "ChromaVectorStore",
    "NumpyVectorStore",
]
