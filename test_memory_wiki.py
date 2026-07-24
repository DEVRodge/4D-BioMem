from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import create_app
from core.memory_cell import MemoryCell
from core.memory_wiki import build_memory_wiki


def _cell(cell_id: str, content: str, *, project: str = "4D-BioMem") -> MemoryCell:
    return MemoryCell(
        id=cell_id,
        content=content,
        user_id="hermes",
        agent_id="codex",
        task_tags={"project": project, "type": "tech"},
        entities=[{"name": project, "type": "project"}],
        base_intensity=8.0,
    )


class MemoryWikiBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_wiki_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_memory_wiki_writes_index_project_and_daily_pages(self) -> None:
        now = datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc)
        events = [
            {
                "id": "event-1",
                "user_id": "hermes",
                "agent_id": "codex",
                "content": "用户同意 v1.7 生成 Memory Wiki",
                "event_type": "decision",
                "task_tags": {"project": "4D-BioMem"},
                "created_at": "2026-07-24T08:59:00+00:00",
                "occurred_at": "2026-07-24T08:58:00+00:00",
                "archived": False,
                "archive_cell_id": None,
            }
        ]

        result = build_memory_wiki(
            cells=[
                _cell("memory-1", "[项目进展] v1.7 增加生成式 Memory Wiki"),
                _cell("memory-2", "[用户偏好] 版本说明和 tag 描述使用中文"),
            ],
            events=events,
            output_dir=self.tmp,
            now=now,
        )

        self.assertEqual(result["format"], "4d-biomem-memory-wiki")
        self.assertEqual(result["page_count"], 4)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "manifest.json")))

        with open(os.path.join(self.tmp, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        paths = [page["path"] for page in manifest["pages"]]
        self.assertIn("index.md", paths)
        self.assertIn("users/hermes/agents/codex/index.md", paths)
        self.assertIn("users/hermes/agents/codex/projects/4D-BioMem/timeline.md", paths)
        self.assertIn("users/hermes/agents/codex/daily/2026-07-24.md", paths)

        with open(
            os.path.join(self.tmp, "users/hermes/agents/codex/projects/4D-BioMem/timeline.md"),
            encoding="utf-8",
        ) as f:
            project_page = f.read()
        self.assertIn("source_memory_ids:", project_page)
        self.assertIn("memory-1", project_page)
        self.assertIn("[项目进展] v1.7 增加生成式 Memory Wiki", project_page)

        with open(
            os.path.join(self.tmp, "users/hermes/agents/codex/daily/2026-07-24.md"),
            encoding="utf-8",
        ) as f:
            daily_page = f.read()
        self.assertIn("source_event_ids:", daily_page)
        self.assertIn("event-1", daily_page)
        self.assertIn("[decision] 用户同意 v1.7 生成 Memory Wiki", daily_page)


class MemoryWikiApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_wiki_api_")
        self.app = create_app(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            wiki_path=os.path.join(self.tmp, "wiki"),
            prefer_chroma=False,
            seed=False,
        )
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_list_and_read_wiki_pages(self) -> None:
        add_response = self.client.post(
            "/v1/memory/add",
            json={
                "user_id": "hermes",
                "agent_id": "codex",
                "content": "[项目进展] v1.7 的 wiki API 已进入测试",
                "task_tags": {"project": "4D-BioMem", "type": "tech"},
            },
        )
        self.assertEqual(add_response.status_code, 200)

        import time

        for _ in range(50):
            list_response = self.client.get("/v1/memory/list", params={"user_id": "hermes"})
            if list_response.json()["count"] == 1:
                break
            time.sleep(0.05)
        else:
            self.fail("memory was not persisted")

        event_response = self.client.post(
            "/v1/memory/ingest_event",
            json={
                "user_id": "hermes",
                "agent_id": "codex",
                "content": "用户要求 Memory Wiki 可以在 web 端查看",
                "event_type": "requirement",
                "task_tags": {"project": "4D-BioMem"},
                "occurred_at": "2026-07-24T09:30:00+00:00",
            },
        )
        self.assertEqual(event_response.status_code, 200)

        build_response = self.client.post("/v1/wiki/build", json={"user_id": "hermes"})
        self.assertEqual(build_response.status_code, 200)
        self.assertEqual(build_response.json()["status"], "built")

        pages_response = self.client.get("/v1/wiki/pages")
        self.assertEqual(pages_response.status_code, 200)
        page_paths = [page["path"] for page in pages_response.json()["pages"]]
        self.assertIn("index.md", page_paths)
        self.assertIn("users/hermes/agents/codex/projects/4D-BioMem/timeline.md", page_paths)

        page_response = self.client.get(
            "/v1/wiki/page",
            params={"path": "users/hermes/agents/codex/projects/4D-BioMem/timeline.md"},
        )
        self.assertEqual(page_response.status_code, 200)
        self.assertIn("v1.7 的 wiki API 已进入测试", page_response.json()["content"])

        escape_response = self.client.get("/v1/wiki/page", params={"path": "../biomem.db"})
        self.assertEqual(escape_response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
