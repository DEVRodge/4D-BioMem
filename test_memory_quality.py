from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from api.main import create_app


class MemoryQualityApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_quality_api_")
        self.app = create_app(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            prefer_chroma=False,
            seed=False,
            auto_maintenance_enabled=False,
        )
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add_memory(self, content: str, task_tags: dict | None = None) -> dict:
        response = self.client.post(
            "/v1/memory/add",
            json={"user_id": "hermes", "content": content, "task_tags": task_tags or {}},
        )
        self.assertEqual(response.status_code, 200)
        for _ in range(50):
            items = self.client.get("/v1/memory/list", params={"user_id": "hermes"}).json()["items"]
            for item in items:
                if item["content"] == content:
                    return item
            time.sleep(0.05)
        self.fail(f"memory was not persisted: {content}")

    def test_retrieve_can_return_score_and_pathway_trace(self) -> None:
        self._add_memory(
            "项目 Alpha 的 Bug 修复方案采用重试队列加幂等键",
            {"project": "Alpha", "type": "tech"},
        )
        self._add_memory("我对青霉素过敏，开药务必避开青霉素类", {"type": "medical"})

        response = self.client.post(
            "/v1/memory/retrieve",
            json={
                "user_id": "hermes",
                "query": "项目 Alpha Bug 修复",
                "query_tags": {"project": "Alpha", "type": "tech"},
                "top_k": 3,
                "include_trace": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("trace", body)
        self.assertEqual(body["trace"]["effective_tags"]["project"], "Alpha")
        self.assertTrue(body["trace"]["soft_activated"])
        self.assertGreaterEqual(body["trace"]["candidate_count"], 2)
        trace_by_id = {item["id"]: item for item in body["trace"]["hits"]}
        self.assertEqual(len(body["hits"]), body["trace"]["selected_count"])
        for hit in body["hits"]:
            self.assertIn(hit["id"], trace_by_id)
            self.assertTrue(trace_by_id[hit["id"]]["selected"])
            self.assertIn("detail", trace_by_id[hit["id"]])

    def test_memory_feedback_is_persisted_and_listed(self) -> None:
        memory = self._add_memory("项目 Alpha 检索反馈要能闭环", {"project": "Alpha", "type": "tech"})

        response = self.client.post(
            "/v1/memory/feedback",
            json={
                "user_id": "hermes",
                "memory_id": memory["id"],
                "feedback": "wrong",
                "note": "这条命中不该出现在 Beta 查询里",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "recorded")
        self.assertEqual(body["feedback"]["memory_id"], memory["id"])
        self.assertEqual(body["feedback"]["feedback"], "wrong")

        list_response = self.client.get(
            "/v1/memory/feedback",
            params={"user_id": "hermes", "memory_id": memory["id"]},
        )
        self.assertEqual(list_response.status_code, 200)
        listed = list_response.json()
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["items"][0]["note"], "这条命中不该出现在 Beta 查询里")

    def test_archive_day_can_create_atomic_memory_cells(self) -> None:
        for content in ["用户确认 v2.0 要做记忆反馈", "检索结果需要返回可解释 trace"]:
            response = self.client.post(
                "/v1/memory/ingest_event",
                json={
                    "user_id": "hermes",
                    "agent_id": "codex",
                    "content": content,
                    "event_type": "decision",
                    "task_tags": {"project": "4D-BioMem"},
                    "occurred_at": "2026-07-28T09:30:00+00:00",
                },
            )
            self.assertEqual(response.status_code, 200)

        archive_response = self.client.post(
            "/v1/memory/archive_day",
            json={
                "user_id": "hermes",
                "agent_id": "codex",
                "date": "2026-07-28",
                "atomize": True,
            },
        )

        self.assertEqual(archive_response.status_code, 200)
        body = archive_response.json()
        self.assertEqual(body["status"], "archived")
        self.assertEqual(body["event_count"], 2)
        self.assertEqual(len(body["atomic_cell_ids"]), 2)

        memory_response = self.client.get("/v1/memory/list", params={"user_id": "hermes"})
        items = memory_response.json()["items"]
        atomic = [item for item in items if item["task_tags"].get("type") == "atomic_memory"]
        daily = [item for item in items if item["task_tags"].get("type") == "daily_archive"]
        self.assertEqual(len(atomic), 2)
        self.assertEqual(len(daily), 1)
        self.assertTrue(all(item["task_tags"].get("archive_cell_id") == body["archive_cell_id"] for item in atomic))


if __name__ == "__main__":
    unittest.main()
