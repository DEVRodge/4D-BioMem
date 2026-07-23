from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import create_app
from storage.db_manager import DBManager


class MemoryEventStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_events_")
        self.db = DBManager(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            prefer_chroma=False,
        )

    def tearDown(self) -> None:
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_events_can_be_saved_listed_and_marked_archived(self) -> None:
        first = self.db.save_event(
            user_id="hermes",
            agent_id="codex",
            content="用户确认 v1.6 要做每日片段摄取",
            event_type="decision",
            task_tags={"project": "4D-BioMem"},
            occurred_at=datetime(2026, 7, 23, 9, 30, tzinfo=timezone.utc),
        )
        second = self.db.save_event(
            user_id="hermes",
            agent_id="codex",
            content="左侧记忆树需要支持折叠",
            event_type="observation",
            task_tags={"project": "4D-BioMem"},
            occurred_at=datetime(2026, 7, 23, 10, 0, tzinfo=timezone.utc),
        )

        events = self.db.list_events(user_id="hermes", date="2026-07-23")

        self.assertEqual([event["id"] for event in events], [first["id"], second["id"]])
        self.assertEqual(events[0]["content"], "用户确认 v1.6 要做每日片段摄取")
        self.assertEqual(events[0]["task_tags"], {"project": "4D-BioMem"})
        self.assertFalse(events[0]["archived"])

        self.db.mark_events_archived([first["id"], second["id"]], "archive-cell-1")
        archived = self.db.list_events(user_id="hermes", date="2026-07-23", archived=True)

        self.assertEqual(len(archived), 2)
        self.assertTrue(all(event["archived"] for event in archived))
        self.assertEqual({event["archive_cell_id"] for event in archived}, {"archive-cell-1"})


class MemoryEventApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_events_api_")
        self.app = create_app(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            prefer_chroma=False,
            seed=False,
        )
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ingest_list_and_archive_day(self) -> None:
        payloads = [
            {
                "user_id": "hermes",
                "agent_id": "codex",
                "content": "用户确认 v1.6 要做每日片段摄取",
                "event_type": "decision",
                "task_tags": {"project": "4D-BioMem"},
                "occurred_at": "2026-07-23T09:30:00+00:00",
            },
            {
                "user_id": "hermes",
                "agent_id": "codex",
                "content": "左侧记忆树需要支持折叠",
                "event_type": "observation",
                "task_tags": {"project": "4D-BioMem"},
                "occurred_at": "2026-07-23T10:00:00+00:00",
            },
        ]

        for payload in payloads:
            response = self.client.post("/v1/memory/ingest_event", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "stored")

        list_response = self.client.get(
            "/v1/memory/events",
            params={"user_id": "hermes", "date": "2026-07-23", "archived": "false"},
        )
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["count"], 2)

        archive_response = self.client.post(
            "/v1/memory/archive_day",
            json={"user_id": "hermes", "agent_id": "codex", "date": "2026-07-23"},
        )
        self.assertEqual(archive_response.status_code, 200)
        archive_body = archive_response.json()
        self.assertEqual(archive_body["status"], "archived")
        self.assertEqual(archive_body["event_count"], 2)
        self.assertIn("[每日片段] 2026-07-23 hermes/codex 共 2 条", archive_body["content"])

        # The memory write path is async; wait for the archive cell to appear.
        archive_id = archive_body["archive_cell_id"]
        for _ in range(50):
            memory_response = self.client.get("/v1/memory/list", params={"user_id": "hermes"})
            items = memory_response.json()["items"]
            if any(item["id"] == archive_id for item in items):
                break
            time.sleep(0.05)
        else:
            self.fail("archive memory cell was not persisted")

        archived_response = self.client.get(
            "/v1/memory/events",
            params={"user_id": "hermes", "date": "2026-07-23", "archived": "true"},
        )
        archived_items = archived_response.json()["items"]
        self.assertEqual(len(archived_items), 2)
        self.assertEqual({item["archive_cell_id"] for item in archived_items}, {archive_id})


if __name__ == "__main__":
    unittest.main()
