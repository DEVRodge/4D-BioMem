from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import create_app
from storage.db_manager import DBManager


class MaintenanceStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_maintenance_")
        self.db = DBManager(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            prefer_chroma=False,
        )

    def tearDown(self) -> None:
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unarchived_event_groups_skip_today_by_default(self) -> None:
        yesterday_a = self.db.save_event(
            user_id="hermes",
            agent_id="codex",
            content="昨天第一个片段",
            event_type="observation",
            occurred_at=datetime(2026, 7, 23, 9, 0, tzinfo=timezone.utc),
        )
        self.db.save_event(
            user_id="hermes",
            agent_id="codex",
            content="昨天第二个片段",
            event_type="decision",
            occurred_at=datetime(2026, 7, 23, 10, 0, tzinfo=timezone.utc),
        )
        self.db.save_event(
            user_id="hermes",
            agent_id="biomem-api",
            content="前天片段",
            event_type="observation",
            occurred_at=datetime(2026, 7, 22, 11, 0, tzinfo=timezone.utc),
        )
        self.db.save_event(
            user_id="hermes",
            agent_id="codex",
            content="今天还在继续的片段",
            event_type="observation",
            occurred_at=datetime(2026, 7, 24, 8, 0, tzinfo=timezone.utc),
        )
        self.db.mark_events_archived([yesterday_a["id"]], "existing-archive")

        groups = self.db.list_unarchived_event_groups(today="2026-07-24")

        self.assertEqual(
            groups,
            [
                {"user_id": "hermes", "agent_id": "biomem-api", "date": "2026-07-22", "event_count": 1},
                {"user_id": "hermes", "agent_id": "codex", "date": "2026-07-23", "event_count": 1},
            ],
        )

        groups_with_today = self.db.list_unarchived_event_groups(today="2026-07-24", include_today=True)
        self.assertEqual([group["date"] for group in groups_with_today], ["2026-07-22", "2026-07-23", "2026-07-24"])


class MaintenanceApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_maintenance_api_")
        self.app = create_app(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            wiki_path=os.path.join(self.tmp, "wiki"),
            prefer_chroma=False,
            seed=False,
            auto_maintenance_enabled=False,
        )
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _event(self, content: str, occurred_at: str, *, agent_id: str = "codex") -> None:
        response = self.client.post(
            "/v1/memory/ingest_event",
            json={
                "user_id": "hermes",
                "agent_id": agent_id,
                "content": content,
                "event_type": "observation",
                "task_tags": {"project": "4D-BioMem"},
                "occurred_at": occurred_at,
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_run_once_archives_backlog_skips_today_and_refreshes_wiki(self) -> None:
        self._event("前天 Hermes 片段", "2026-07-22T10:00:00+00:00", agent_id="biomem-api")
        self._event("昨天第一个片段", "2026-07-23T09:00:00+00:00")
        self._event("昨天第二个片段", "2026-07-23T11:00:00+00:00")
        self._event("今天还在继续，不应自动归档", "2026-07-24T08:30:00+00:00")

        run_response = self.client.post(
            "/v1/maintenance/run_once",
            json={"trigger": "manual-test", "today": "2026-07-24"},
        )

        self.assertEqual(run_response.status_code, 200)
        result = run_response.json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["groups_archived"], 2)
        self.assertEqual(result["events_archived"], 3)
        self.assertGreaterEqual(result["wiki_page_count"], 1)

        archived_response = self.client.get(
            "/v1/memory/events",
            params={"user_id": "hermes", "archived": "true"},
        )
        self.assertEqual(archived_response.status_code, 200)
        self.assertEqual(archived_response.json()["count"], 3)

        today_response = self.client.get(
            "/v1/memory/events",
            params={"user_id": "hermes", "date": "2026-07-24", "archived": "false"},
        )
        self.assertEqual(today_response.status_code, 200)
        self.assertEqual(today_response.json()["count"], 1)

        memory_response = self.client.get("/v1/memory/list", params={"user_id": "hermes"})
        daily_archives = [
            item for item in memory_response.json()["items"]
            if item["task_tags"].get("type") == "daily_archive"
        ]
        self.assertEqual(len(daily_archives), 2)

        pages_response = self.client.get("/v1/wiki/pages")
        self.assertEqual(pages_response.status_code, 200)
        self.assertGreaterEqual(pages_response.json()["page_count"], 1)

    def test_status_exposes_scheduler_settings_and_last_run(self) -> None:
        status_response = self.client.get("/v1/maintenance/status")

        self.assertEqual(status_response.status_code, 200)
        status = status_response.json()
        self.assertFalse(status["enabled"])
        self.assertEqual(status["maintenance_time"], "03:30")
        self.assertEqual(status["maintenance_timezone"], "Asia/Shanghai")
        self.assertEqual(status["periodic_scan_minutes"], 30)
        self.assertIsNone(status["last_run"])


class MaintenanceStartupCatchupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_maintenance_startup_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_startup_catches_up_missed_archives(self) -> None:
        db_path = os.path.join(self.tmp, "biomem.db")
        vector_path = os.path.join(self.tmp, "vectors")
        with DBManager(db_path=db_path, vector_path=vector_path, prefer_chroma=False) as db:
            db.save_event(
                user_id="hermes",
                agent_id="codex",
                content="昨天部署环境关机，启动后要补归档",
                event_type="observation",
                task_tags={"project": "4D-BioMem"},
                occurred_at=datetime(2026, 7, 23, 22, 0, tzinfo=timezone.utc),
            )

        app = create_app(
            db_path=db_path,
            vector_path=vector_path,
            wiki_path=os.path.join(self.tmp, "wiki"),
            prefer_chroma=False,
            seed=False,
            auto_maintenance_enabled=True,
            maintenance_interval_minutes=60,
        )
        with TestClient(app) as client:
            import time

            for _ in range(50):
                status = client.get("/v1/maintenance/status").json()
                if status["last_run"]:
                    break
                time.sleep(0.05)
            else:
                self.fail("startup maintenance did not run")

            self.assertTrue(status["enabled"])
            self.assertEqual(status["last_run"]["trigger"], "startup")
            self.assertEqual(status["last_run"]["groups_archived"], 1)
            self.assertEqual(status["last_run"]["events_archived"], 1)

            archived = client.get(
                "/v1/memory/events",
                params={"user_id": "hermes", "archived": "true"},
            ).json()
            self.assertEqual(archived["count"], 1)

            pages = client.get("/v1/wiki/pages").json()
            self.assertGreaterEqual(pages["page_count"], 1)


if __name__ == "__main__":
    unittest.main()
