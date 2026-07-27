from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from api.main import create_app
from core.connectors import extract_connector_items
from storage.db_manager import DBManager


class ConnectorStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_connectors_")
        self.db = DBManager(
            db_path=os.path.join(self.tmp, "biomem.db"),
            vector_path=os.path.join(self.tmp, "vectors"),
            prefer_chroma=False,
        )

    def tearDown(self) -> None:
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_connector_registry_runs_and_source_event_idempotency(self) -> None:
        connector = self.db.register_connector(
            connector_id="fs-main",
            name="Local Inbox",
            connector_type="filesystem",
            config={"path": "/tmp/inbox", "user_id": "hermes"},
        )
        self.assertEqual(connector["id"], "fs-main")
        self.assertTrue(connector["enabled"])

        connectors = self.db.list_connectors()
        self.assertEqual(len(connectors), 1)
        self.assertEqual(connectors[0]["config"]["path"], "/tmp/inbox")

        first = self.db.save_event(
            user_id="hermes",
            agent_id="filesystem",
            content="从文件导入的片段",
            event_type="connector",
            occurred_at=datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc),
            connector_id="fs-main",
            source="filesystem",
            source_id="file-1",
        )
        duplicate = self.db.save_event(
            user_id="hermes",
            agent_id="filesystem",
            content="从文件导入的片段",
            event_type="connector",
            occurred_at=datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc),
            connector_id="fs-main",
            source="filesystem",
            source_id="file-1",
        )

        self.assertEqual(first["id"], duplicate["id"])
        self.assertTrue(duplicate["duplicate"])
        events = self.db.list_events(user_id="hermes")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["connector_id"], "fs-main")
        self.assertEqual(events[0]["source"], "filesystem")
        self.assertEqual(events[0]["source_id"], "file-1")

        run = self.db.record_connector_run(
            connector_id="fs-main",
            status="completed",
            imported_count=1,
            skipped_count=1,
            details={"seen": 2},
        )
        self.assertEqual(run["connector_id"], "fs-main")
        self.assertEqual(run["details"], {"seen": 2})
        self.assertEqual(self.db.list_connector_runs(connector_id="fs-main")[0]["imported_count"], 1)

    def test_migration_preserves_events_when_duplicate_connector_identity_exists(self) -> None:
        self.db.close()
        db_path = os.path.join(self.tmp, "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE memory_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                event_type TEXT NOT NULL,
                task_tags TEXT,
                created_at TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                archive_cell_id TEXT,
                source TEXT,
                source_id TEXT,
                connector_id TEXT
            );
            INSERT INTO memory_events
                (id, user_id, agent_id, content, event_type, task_tags, created_at, occurred_at,
                 archived, archive_cell_id, source, source_id, connector_id)
            VALUES
                ('e1', 'hermes', 'filesystem', 'first', 'connector', '{}',
                 '2026-07-27T08:00:00+00:00', '2026-07-27T08:00:00+00:00',
                 0, NULL, 'filesystem', 'same', 'fs-main'),
                ('e2', 'hermes', 'filesystem', 'second', 'connector', '{}',
                 '2026-07-27T08:01:00+00:00', '2026-07-27T08:01:00+00:00',
                 0, NULL, 'filesystem', 'same', 'fs-main');
            """
        )
        conn.commit()
        conn.close()

        migrated = DBManager(
            db_path=db_path,
            vector_path=os.path.join(self.tmp, "legacy_vectors"),
            prefer_chroma=False,
        )
        try:
            events = migrated.list_events(user_id="hermes")
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["source_id"], "same")
            self.assertTrue(events[1]["source_id"].startswith("same#duplicate:e2"))
            duplicate = migrated.save_event(
                user_id="hermes",
                agent_id="filesystem",
                content="again",
                connector_id="fs-main",
                source="filesystem",
                source_id="same",
            )
            self.assertTrue(duplicate["duplicate"])
            self.assertEqual(duplicate["id"], "e1")
        finally:
            migrated.close()


class ConnectorExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_connector_extract_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_filesystem_extracts_text_markdown_and_jsonl_items(self) -> None:
        with open(os.path.join(self.tmp, "note.md"), "w", encoding="utf-8") as f:
            f.write("# 项目进展\nv1.9 开始做连接器层\n")
        with open(os.path.join(self.tmp, "clip.txt"), "w", encoding="utf-8") as f:
            f.write("用户希望本机文件可以进入记忆片段")
        with open(os.path.join(self.tmp, "events.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"content":"jsonl 片段","event_type":"decision","occurred_at":"2026-07-27T08:00:00+00:00","source_id":"jsonl-1"}\n')

        items = extract_connector_items(
            "filesystem",
            {"path": self.tmp, "user_id": "hermes", "agent_id": "filesystem", "project": "4D-BioMem"},
        )

        self.assertEqual(len(items), 3)
        contents = [item["content"] for item in items]
        self.assertTrue(any("v1.9 开始做连接器层" in content for content in contents))
        self.assertTrue(any("jsonl 片段" == content for content in contents))
        self.assertTrue(all(item["source"] == "filesystem" for item in items))
        self.assertTrue(all(item["source_id"] for item in items))

    def test_filesystem_source_ids_are_relative_to_connector_root(self) -> None:
        first_root = os.path.join(self.tmp, "mount-a")
        second_root = os.path.join(self.tmp, "mount-b")
        os.makedirs(os.path.join(first_root, "notes"))
        os.makedirs(os.path.join(second_root, "notes"))
        for root in (first_root, second_root):
            with open(os.path.join(root, "notes", "same.txt"), "w", encoding="utf-8") as f:
                f.write("同一条宿主记忆")

        first = extract_connector_items("filesystem", {"path": first_root})
        second = extract_connector_items("filesystem", {"path": second_root})

        self.assertEqual(first[0]["source_id"], second[0]["source_id"])
        self.assertEqual(first[0]["source_id"], "filesystem:notes/same.txt:3aed028ebd3d70b2")

    def test_filesystem_limits_items_and_skips_oversized_files(self) -> None:
        with open(os.path.join(self.tmp, "a.txt"), "w", encoding="utf-8") as f:
            f.write("导入一")
        with open(os.path.join(self.tmp, "b.txt"), "w", encoding="utf-8") as f:
            f.write("导入二")
        with open(os.path.join(self.tmp, "large.txt"), "w", encoding="utf-8") as f:
            f.write("x" * 20)

        items = extract_connector_items(
            "filesystem",
            {"path": self.tmp, "max_items": 1, "max_file_bytes": 10},
        )

        self.assertEqual(len(items), 1)
        self.assertNotIn("x" * 20, items[0]["content"])

    def test_git_extracts_commit_log_items(self) -> None:
        subprocess.run(["git", "init"], cwd=self.tmp, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.tmp, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.tmp, check=True)
        with open(os.path.join(self.tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write("hello")
        subprocess.run(["git", "add", "README.md"], cwd=self.tmp, check=True)
        subprocess.run(["git", "commit", "-m", "feat: add connector notes"], cwd=self.tmp, check=True, stdout=subprocess.DEVNULL)

        items = extract_connector_items(
            "git",
            {"repo_path": self.tmp, "user_id": "hermes", "agent_id": "git", "project": "4D-BioMem", "max_commits": 5},
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "git")
        self.assertIn("feat: add connector notes", items[0]["content"])
        self.assertEqual(items[0]["task_tags"]["project"], "4D-BioMem")


class ConnectorApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="biomem_connector_api_")
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

    def test_direct_connector_ingest_is_idempotent_and_records_run(self) -> None:
        register = self.client.post(
            "/v1/connectors/register",
            json={
                "id": "hermes-manual",
                "name": "Hermes Manual",
                "connector_type": "hermes_manual",
                "config": {"user_id": "hermes", "agent_id": "hermes"},
            },
        )
        self.assertEqual(register.status_code, 200)

        payload = {
            "connector_id": "hermes-manual",
            "items": [
                {
                    "user_id": "hermes",
                    "agent_id": "hermes",
                    "content": "Hermes 手动推送 v1.9 片段",
                    "event_type": "observation",
                    "source": "hermes_manual",
                    "source_id": "manual-1",
                    "occurred_at": "2026-07-27T09:30:00+00:00",
                    "task_tags": {"project": "4D-BioMem"},
                },
                {
                    "user_id": "hermes",
                    "agent_id": "hermes",
                    "content": "Hermes 手动推送 v1.9 片段",
                    "event_type": "observation",
                    "source": "hermes_manual",
                    "source_id": "manual-1",
                    "occurred_at": "2026-07-27T09:30:00+00:00",
                    "task_tags": {"project": "4D-BioMem"},
                },
            ],
        }
        ingest = self.client.post("/v1/connectors/ingest", json=payload)

        self.assertEqual(ingest.status_code, 200)
        self.assertEqual(ingest.json()["imported_count"], 1)
        self.assertEqual(ingest.json()["skipped_count"], 1)

        events = self.client.get("/v1/memory/events", params={"user_id": "hermes"}).json()
        self.assertEqual(events["count"], 1)
        self.assertEqual(events["items"][0]["connector_id"], "hermes-manual")

        runs = self.client.get("/v1/connectors/runs", params={"connector_id": "hermes-manual"}).json()
        self.assertEqual(runs["count"], 1)
        self.assertEqual(runs["items"][0]["status"], "completed")

    def test_direct_ingest_requires_registered_enabled_connector(self) -> None:
        payload = {
            "connector_id": "missing",
            "items": [
                {
                    "user_id": "hermes",
                    "content": "不应该写入",
                    "source": "hermes_manual",
                    "source_id": "manual-1",
                }
            ],
        }
        missing = self.client.post("/v1/connectors/ingest", json=payload)
        self.assertEqual(missing.status_code, 404)

        self.client.post(
            "/v1/connectors/register",
            json={
                "id": "disabled-manual",
                "name": "Disabled Manual",
                "connector_type": "hermes_manual",
                "enabled": False,
            },
        )
        payload["connector_id"] = "disabled-manual"
        disabled = self.client.post("/v1/connectors/ingest", json=payload)
        self.assertEqual(disabled.status_code, 400)

        events = self.client.get("/v1/memory/events", params={"user_id": "hermes"}).json()
        self.assertEqual(events["count"], 0)

    def test_direct_ingest_rejects_blank_source_identity_and_records_failed_run(self) -> None:
        self.client.post(
            "/v1/connectors/register",
            json={"id": "manual", "name": "Manual", "connector_type": "hermes_manual"},
        )
        bad = self.client.post(
            "/v1/connectors/ingest",
            json={
                "connector_id": "manual",
                "items": [
                    {
                        "user_id": "hermes",
                        "content": "source_id 是空白时不应写入",
                        "source": "hermes_manual",
                        "source_id": " ",
                    }
                ],
            },
        )

        self.assertEqual(bad.status_code, 400)
        events = self.client.get("/v1/memory/events", params={"user_id": "hermes"}).json()
        self.assertEqual(events["count"], 0)
        runs = self.client.get("/v1/connectors/runs", params={"connector_id": "manual"}).json()
        self.assertEqual(runs["count"], 1)
        self.assertEqual(runs["items"][0]["status"], "failed")

    def test_run_filesystem_connector_imports_new_items_once(self) -> None:
        inbox = os.path.join(self.tmp, "inbox")
        os.makedirs(inbox)
        with open(os.path.join(inbox, "note.txt"), "w", encoding="utf-8") as f:
            f.write("文件连接器导入的片段")
        self.client.post(
            "/v1/connectors/register",
            json={
                "id": "fs-main",
                "name": "Local Inbox",
                "connector_type": "filesystem",
                "config": {"path": inbox, "user_id": "hermes", "agent_id": "filesystem"},
            },
        )

        first = self.client.post("/v1/connectors/run", json={"connector_id": "fs-main"})
        second = self.client.post("/v1/connectors/run", json={"connector_id": "fs-main"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["imported_count"], 1)
        self.assertEqual(first.json()["skipped_count"], 0)
        self.assertEqual(second.json()["imported_count"], 0)
        self.assertEqual(second.json()["skipped_count"], 1)


if __name__ == "__main__":
    unittest.main()
