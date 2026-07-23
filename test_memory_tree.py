from __future__ import annotations

import unittest
from datetime import datetime, timezone

from api.main import _build_memory_tree
from core.memory_cell import MemoryCell


def _cell(
    cell_id: str,
    content: str,
    *,
    user_id: str = "user-a",
    agent_id: str = "hermes",
    task_tags: dict | None = None,
    entities: list[dict] | None = None,
    is_risk: bool = False,
) -> MemoryCell:
    return MemoryCell(
        id=cell_id,
        content=content,
        user_id=user_id,
        agent_id=agent_id,
        task_tags=task_tags or {},
        entities=entities or [],
        is_risk=is_risk,
        base_intensity=8.0 if not is_risk else 10.0,
    )


class MemoryTreeTest(unittest.TestCase):
    def test_memory_tree_groups_rows_into_virtual_files_without_markdown_storage(self) -> None:
        now = datetime(2026, 7, 23, tzinfo=timezone.utc)
        cells = [
            _cell(
                "m1",
                "[项目进展] v1.4.0 已完成检索融合",
                task_tags={"project": "4D-BioMem", "type": "tech"},
                entities=[{"name": "v1.4.0", "type": "version"}],
            ),
            _cell(
                "m2",
                "[用户偏好] tag 描述以后用中文",
                task_tags={"project": "4D-BioMem", "type": "preference"},
            ),
            _cell(
                "m3",
                "我对青霉素过敏",
                user_id="user-b",
                agent_id="biomem-api",
                task_tags={"type": "medical"},
                is_risk=True,
            ),
        ]

        result = _build_memory_tree(cells, now)

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["format"], "virtual-memory-tree")
        self.assertEqual(result["storage"], "sqlite_rows_plus_vector_store")

        root = result["tree"]
        self.assertEqual(root["kind"], "folder")
        self.assertEqual([child["name"] for child in root["children"]], ["user-a", "user-b"])

        user_a = root["children"][0]
        hermes = user_a["children"][0]
        project = hermes["children"][0]
        self.assertEqual(project["name"], "4D-BioMem")

        file_names = [child["name"] for child in project["children"]]
        self.assertEqual(file_names, ["用户偏好.mem", "项目进展.mem"])
        self.assertTrue(all(not name.endswith(".md") for name in file_names))

        progress_file = project["children"][1]
        self.assertEqual(progress_file["kind"], "file")
        self.assertEqual(progress_file["virtual_path"], "user-a/hermes/4D-BioMem/项目进展.mem")
        self.assertEqual(progress_file["memory_count"], 1)
        self.assertEqual(progress_file["items"][0]["id"], "m1")
        self.assertEqual(progress_file["items"][0]["content"], "[项目进展] v1.4.0 已完成检索融合")
        self.assertEqual(progress_file["items"][0]["entities"], [{"name": "v1.4.0", "type": "version"}])

        user_b_file = result["tree"]["children"][1]["children"][0]["children"][0]["children"][0]
        self.assertEqual(user_b_file["name"], "风险与医疗.mem")
        self.assertTrue(user_b_file["items"][0]["is_risk"])


if __name__ == "__main__":
    unittest.main()
