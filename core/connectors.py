"""Connector item extraction for 4D-BioMem.

Connectors normalize external sources into memory_event-shaped dictionaries.
They do not write long-term memories directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _base_item(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": config.get("user_id", "hermes"),
        "agent_id": config.get("agent_id", "connector"),
        "event_type": config.get("event_type", "connector"),
        "task_tags": {"source": "connector", **({"project": config["project"]} if config.get("project") else {})},
    }


def _positive_int(config: dict[str, Any], key: str, default: int, maximum: int) -> int:
    value = int(config.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return min(value, maximum)


def _filesystem_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(config.get("path", ""))).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"filesystem connector path is not a directory: {root}")
    root = root.resolve()
    max_files = _positive_int(config, "max_files", 200, 5000)
    max_items = _positive_int(config, "max_items", 500, 10000)
    max_file_bytes = _positive_int(config, "max_file_bytes", 256 * 1024, 5 * 1024 * 1024)
    base = _base_item(config)
    items: list[dict[str, Any]] = []
    scanned_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".jsonl"}:
            continue
        scanned_files += 1
        if scanned_files > max_files:
            break
        stat = path.stat()
        if stat.st_size > max_file_bytes:
            continue
        rel_path = path.resolve().relative_to(root).as_posix()
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for index, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    content = str(data["content"]).strip()
                    if not content:
                        continue
                    source_id = str(data.get("source_id") or f"filesystem:{rel_path}:jsonl:{_content_hash(content)}")
                    items.append({
                        **base,
                        "content": content,
                        "event_type": data.get("event_type", base["event_type"]),
                        "occurred_at": data.get("occurred_at"),
                        "task_tags": {**base["task_tags"], **data.get("task_tags", {})},
                        "source": "filesystem",
                        "source_id": source_id,
                    })
                    if len(items) >= max_items:
                        return items
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        occurred_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        items.append({
            **base,
            "content": content,
            "occurred_at": occurred_at,
            "source": "filesystem",
            "source_id": f"filesystem:{rel_path}:{_content_hash(content)}",
        })
        if len(items) >= max_items:
            return items
    return items


def _git_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    repo_path = Path(str(config.get("repo_path", "."))).expanduser()
    if not (repo_path / ".git").exists():
        raise ValueError(f"git connector path is not a git repository: {repo_path}")
    max_commits = _positive_int(config, "max_commits", 50, 500)
    timeout_seconds = _positive_int(config, "timeout_seconds", 10, 120)
    fmt = "%H%x1f%aI%x1f%s%x1e"
    result = subprocess.run(
        ["git", "log", f"-n{max_commits}", f"--pretty=format:{fmt}"],
        cwd=str(repo_path),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    base = _base_item({**config, "agent_id": config.get("agent_id", "git"), "event_type": "git_commit"})
    items = []
    for record in result.stdout.strip("\x1e\n").split("\x1e"):
        if not record.strip():
            continue
        commit, authored_at, subject = record.strip().split("\x1f", 2)
        items.append({
            **base,
            "content": f"[Git提交] {subject} ({commit[:8]})",
            "occurred_at": authored_at,
            "source": "git",
            "source_id": commit,
            "task_tags": {**base["task_tags"], "type": "project_progress"},
        })
    return items


def extract_connector_items(connector_type: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized event fragments for a connector type."""
    if connector_type == "filesystem":
        return _filesystem_items(config)
    if connector_type == "git":
        return _git_items(config)
    if connector_type == "hermes_manual":
        return []
    raise ValueError(f"unsupported connector type: {connector_type}")


__all__ = ["extract_connector_items"]
