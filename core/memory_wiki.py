"""Generated Markdown wiki projection for 4D-BioMem memories.

The wiki is a derived artifact. SQLite rows and the vector store remain the
source of truth; generated Markdown helps humans and agents inspect memory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.memory_cell import MemoryCell


def _safe_segment(value: str) -> str:
    segment = re.sub(r"[\\/:*?\"<>|]+", "-", str(value).strip())
    segment = re.sub(r"\s+", "-", segment).strip(".-")
    return segment or "unknown"


def _project_name(cell: MemoryCell) -> str:
    project = cell.task_tags.get("project")
    if isinstance(project, str) and project.strip():
        return project.strip()
    return "通用记忆"


def _front_matter(
    *,
    title: str,
    generated_at: str,
    source_memory_ids: list[str] | None = None,
    source_event_ids: list[str] | None = None,
) -> str:
    lines = [
        "---",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"generated_at: {json.dumps(generated_at, ensure_ascii=False)}",
        "source_memory_ids:",
    ]
    for memory_id in source_memory_ids or []:
        lines.append(f"  - {json.dumps(memory_id, ensure_ascii=False)}")
    lines.append("source_event_ids:")
    for event_id in source_event_ids or []:
        lines.append(f"  - {json.dumps(event_id, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def _write_page(
    output_dir: Path,
    *,
    path: str,
    title: str,
    body: str,
    generated_at: str,
    source_memory_ids: list[str] | None = None,
    source_event_ids: list[str] | None = None,
) -> dict[str, Any]:
    page_path = output_dir / path
    page_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        _front_matter(
            title=title,
            generated_at=generated_at,
            source_memory_ids=source_memory_ids,
            source_event_ids=source_event_ids,
        )
        + "\n\n"
        + body.rstrip()
        + "\n"
    )
    page_path.write_text(content, encoding="utf-8")
    return {
        "path": path,
        "title": title,
        "source_memory_ids": source_memory_ids or [],
        "source_event_ids": source_event_ids or [],
        "bytes": len(content.encode("utf-8")),
    }


def _format_cell(cell: MemoryCell) -> str:
    tags = ", ".join(f"{key}={value}" for key, value in sorted(cell.task_tags.items()))
    entities = ", ".join(
        item.get("name", "") for item in cell.entities if isinstance(item, dict) and item.get("name")
    )
    meta = f"id={cell.id}"
    if tags:
        meta += f" | tags={tags}"
    if entities:
        meta += f" | entities={entities}"
    return f"- {cell.created_at.isoformat()} | {meta}\n  {cell.content}"


def _format_event(event: dict[str, Any]) -> str:
    occurred_at = event.get("occurred_at") or event.get("created_at") or ""
    event_type = event.get("event_type") or "generic"
    archived = "archived" if event.get("archived") else "open"
    return f"- {occurred_at} | {event.get('id')} | {archived}\n  [{event_type}] {event.get('content', '')}"


def build_memory_wiki(
    *,
    cells: list[MemoryCell],
    events: list[dict[str, Any]],
    output_dir: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Generate a Markdown wiki from memory cells and daily events."""
    generated_at = (now or datetime.now(tz=timezone.utc)).isoformat()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    cells_sorted = sorted(cells, key=lambda c: (c.user_id, c.agent_id, _project_name(c), c.created_at.isoformat(), c.id))
    events_sorted = sorted(
        events,
        key=lambda e: (
            e.get("user_id", ""),
            e.get("agent_id", ""),
            e.get("occurred_at", ""),
            e.get("created_at", ""),
            e.get("id", ""),
        ),
    )

    user_agent_cells: dict[tuple[str, str], list[MemoryCell]] = defaultdict(list)
    project_cells: dict[tuple[str, str, str], list[MemoryCell]] = defaultdict(list)
    for cell in cells_sorted:
        user_agent_cells[(cell.user_id, cell.agent_id)].append(cell)
        project_cells[(cell.user_id, cell.agent_id, _project_name(cell))].append(cell)

    user_agent_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    daily_events: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events_sorted:
        user_id = str(event.get("user_id", "unknown"))
        agent_id = str(event.get("agent_id", "unknown"))
        date = str(event.get("occurred_at") or event.get("created_at") or "unknown-date")[:10]
        user_agent_events[(user_id, agent_id)].append(event)
        daily_events[(user_id, agent_id, date)].append(event)

    user_agents = sorted(set(user_agent_cells) | set(user_agent_events))
    index_lines = [
        "# 4D-BioMem Memory Wiki",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Memory cells: `{len(cells_sorted)}`",
        f"- Daily events: `{len(events_sorted)}`",
        "",
        "## Agents",
    ]
    for user_id, agent_id in user_agents:
        user_seg = _safe_segment(user_id)
        agent_seg = _safe_segment(agent_id)
        index_lines.append(
            f"- [{user_id}/{agent_id}](users/{user_seg}/agents/{agent_seg}/index.md)"
        )
    pages.append(
        _write_page(
            root,
            path="index.md",
            title="4D-BioMem Memory Wiki",
            body="\n".join(index_lines),
            generated_at=generated_at,
            source_memory_ids=[cell.id for cell in cells_sorted],
            source_event_ids=[str(event.get("id")) for event in events_sorted],
        )
    )

    for user_id, agent_id in user_agents:
        cells_for_agent = user_agent_cells.get((user_id, agent_id), [])
        events_for_agent = user_agent_events.get((user_id, agent_id), [])
        projects = sorted({_project_name(cell) for cell in cells_for_agent})
        dates = sorted({str(event.get("occurred_at") or event.get("created_at") or "unknown-date")[:10] for event in events_for_agent})
        user_seg = _safe_segment(user_id)
        agent_seg = _safe_segment(agent_id)
        lines = [
            f"# {user_id}/{agent_id}",
            "",
            f"- Memory cells: `{len(cells_for_agent)}`",
            f"- Daily events: `{len(events_for_agent)}`",
            "",
            "## Projects",
        ]
        for project in projects:
            project_seg = _safe_segment(project)
            lines.append(f"- [{project}](projects/{project_seg}/timeline.md)")
        lines.append("")
        lines.append("## Daily Fragments")
        for date in dates:
            lines.append(f"- [{date}](daily/{date}.md)")
        pages.append(
            _write_page(
                root,
                path=f"users/{user_seg}/agents/{agent_seg}/index.md",
                title=f"{user_id}/{agent_id}",
                body="\n".join(lines),
                generated_at=generated_at,
                source_memory_ids=[cell.id for cell in cells_for_agent],
                source_event_ids=[str(event.get("id")) for event in events_for_agent],
            )
        )

    for (user_id, agent_id, project), grouped_cells in sorted(project_cells.items()):
        user_seg = _safe_segment(user_id)
        agent_seg = _safe_segment(agent_id)
        project_seg = _safe_segment(project)
        body = "\n".join(
            [
                f"# {project} 时间线",
                "",
                *[_format_cell(cell) for cell in grouped_cells],
            ]
        )
        pages.append(
            _write_page(
                root,
                path=f"users/{user_seg}/agents/{agent_seg}/projects/{project_seg}/timeline.md",
                title=f"{project} 时间线",
                body=body,
                generated_at=generated_at,
                source_memory_ids=[cell.id for cell in grouped_cells],
            )
        )

    for (user_id, agent_id, date), grouped_events in sorted(daily_events.items()):
        user_seg = _safe_segment(user_id)
        agent_seg = _safe_segment(agent_id)
        body = "\n".join(
            [
                f"# {date} 每日片段",
                "",
                *[_format_event(event) for event in grouped_events],
            ]
        )
        pages.append(
            _write_page(
                root,
                path=f"users/{user_seg}/agents/{agent_seg}/daily/{date}.md",
                title=f"{date} 每日片段",
                body=body,
                generated_at=generated_at,
                source_event_ids=[str(event.get("id")) for event in grouped_events],
            )
        )

    manifest = {
        "format": "4d-biomem-memory-wiki",
        "generated_at": generated_at,
        "page_count": len(pages),
        "pages": pages,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = ["build_memory_wiki"]
