# Memory Wiki v1.7 Design

## Goal

v1.7 adds an OpenWiki-inspired Memory Wiki layer to 4D-BioMem. The current SQLite rows, vector store, and memory event fragments remain the source of truth; the wiki is a generated, readable, local Markdown projection for humans and agents.

## Scope

- Generate Markdown pages from `memory_cells` and `memory_events`.
- Store generated pages under `WIKI_PATH`, defaulting to `/data/wiki` in Docker.
- Expose API endpoints to build, list, and read wiki pages.
- Add a dashboard tab to build and browse wiki pages.
- Keep the feature read-only with respect to existing memories, except writing derived wiki files.

## Architecture

`core/memory_wiki.py` owns wiki generation. It consumes `MemoryCell` objects and event dictionaries, groups them by `user_id`, `agent_id`, project, and date, then writes Markdown files with YAML front matter and source IDs.

FastAPI provides:

- `POST /v1/wiki/build`
- `GET /v1/wiki/pages`
- `GET /v1/wiki/page?path=...`

The dashboard adds a third tab, `Memory Wiki`, with a build button, a page list, and a Markdown text viewer.

## Data Model

Generated page examples:

- `index.md`
- `users/hermes/agents/codex/index.md`
- `users/hermes/agents/codex/projects/4D-BioMem/timeline.md`
- `users/hermes/agents/codex/daily/2026-07-23.md`

Each page includes front matter:

```yaml
---
title: 4D-BioMem 时间线
generated_at: "2026-07-24T00:00:00+00:00"
source_memory_ids:
  - memory-id
source_event_ids:
  - event-id
---
```

## Constraints

- Do not migrate or rewrite existing memory storage.
- Do not delete or mutate live `memory_cells` or `memory_events`.
- Avoid dependencies; generate plain Markdown using the Python standard library.
- Use Chinese user-facing docs and changelog entries.

## Testing

- Unit test the wiki builder with deterministic memory cells and events.
- API test build/list/read endpoints using a temporary database and wiki directory.
- Regression test existing memory tree and events tests.
