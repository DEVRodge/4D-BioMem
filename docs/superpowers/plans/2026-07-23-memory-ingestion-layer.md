# Memory Ingestion Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add daily memory fragment ingestion and manual daily archiving for v1.6.

**Architecture:** Add a SQLite-only `memory_events` table to the storage layer, expose append/list/archive endpoints in FastAPI, and update the dashboard to display daily fragments next to long-term memory. Archiving reuses the existing auditor, embedder, and `save_memory` path.

**Tech Stack:** FastAPI, SQLite, vanilla JavaScript dashboard, Python `unittest` scripts.

## Global Constraints

- Do not mutate existing `memory_cells` schema.
- Do not add automatic scheduled archiving in v1.6.
- Keep event ingestion additive and safe for Docker deployment.
- Use `unittest`, not `pytest`, because the local environment does not include pytest.

---

### Task 1: Event Storage Contract

**Files:**
- Modify: `storage/db_manager.py`
- Create: `test_memory_events.py`

**Interfaces:**
- Produces: `save_event(...) -> dict`
- Produces: `list_events(...) -> list[dict]`
- Produces: `mark_events_archived(event_ids: list[str], archive_cell_id: str) -> None`

- [ ] Write failing storage test with two events on the same day.
- [ ] Run `python -m unittest test_memory_events.py -v` and verify missing methods fail.
- [ ] Implement event table and methods.
- [ ] Run `python -m unittest test_memory_events.py -v` and verify pass.

### Task 2: API Ingestion And Archive

**Files:**
- Modify: `api/main.py`
- Modify: `test_memory_events.py`

**Interfaces:**
- Produces: `POST /v1/memory/ingest_event`
- Produces: `GET /v1/memory/events`
- Produces: `POST /v1/memory/archive_day`

- [ ] Add API tests using `create_app` with a temporary database.
- [ ] Run `python -m unittest test_memory_events.py -v` and verify API routes fail.
- [ ] Implement request models and routes.
- [ ] Run `python -m unittest test_memory_events.py -v` and verify pass.

### Task 3: Dashboard Daily Fragments

**Files:**
- Modify: `api/static/index.html`

**Interfaces:**
- Consumes: `GET /v1/memory/events`

- [ ] Add daily fragment panel to the memory tree view.
- [ ] Render event date groups, event content, archived state, and archive cell ids.
- [ ] Validate JavaScript syntax with the existing Node inline script parser.

### Task 4: Docs And Verification

**Files:**
- Modify: `README.md`
- Modify: `api/main.py`

**Interfaces:**
- API version `1.6.0`

- [ ] Update README badge, API table, and changelog.
- [ ] Run `python -m unittest test_memory_tree.py -v`.
- [ ] Run `python -m unittest test_memory_events.py -v`.
- [ ] Run `python test_core.py`, `python test_storage.py`, `python test_retrieval.py`, and `python test_api.py`.
