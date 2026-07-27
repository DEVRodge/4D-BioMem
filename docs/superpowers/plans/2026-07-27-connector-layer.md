# Connector Layer v1.9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a connector layer that imports external fragments into `memory_events` with idempotency and run history.

**Architecture:** Extend `DBManager` with connector tables and source-aware event writes. Add a small `core.connectors` module for filesystem and git item extraction. Expose connector registry, run, run-history, and direct ingest endpoints through FastAPI.

**Tech Stack:** Python 3.10+, FastAPI, SQLite, standard library filesystem/json/subprocess/hashlib, `unittest`.

## Global Constraints

- Connectors write only to `memory_events`.
- `connector_id + source + source_id` is idempotent.
- v1.9 does not add OAuth connectors.
- Tests must disable automatic background maintenance where they need to inspect raw events.

---

### Task 1: Connector Storage

**Files:**
- Modify: `storage/db_manager.py`
- Test: `test_connectors.py`

**Interfaces:**
- Produces `register_connector(...) -> dict`
- Produces `list_connectors(...) -> list[dict]`
- Produces `record_connector_run(...) -> dict`
- Produces `list_connector_runs(...) -> list[dict]`
- Extends `save_event(..., source=None, source_id=None, connector_id=None) -> dict`

- [x] **Step 1: Write failing storage tests**
- [x] **Step 2: Implement additive schema and migrations**
- [x] **Step 3: Verify storage tests pass**

### Task 2: Connector Extractors

**Files:**
- Create: `core/connectors.py`
- Test: `test_connectors.py`

**Interfaces:**
- Produces `extract_connector_items(connector_type: str, config: dict) -> list[dict]`

- [x] **Step 1: Write failing filesystem/git extractor tests**
- [x] **Step 2: Implement filesystem and git extraction**
- [x] **Step 3: Verify extractor tests pass**

### Task 3: Connector API

**Files:**
- Modify: `api/main.py`
- Test: `test_connectors.py`

**Interfaces:**
- Produces `POST /v1/connectors/register`
- Produces `GET /v1/connectors`
- Produces `POST /v1/connectors/run`
- Produces `GET /v1/connectors/runs`
- Produces `POST /v1/connectors/ingest`

- [x] **Step 1: Write failing API tests**
- [x] **Step 2: Implement connector routes and run orchestration**
- [x] **Step 3: Verify API tests pass**

### Task 4: Docs and Regression

**Files:**
- Modify: `README.md`

**Verification:**

```bash
python -m unittest test_connectors.py test_maintenance.py test_memory_events.py test_memory_wiki.py test_memory_tree.py -v
python -m py_compile api/main.py storage/db_manager.py core/connectors.py core/memory_wiki.py config.py
python test_api.py
```
