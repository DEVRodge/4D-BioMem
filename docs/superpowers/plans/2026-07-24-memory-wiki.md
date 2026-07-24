# Memory Wiki v1.7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenWiki-inspired generated Markdown wiki over existing 4D-BioMem memories and daily events.

**Architecture:** Add a focused `core.memory_wiki` module that writes Markdown pages and a manifest from `MemoryCell` rows and event dictionaries. FastAPI exposes build/list/read endpoints, and the dashboard adds a Memory Wiki browser.

**Tech Stack:** Python 3.10+, FastAPI, SQLite via existing `DBManager`, standard-library Markdown file generation, vanilla dashboard JavaScript.

## Global Constraints

- Source of truth remains SQLite `memory_cells`, `memory_events`, and the vector store.
- Generated wiki files live under `WIKI_PATH`, default `/data/wiki`.
- Existing memory rows and events must not be deleted or changed by wiki build.
- Tests use `unittest`; `pytest` is not required.

---

### Task 1: Wiki Builder

**Files:**
- Create: `core/memory_wiki.py`
- Test: `test_memory_wiki.py`

**Interfaces:**
- Consumes: `MemoryCell`, event dictionaries from `DBManager.list_events`.
- Produces: `build_memory_wiki(cells, events, output_dir, now=None) -> dict`.

- [x] **Step 1: Write failing builder test**

Run: `python -m unittest test_memory_wiki.py -v`
Expected: import failure for `core.memory_wiki`.

- [x] **Step 2: Implement builder**

Create deterministic Markdown pages and `manifest.json`.

- [x] **Step 3: Verify builder test passes**

Run: `python -m unittest test_memory_wiki.py -v`
Expected: PASS.

### Task 2: Wiki API

**Files:**
- Modify: `api/main.py`
- Modify: `config.py`
- Test: `test_memory_wiki.py`

**Interfaces:**
- Consumes: `build_memory_wiki`.
- Produces: `POST /v1/wiki/build`, `GET /v1/wiki/pages`, `GET /v1/wiki/page`.

- [x] **Step 1: Write failing API test**

Run: `python -m unittest test_memory_wiki.py -v`
Expected: 404 for `/v1/wiki/build`.

- [x] **Step 2: Implement API and config**

Add `Settings.wiki_path`, `AppState.wiki_path`, route models, and safe page reading.

- [x] **Step 3: Verify API test passes**

Run: `python -m unittest test_memory_wiki.py -v`
Expected: PASS.

### Task 3: Dashboard and Docs

**Files:**
- Modify: `api/static/index.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: `/v1/wiki/pages`, `/v1/wiki/page`, `/v1/wiki/build`.
- Produces: a dashboard `Memory Wiki` tab.

- [x] **Step 1: Add dashboard UI**

Add tab, build button, page list, and Markdown text viewer.

- [x] **Step 2: Update README**

Document v1.7 endpoints and changelog in Chinese.

- [x] **Step 3: Verify**

Run:

```bash
python -m unittest test_memory_wiki.py -v
python -m unittest test_memory_events.py -v
python -m unittest test_memory_tree.py -v
python -m py_compile api/main.py storage/db_manager.py core/memory_wiki.py config.py
```

Expected: all pass.
