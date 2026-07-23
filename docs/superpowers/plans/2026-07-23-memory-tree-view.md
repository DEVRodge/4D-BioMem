# Memory Tree View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only web memory tree viewer for the existing dashboard.

**Architecture:** Add a backend tree serializer over existing `MemoryCell` rows, expose it through `GET /v1/memory/tree`, and render it in `api/static/index.html` as a virtual file tree. The storage layer remains unchanged.

**Tech Stack:** FastAPI, SQLite-backed `DBManager`, static HTML, Tailwind CDN, vanilla JavaScript.

## Global Constraints

- Do not change the existing SQLite schema or vector store.
- Do not treat `.mem` virtual files as real disk files.
- Keep the live Docker data safe by making the feature read-only.
- Preserve existing dashboard monitoring and pruning behavior.

---

### Task 1: Backend Tree Contract

**Files:**
- Create: `test_memory_tree.py`
- Modify: `api/main.py`

**Interfaces:**
- Consumes: `MemoryCell`
- Produces: `_build_memory_tree(cells: list[MemoryCell], now: datetime) -> dict`
- Produces: `GET /v1/memory/tree?user_id=...`

- [ ] **Step 1: Write the failing test**

Create tests that seed `MemoryCell` objects and assert the tree groups them by `user_id / agent_id / project / virtual-file`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest test_memory_tree.py -v`

- [ ] **Step 3: Implement tree serialization**

Add deterministic virtual file naming, tree node generation, and route registration.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest test_memory_tree.py -v`

### Task 2: Dashboard Tree UI

**Files:**
- Modify: `api/static/index.html`

**Interfaces:**
- Consumes: `GET /v1/memory/tree`
- Produces: clickable tree view and selected virtual file detail panel

- [ ] **Step 1: Add DOM structure**

Add tabs, a tree browser section, and an empty/detail panel.

- [ ] **Step 2: Add rendering logic**

Fetch `/v1/memory/tree`, render folder/file nodes, and show grouped memory entries.

- [ ] **Step 3: Verify manually**

Run the app locally and open `/dashboard`.

### Task 3: Full Verification

**Files:**
- Modify if needed: `README.md`, `api/main.py`

**Interfaces:**
- Existing tests and live dashboard

- [ ] **Step 1: Run focused tests**

Run: `python -m unittest test_memory_tree.py -v`

- [ ] **Step 2: Run full tests**

Run the project scripts: `python test_core.py`, `python test_storage.py`, `python test_retrieval.py`, and `python test_api.py`

- [ ] **Step 3: Smoke-test dashboard**

Start the API locally with a temporary database and inspect `/dashboard`.
