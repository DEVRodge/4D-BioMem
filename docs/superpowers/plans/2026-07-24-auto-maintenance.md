# Auto Maintenance v1.8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-healing daily maintenance that archives older event fragments and refreshes Memory Wiki automatically.

**Architecture:** Extend `DBManager` with unarchived event group discovery. Add maintenance orchestration to `AppState`, using an `asyncio.Lock`, startup catch-up, daily scheduling, periodic catch-up scans, and manual/status API endpoints. Dashboard consumes those endpoints for visibility and manual triggering.

**Tech Stack:** Python 3.10+, FastAPI, SQLite, standard-library `zoneinfo`, vanilla dashboard JavaScript, `unittest`.

## Global Constraints

- `AUTO_MAINTENANCE_ENABLED` defaults to `true`.
- Maintenance only auto-archives dates before the current local date unless `include_today=true`.
- Maintenance never deletes or prunes `memory_cells`.
- Maintenance refreshes Memory Wiki after each run.
- Tests must explicitly disable automatic background maintenance when validating old manual archive behavior.

---

### Task 1: Event Group Discovery

**Files:**
- Modify: `storage/db_manager.py`
- Test: `test_maintenance.py`

**Interfaces:**
- Produces: `DBManager.list_unarchived_event_groups(before_date: str | None = None, include_today: bool = False, today: str | None = None) -> list[dict[str, str | int]]`

- [x] **Step 1: Write failing test**

Run: `python -m unittest test_maintenance.py -v`
Expected: FAIL because `list_unarchived_event_groups` is missing.

- [x] **Step 2: Implement minimal query**

Use SQLite grouping over `substr(occurred_at, 1, 10)`, returning `user_id`, `agent_id`, `date`, and `event_count`.

- [x] **Step 3: Verify**

Run: `python -m unittest test_maintenance.py -v`
Expected: PASS for group discovery.

### Task 2: Maintenance Runner

**Files:**
- Modify: `api/main.py`
- Test: `test_maintenance.py`

**Interfaces:**
- Produces: `AppState.run_maintenance_once(trigger: str = "manual", include_today: bool = False) -> dict`
- Produces: `AppState.maintenance_status(now: datetime | None = None) -> dict`

- [x] **Step 1: Write failing API test**

Run: `python -m unittest test_maintenance.py -v`
Expected: FAIL because `/v1/maintenance/run_once` does not exist.

- [x] **Step 2: Implement maintenance orchestration**

Move archive-day internals into a reusable helper, run group archives under `asyncio.Lock`, refresh Wiki, and record the last result.

- [x] **Step 3: Verify**

Run: `python -m unittest test_maintenance.py -v`
Expected: PASS.

### Task 3: Background Scheduler

**Files:**
- Modify: `config.py`
- Modify: `api/main.py`
- Test: `test_maintenance.py`

**Interfaces:**
- Produces settings: `auto_maintenance_enabled`, `maintenance_time`, `maintenance_timezone`, `maintenance_interval_minutes`.
- Produces `AppState.start_maintenance()` and maintenance loop shutdown support.

- [x] **Step 1: Write failing scheduler config/status test**

Run: `python -m unittest test_maintenance.py -v`
Expected: FAIL because status lacks scheduler fields.

- [x] **Step 2: Implement settings and loop**

Start maintenance in lifespan when enabled. Run startup catch-up, then sleep until the earlier of next daily run or periodic scan.

- [x] **Step 3: Verify**

Run: `python -m unittest test_maintenance.py -v`
Expected: PASS.

### Task 4: Dashboard and Docs

**Files:**
- Modify: `api/static/index.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: `GET /v1/maintenance/status`, `POST /v1/maintenance/run_once`.

- [x] **Step 1: Add Dashboard maintenance panel**

Show enabled status, last run summary, and a manual run button.

- [x] **Step 2: Update README**

Add v1.8 API/config/docs/changelog in Chinese.

- [x] **Step 3: Verify**

Run:

```bash
python -m unittest test_maintenance.py test_memory_events.py test_memory_wiki.py test_memory_tree.py -v
python -m py_compile api/main.py storage/db_manager.py core/memory_wiki.py config.py
node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('api/static/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
for (const script of scripts) new Function(script);
console.log(`checked ${scripts.length} inline scripts`);
NODE
```

Expected: all commands pass.
