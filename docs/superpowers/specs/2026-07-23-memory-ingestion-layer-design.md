# Memory Ingestion Layer Design

## Goal

Add the missing front half of 4D-BioMem: daily memory fragment ingestion before long-term memory pruning. Hermes should be able to write many small events over time, then archive a day of fragments into durable long-term `MemoryCell` records.

## Current Gap

The current system stores long-term memories only when Hermes or a caller invokes `remember_fact` / `POST /v1/memory/add`. That means the pruning engine works on already-curated memory cards, but the system does not continuously collect daily interaction fragments.

## v1.6 Scope

Add a lightweight `memory_events` layer:

- `memory_events` stores raw or semi-raw daily fragments.
- `POST /v1/memory/ingest_event` appends a fragment.
- `GET /v1/memory/events` lists fragments by user, date, and archive status.
- `POST /v1/memory/archive_day` aggregates one day of unarchived fragments into one long-term `MemoryCell`.
- Archived fragments keep their original text and point at the generated long-term memory cell.

This version deliberately does not add an automatic scheduler. Manual or Hermes-triggered archiving is safer for the current Docker-backed local memory system.

## Data Model

`memory_events` fields:

- `id`: event id
- `user_id`: owner
- `agent_id`: source agent
- `content`: original fragment
- `event_type`: conversation, task, observation, decision, or generic
- `task_tags`: JSON dict
- `created_at`: server insertion time
- `occurred_at`: event time, defaults to insertion time
- `archived`: 0 / 1
- `archive_cell_id`: generated long-term memory id after archiving

## Archive Behavior

`archive_day` selects unarchived events matching `user_id`, optional `agent_id`, and a calendar date derived from `occurred_at`. It creates deterministic archive content:

`[每日片段] YYYY-MM-DD <user>/<agent> 共 N 条：`

followed by short bullet lines from the original events. The archive content is audited, embedded, and saved through the existing `DBManager.save_memory`, so it joins the normal long-term memory pool and can be retrieved, strengthened, and pruned.

## Dashboard

The dashboard keeps the current long-term memory tree and adds a daily fragment view. Users can see:

- long-term memory groups
- daily fragment groups by date
- whether fragments are archived
- which long-term memory cell an archived fragment belongs to

## Safety

The migration is additive: it creates a new table and does not alter existing `memory_cells` rows or vectors. Archiving is explicit and only touches selected unarchived events plus the newly created long-term memory.
