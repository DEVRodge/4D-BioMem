# Memory Tree View Design

## Goal

Add a web-based memory tree viewer to 4D-BioMem so users can browse stored memories by hierarchy and click a virtual file to inspect the grouped memory content.

## Storage Reality

4D-BioMem does not store memories as Markdown files. The durable record is a SQLite row in `memory_cells`:

- `content`: original memory text
- `task_tags`: JSON text, deserialized to a Python dict
- `entities`: JSON text, deserialized to a Python list
- metadata fields such as `user_id`, `agent_id`, `is_risk`, `base_intensity`, `access_count`, timestamps, and weight

Vectors are stored separately in the configured vector backend. The web tree must be a read-only virtual grouping layer over these existing records.

## UX

The dashboard will keep the current monitoring view and add a tab-like switch:

- `记忆能量`: existing card grid and pruning monitor
- `记忆树`: new hierarchy browser

The tree hierarchy is:

`user_id / agent_id / project-or-general / virtual-file`

Virtual file names use `.mem` to avoid implying real Markdown storage. Clicking a virtual file displays the memory entries grouped under it, with original content and metadata.

## API

Add `GET /v1/memory/tree`.

Query parameters:

- `user_id` optional, same semantics as `/v1/monitor/cells`

Response:

- `count`: total memory count
- `tree`: nested folder/file nodes
- `items`: flat memory entries for the selected UI to reference

This endpoint is read-only and does not alter access count, timestamps, weights, vectors, or SQLite schema.
