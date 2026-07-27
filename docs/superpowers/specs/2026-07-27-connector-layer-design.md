# Connector Layer v1.9 Design

## Goal

v1.9 adds a connector ingestion layer so external sources can feed standardized fragments into `memory_events`. This turns v1.6-v1.8 into a pipeline: connectors import raw fragments, daily maintenance archives them, and Memory Wiki exposes the readable state.

## Scope

- Add connector registry and connector run history to SQLite.
- Extend `memory_events` with source metadata: `source`, `source_id`, and `connector_id`.
- Enforce idempotency with `connector_id + source + source_id`.
- Add API endpoints to register/list/run connectors and ingest standardized fragments.
- Implement first connectors:
  - `filesystem`: import `.md`, `.txt`, and `.jsonl` files from a local directory.
  - `git`: import commit log entries from a local git repository.
  - `hermes_manual`: accept direct push fragments from Hermes or manual tools.

## Data Model

`memory_connectors`:

- `id`
- `name`
- `connector_type`
- `config` JSON
- `enabled`
- `created_at`
- `updated_at`

`connector_runs`:

- `id`
- `connector_id`
- `status`
- `started_at`
- `finished_at`
- `imported_count`
- `skipped_count`
- `error`
- `details` JSON

`memory_events` new metadata:

- `source`
- `source_id`
- `connector_id`

## API

- `POST /v1/connectors/register`
- `GET /v1/connectors`
- `POST /v1/connectors/run`
- `GET /v1/connectors/runs`
- `POST /v1/connectors/ingest`

## Connector Configs

`filesystem` config:

```json
{
  "path": "/data/inbox",
  "user_id": "hermes",
  "agent_id": "filesystem",
  "project": "4D-BioMem",
  "max_files": 200,
  "max_items": 500,
  "max_file_bytes": 262144
}
```

`git` config:

```json
{
  "repo_path": "/app",
  "user_id": "hermes",
  "agent_id": "git",
  "project": "4D-BioMem",
  "max_commits": 50
}
```

`hermes_manual` uses `POST /v1/connectors/ingest` and does not require scanning. Direct ingest requires the connector to be registered and enabled.

## Safety

- Connectors write only to `memory_events`; they never directly write long-term `memory_cells`.
- Duplicate external items are skipped, not duplicated.
- Connector identity fields are stripped and must be non-empty.
- Filesystem default `source_id` uses root-relative paths plus content hash instead of absolute mount paths.
- Connector failures create a failed `connector_runs` row and do not interrupt the API process.
- v1.9 does not add OAuth connectors.

## Testing

- DB tests for connector registration, run logging, event metadata, and idempotent source identity.
- API tests for direct connector ingest idempotency.
- API tests for filesystem and git connector runs.
- Regression tests for events, maintenance, wiki, and API core flow.
