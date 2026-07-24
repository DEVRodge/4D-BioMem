# Auto Maintenance v1.8 Design

## Goal

v1.8 lets 4D-BioMem整理每天的记忆：自动归档昨天及更早的 `memory_events`，然后刷新 Memory Wiki。它必须能在部署环境关机、重启、错过凌晨窗口或单次任务失败后自动补账。

## Behavior

- 自动维护默认开启：`AUTO_MAINTENANCE_ENABLED=true`。
- 默认维护时间：`03:30`。
- 默认时区：`Asia/Shanghai`。
- 默认周期补账扫描：每 `30` 分钟。
- 服务启动后会先执行一次补账扫描。
- 每天定时点会执行一次维护。
- 周期扫描会查漏补缺，修复错过凌晨、Docker 重启或临时失败造成的漏归档。

## Maintenance Scope

- 只处理“今天之前”的未归档事件，避免当天对话还在继续时提前打包。
- 归档单位是 `user_id + agent_id + date`。
- 每个归档组生成一条长期 `MemoryCell`，内容沿用现有 `[每日片段] YYYY-MM-DD user/agent 共 N 条：` 格式。
- 归档后标记对应事件为已归档并写入 `archive_cell_id`。
- 每轮维护结束后刷新 Memory Wiki。

## Safety

- 维护器使用进程内 `asyncio.Lock`，防止启动补账、周期扫描、手动触发并发运行。
- 已归档事件不会被重复归档。
- 当天事件默认跳过。
- 某个分组归档失败时记录错误，并继续处理其他分组。
- 维护失败不会阻断 FastAPI 服务启动和请求处理。

## API

- `GET /v1/maintenance/status`
  - 返回是否启用、维护时间、时区、周期扫描间隔、上次运行结果。
- `POST /v1/maintenance/run_once`
  - 手动触发一次维护。
  - 默认也只处理今天之前。
  - 可选 `include_today=true` 用于测试或人工强制整理当天片段。

## Dashboard

- 在 Dashboard 增加自动整理状态区域。
- 显示启用状态、上次运行时间、上次归档组数、事件数、Wiki 页数。
- 提供“一键整理并刷新 Wiki”按钮。

## Non-Goals

- v1.8 不引入外部 cron、Celery、Redis 或数据库迁移服务。
- v1.8 不自动剪枝 `memory_cells`；它只做每日片段归档和 Wiki 刷新。
- v1.8 不改变 SQLite + 向量库作为主存储的架构。

## Testing

- 单元测试未归档事件分组发现逻辑。
- API 测试手动维护只归档昨天及更早的事件，并跳过当天事件。
- API 测试维护后自动刷新 Wiki。
- 回归测试 v1.6 事件归档、v1.7 Wiki、记忆树和 API 主链路。
