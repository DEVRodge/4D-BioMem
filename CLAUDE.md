# 4D-BioMem 项目规范与长效记忆

## 长效记忆系统 (Claude Code × 4D-BioMem)

本机运行着 4D-BioMem 长效记忆服务 (`http://localhost:8000`),所有用户记忆、项目进展、架构决策都存储在其中。
你（Claude Code）可以通过 curl 直接读写这些记忆,**实现跨对话的持续记忆**——下次回来你不会忘记之前做过什么、计划了什么。

这是目前项目已知的一些关键信息，你可以作为参考：

### 项目已完成的里程碑

| 里程碑 | 内容 | 对应文件 |
|--------|------|---------|
| M1 - 核心剪枝算法 | MemoryCell + SynapticPruningEngine | `core/memory_cell.py` |
| M2 - 双轨存储层 | DBManager (SQLite + 向量库, Dual-Write/Delete) | `storage/db_manager.py` |
| M3 - 双通路检索 | DualPathwayRetriever (硬/软检索 + τ 激活阈值) | `core/retrieval.py` |
| M4 - Hermes 工具 + 评测 | 30 轮科学评测 + A/B/C 三组对照 | `integrations/hermes_tools.py`, `run_benchmark.py`, `experiment/` |
| M5 - 可视化看板 | 暗黑科技风 Dashboard + ECharts | `api/static/index.html` |
| 生产化 | Docker Compose + OpenAI 可选 + API Key 鉴权 | `Dockerfile`, `docker-compose.yml` |

### 会话开始前——读取记忆

每当开始新对话时,先检索 4D-BioMem 中关于项目进展的记忆,了解上下文：

```bash
curl -s -X POST http://localhost:8000/v1/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_id":"hermes","query":"4D-BioMem 项目进展 里程碑 当前任务","top_k":10}'
```

同时读取用户的个人记忆（偏好、约定等）：

```bash
curl -s -X POST http://localhost:8000/v1/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_id":"hermes","query":"用户偏好 约定 开发方向","top_k":5}'
```

### 对话过程中——主动存入记忆

当以下情况发生时,帮我存入 4D-BioMem：

1. **完成了一个里程碑或阶段性任务** → 记下完成的内容和状态
2. **做出了关键架构决策** → 记下决策内容和原因
3. **用户表达了对某事的偏好/要求** → 记下用户的偏好
4. **制定了下一步计划** → 记下计划内容
5. **修复了重要 bug** → 记下 bug 原因和修复方式

```bash
# 统一用 hermes 用户,前缀标注信息类型
curl -s -X POST http://localhost:8000/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"user_id":"hermes","content":"[项目进展] 这里写上里程碑/任务完成情况和状态"}'
```

### 信息类型前缀规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `[项目进展]` | 里程碑/任务完成 | `[项目进展] M3 双通路检索已完成，5 场景全绿` |
| `[架构决策]` | 技术选型/设计变更 | `[架构决策] 默认向量后端切换为 numpy 保持轻量` |
| `[用户偏好]` | 用户明确表达的喜好 | `[用户偏好] 偏好 docker 部署，不配 OpenAI 用 mock` |
| `[迭代计划]` | 约定好的下一步计划 | `[迭代计划] 下一步做对照组实验框架` |
| `[Bug修复]` | 修复的重要 bug | `[Bug修复] retrieve 融合排序的尺度不匹配导致闲聊被错误强化` |
| `[对话约定]` | 当前对话中达成的共识 | `[对话约定] 用户确认使用 4D-BioMem 替代内置 memory` |
