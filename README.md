# 4D-BioMem · 智能体长效记忆系统

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

**4D-BioMem** 是一个受生物突触机制启发的 Agent 长效记忆系统。它模拟人脑的"新陈代谢"——记忆有强弱之分，高频使用的记忆被强化，低频噪声被物理抹除，安全底线永久锁定。

与传统的 Vanilla RAG（纯向量检索）或 Mem0/MemGPT 等摘要压缩方案不同，4D-BioMem 在**四维坐标空间**（时间 T / 任务特征 F / 风险规则 R / 语义向量 V）中管理记忆，通过**突触权重衰减**和**双通路唤醒**实现类人记忆的"自然遗忘"。

---
<img width="1909" height="1009" alt="image" src="https://github.com/user-attachments/assets/1c7b0127-7738-4db1-9780-3e5142e20941" />
可视化web控制台

## ✨ 核心特性

### 四维记忆空间 M=(T,F,R,V)

| 维度 | 含义 | 作用 |
|------|------|------|
| **T** - Timeline | 时间线索 | 记忆热度随时间自然衰减，太久没唤醒就变弱 |
| **F** - Feature | 动态任务标签 | LLM 提取的半结构化标签（项目名、类型），用于硬过滤 |
| **R** - Risk & Rule | 风险/生存本能 | **永久锁定**——如过敏史、密码等，豁免剪枝 |
| **V** - Vector Space | 高维语义向量 | 在极小候选集内做相似度软匹配，根本消除噪音 |

### 突触剪枝遗忘机制（Synaptic Pruning）

权重衰减公式：

$$W_i(t) = \begin{cases} \infty & \text{if } R_i = 1 \\ I_i \cdot \ln(1 + C_i) \cdot e^{-\lambda \cdot \Delta t_i} & \text{if } R_i = 0 \end{cases}$$

| 符号 | 含义 |
|------|------|
| $R_i$ | 风险标记（1=永久锁定，0=动态遗忘） |
| $I_i$ | LLM 评估的初始显性强度 [1, 10] |
| $C_i$ | 被检索唤醒的总次数 |
| $\lambda$ | 遗忘衰减因子 |
| $\Delta t_i$ | 距离上次唤醒的时间 |

当 $W_i(t) < \theta_{\text{prune}}$ 时，该记忆被**物理不可逆抹除**。

### 双通路记忆唤醒架构

| 通路 | 触发 | 耗时 | 方式 |
|------|------|------|------|
| **潜意识反射链** | 被动触发，毫秒级 | 低 | 时间流 + 风险规则硬匹配，风险记忆强制常驻 Prompt |
| **显意识搜索链** | 面对复杂任务，主动调用 | 高 | 高维语义向量相似度 × 权重加成，Top-K 召回 |

**激活阈值 τ**：`hard_confidence < τ` 时才升级到软检索，定量刻画"潜意识够用则不调用显意识"的算力调配边界。

---

## 📊 实验数据（对照组三组横向对比）

50 条标注语料（风险=6 / 技术=17 / 闲聊=27），地面真值查询 5 条，剪枝阈值 θ=0.5：

| 指标 | A: Vanilla RAG | B: FIFO+摘要 | C: **4D-BioMem** |
|------|:---:|:---:|:---:|
| **高危召回率 Risk Recall** | 100.0% | 100.0% | **100.0%** |
| **语义检索 Precision@K** | 100.0% | 100.0% | **40.0%** * |
| **上下文噪声比 Noise=闲聊** | 20.0% | 20.0% | **0.0%** |
| **存储收敛**  | 50→50 (线性) | 50→50 (有界) | **50→23 (收敛)** |
| **累计物理抹除** | 0 | 0 | **27 条闲聊** |

> \* Precision@K 40% 是因为双通路检索的融合打分在多相似记忆时推挤了精确匹配项，是当前设计的一个可优化点。

**存储收敛曲线**——C 组在 30 天模拟衰减后，全部 27 条闲聊被物理抹除，库从 50 急剧收敛到 23：

```
A (RAG)     : 10 -> 20 -> 30 -> 40 -> 50 -> 50   (线性增长，无回落)
B (SW+摘要)  : 10 -> 20 -> 30 -> 40 -> 50 -> 50   (有界于 N=100)
C (BioMem)  : 10 -> 20 -> 30 -> 30 -> 50 -> 23   (剪枝后收敛)
```

**30 轮加速衰减评测**（30 天模拟，casual I=2 → 0.31 < 0.5，tech I=7 → 1.08 > 0.5）：

| 指标 | 结果 |
|------|:----:|
| 高危召回率 (Risk Recall) | **100%** |
| 上下文噪声抑制比 | **18/18 = 100%** |
| 存储空间收敛度 (Pickle) | 27555B → 15201B, **缩减 44.8%** |
| 技术记忆存活率 | **10/10 = 100%** |

---

## 🏗️ 系统架构

```
                    ┌─────────────────────────────────────────┐
                    │          Hermes Agent / 客户端           │
                    │   integrations/hermes_tools.py           │
                    │   remember_fact / recall_memory          │
                    └────────────────┬────────────────────────┘
                                     │ HTTP (httpx)
                    ┌────────────────▼────────────────────────┐
                    │         API 层 (FastAPI)                 │
                    │   POST /v1/memory/add  (异步录入)        │
                    │   POST /v1/memory/retrieve (双通路检索)   │
                    │   POST /v1/memory/prune (新陈代谢)        │
                    │   GET  /v1/monitor/cells                 │
                    │   GET  /dashboard/  (可视化看板)          │
                    └────────────────┬────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
   ┌──────▼──────┐          ┌───────▼───────┐          ┌───────▼───────┐
   │  算法层      │          │  存储层        │          │  LLM 审计     │
   │ memory_cell │          │  DBManager    │          │  OpenAI/Mock  │
   │ Synaptic-   │          │  SQLite 元数据 │          │  Embedding    │
   │ Pruning     │          │  向量库(pkl)   │          │  风险检测     │
   │ Dual-       │          │  Dual-Write   │          │  标签提取     │
   │ Pathway     │          │  Dual-Delete  │          │  强度评分     │
   │ Retrieve    │          │               │          │               │
   └─────────────┘          └───────────────┘          └───────────────┘
```

---

## 🔌 Embedding API 说明

4D-BioMem 的**通路段 B（显意识语义检索）**依赖 Embedding API 将文本转换为高维向量，从而实现语义相似度匹配。

### 默认模式（零配置运行）

项目默认使用 **Mock 嵌入器**——基于 4-gram 哈希的确定性伪向量生成器，**无需任何 API Key、不依赖外部服务、不产生任何费用**。所有功能（包括检索、剪枝、看板）在 mock 模式下完整可用。

```bash
# 只需启动服务，无需任何额外配置
docker compose up -d
```

> 但 mock 模式只能识别**关键词级别的匹配**（"项目 Alpha 的 Bug"和"项目 Alpha 的 Bug 修复方案"可以匹配），
> 无法理解语义同义关系（"部署架构"和"k8s 加双活"无法自动关联）。

### 开启真实 Embedding 后获得的能力

| 能力 | Mock 模式 | 真实 Embedding |
|------|:--------:|:-------------:|
| 关键词精确匹配 | ✅ | ✅ |
| 语义同义关联（"车"→"汽车"） | ❌ | ✅ |
| 跨语言语义匹配 | ❌ | ✅ |
| 长文本主题匹配 | ❌ | ✅ |
| 搜索"部署架构"召回"k8s 双活" | ❌ | ✅ |
| 外部依赖 | 零 | 需 API Key |

### 配置方式

```bash
# 1. 填入 Embedding API Key（以 OpenAI 为例）
OPENAI_API_KEY=sk-your-key-here

# 2. 启动后自动生效，Mock 模式自动降级为后备
```

### Embedding API 厂家推荐

| 厂家 | 模型 | 维度 | 国内可用 | 价格 | 语言支持 | 推荐场景 |
|------|------|:---:|:------:|:---:|:-------:|--------|
| **OpenAI** | `text-embedding-3-small` | 1536 | 需代理 | ~$0.02/1M tokens | 英文最佳，中文良好 | **首选，综合质量最高** |
| **OpenAI** | `text-embedding-3-large` | 3072 | 需代理 | ~$0.13/1M tokens | 同上 | 高精度场景 |
| **阿里通义千问** | `text-embedding-v2` | 1536 | ✅ | 百万 token 约 0.5元 | 中文最佳 | **国内用户首选，价格低** |
| **百度文心** | `ERNIE-Bot-Embedding` | 384 | ✅ | 免费额度 50万t/月 | 中文最佳 | 百度云用户 |
| **智谱 GLM** | `embedding-2` | 1024 | ✅ | 百万 token 约 1元 | 中文优秀 | 智谱生态用户 |
| **硅基流动** | `BAAI/bge-m3` | 1024 | ✅ | 免费 | 多语言 | **免费首选，零成本接入** |
| **Ollama 本地** | `nomic-embed-text` | 768 | ✅ | 免费 | 多语言良好 | **完全本地、零网络、保护隐私** |

### 配置示例

**OpenAI（首选，英文/通用场景）：**
```bash
LLM_BACKEND=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**阿里通义千问（国内首选，中文场景）：**
```bash
LLM_BACKEND=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx          # 阿里 DashScope API Key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v2
```

**Ollama 本地部署（零网络、隐私保护）：**
```bash
# 先安装 Ollama 并拉取模型
# ollama pull nomic-embed-text

# 再启动 4D-BioMem（mock 模式下 Embedding 自动走本地 4-gram hash）
# Ollama 集成将在后续版本提供原生支持
```

> **提示**：Embedding API 仅影响**通路 B（语义软检索）**的质量。通路段 A（风险常驻 + 任务/时间硬匹配）和剪枝算法完全在本地运行，不受 Embedding 影响。即使 Embedding API 不可用或降级，系统仍然完整可用。

---

## 🚀 快速开始

### 方式一：Docker Compose 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/your-username/4D-BioMem.git
cd 4D-BioMem

# 2. 配置（可选：设一个 API Key 保护服务）
cp .env.example .env
# 编辑 .env 设置 API_KEY，留空则不启用鉴权

# 3. 启动
docker compose up -d

# 4. 验证
curl http://localhost:8000/health
```

### 方式二：直接运行

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 方式三：局域网二机部署

**服务器（运行 4D-BioMem 服务）：**

```bash
cp .env.example .env
# 编辑 .env: API_KEY=my-secret
docker compose up -d
```

**Hermes 客户端（调用记忆工具）：**

```bash
# 从服务器复制工具文件
scp user@server:/path/4D-BioMem/integrations/hermes_tools.py .
scp user@server:/path/4D-BioMem/integrations/__init__.py .

pip install httpx
export BIOMEM_API_URL=http://server-ip:8000
export BIOMEM_API_KEY=my-secret
```

在 Python 中使用：

```python
from integrations import configure, remember_fact, recall_memory

configure(base_url="http://192.168.1.100:8000", api_key="my-secret")

# 存入记忆
remember_fact("用户说他青霉素过敏，开药要避开")
# → {"status": "queued", ...}

# 检索记忆
recall_memory("用户有什么过敏史")
# → {"hits": [{"content": "用户说他青霉素过敏...", "is_risk": True, ...}], ...}
```

---

## 🔧 配置参考

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LLM_BACKEND` | `mock` | `mock` 零依赖模式 / `openai` 使用 OpenAI 兼容 API |
| `OPENAI_API_KEY` | 空 | OpenAI / 兼容 API Key（不填则自动用 mock） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 地址（阿里云/硅基流动等填对应地址） |
| `OPENAI_MODEL` | `gpt-4o-mini` | 审计用大模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型名 |
| `API_KEY` | 空 | 服务鉴权 Key（空=不鉴权，适合内网） |
| `DB_PATH` | `/data/biomem.db` | SQLite 数据库路径 |
| `VECTOR_PATH` | `/data/vector_store` | 向量存储路径 |
| `LAMBDA` | `0.05` | 遗忘衰减因子 |
| `THETA_PRUNE` | `0.5` | 剪枝权重阈值 |
| `TAU` | `1.0` | 软检索激活阈值 |
| `SEED` | `true` | 启动时是否灌入 8 条演示数据 |
| `LOG_LEVEL` | `info` | 日志级别 |
| | **客户端环境变量** | |
| `BIOMEM_API_URL` | `http://localhost:8000` | 4D-BioMem 服务地址（客户端工具用） |
| `BIOMEM_API_KEY` | 空 | API 鉴权 Key（与服务器 API_KEY 一致） |
| `BIOMEM_DEFAULT_USER` | `hermes` | 默认用户 ID |

---

## 🤖 Hermes Agent 集成

4D-BioMem 已原生支持 [Hermes Agent](https://github.com/DEVRodge/Hermes-Agent-Self-Evolution) 框架。安装方式：

```bash
# 在 Hermes Agent 的 tools 目录下已有 biomem_tool.py
# 只需确保 4D-BioMem 服务在运行
curl http://localhost:8000/health
# → {"status":"ok", ...}
```

### 已注册的工具

Hermes Agent 启动后，大模型自动可使用两个 4D-BioMem 工具：

| 工具名称 | 功能 | 大模型何时调用 |
|---------|------|-------------|
| `biomem_remember` | 存入事实到长效记忆 | 用户说了需要跨会话记住的信息：过敏史、密码、偏好、项目方案等 |
| `biomem_recall` | 检索历史记忆 | 用户引用过去讨论、需要检查过敏/禁忌、上下文窗口不够时 |

### 记忆代谢闭环

```
用户说 "我对青霉素过敏" 
  → Hermes 调 biomem_remember 
    → 4D-BioMem 审计标记为 is_risk=True（永久锁定）
      → 双通路检索时风险记忆始终强制返回 ✓

用户说 "上次那个 Bug 怎么修"
  → Hermes 调 biomem_recall
    → 4D-BioMem 双通路唤醒 → 权重排序 Top-K → 返回结果
      → 命中记忆 C_i+=1，突触强化，免于剪枝 ✓

闲聊内容（"今天吃了酸菜鱼"）
  → 低价值标记 is_risk=False, I=2
    → 无人检索 → 权重衰减至 θ_prune 以下 → 物理抹除 ✓
```

### 使用方式

**方式一：直接在 Hermes 对话中告知**
```
你对 Hermes 说：
  → "从现在开始请用 biomem_remember 和 biomem_recall 工具管理我的长期记忆"
之后 Hermes 的大模型就会自动判断何时存、何时查。
```

**方式二：环境变量配置**

```bash
export BIOMEM_API_URL=http://localhost:8000
# 如果 4D-BioMem 开启了 API 鉴权（未配置则无需设此值）
export BIOMEM_API_KEY=your-key
```

### 验证连通

```bash
# 说一句话让 Hermes 存，然后查 4D-BioMem：
curl -s "http://localhost:8000/v1/memory/list" -G -d user_id=hermes | python3 -m json.tool
```

---

## 📡 API 接口

所有 `/v1/` 接口受 `X-API-Key` 头部保护（若配置了 `API_KEY`）。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（免鉴权） |
| `POST` | `/v1/memory/add` | 异步录入记忆（立即返回 queued） |
| `GET` | `/v1/memory/list` | 列出用户全部记忆 |
| `POST` | `/v1/memory/retrieve` | 双通路唤醒检索 |
| `POST` | `/v1/memory/prune` | 触发新陈代谢——抹除死亡记忆 |
| `GET` | `/v1/monitor/cells` | 全量细胞实时监控 |
| `POST` | `/v1/monitor/system_status` | 系统整体指标 |
| `GET` | `/dashboard/` | 可视化监控面板 |

---

## 🧪 测试

所有里程碑回归测试：

```bash
# M1 - 剪枝算法
python3 test_core.py

# M2 - 存储双后端
python3 test_storage.py

# M3 - 双通路检索
python3 test_retrieval.py

# API - 端到端
python3 test_api.py

# 科学评测 - 30 轮
python3 run_benchmark.py

# 对照组横向对比
python3 experiment/runner.py
```

---

## 🖥️ 可视化监控面板

启动服务后访问 `http://localhost:8000/dashboard/`：

- **实时权重列表**——每条记忆的权重、频次、距上次唤醒时间
- **一键新陈代谢**——加速衰减，看闲聊记忆"灰飞烟灭"动画
- **ECharts 实时图表**——记忆构成饼图 + 权重分布对数柱状图
- **系统指标卡片**——有效记忆数、风险锁定率、今日剪枝数

---

## 📁 项目结构

```
4D-BioMem/
├── config.py                   配置管理
├── core/
│   ├── memory_cell.py          MemoryCell + SynapticPruningEngine
│   ├── retrieval.py             DualPathwayRetriever（双通路检索）
│   └── llm_auditor.py           OpenAILLMAuditor + Mock（自动降级）
├── storage/
│   └── db_manager.py            DBManager（SQLite + 向量库，线程安全）
├── api/
│   ├── main.py                  FastAPI 服务（6 路由 + 鉴权）
│   └── static/index.html        暗黑科技风监控看板
├── integrations/
│   └── hermes_tools.py          Hermes Agent 独立工具（httpx 直连）
├── experiment/                   A/B/C 三组对照组实验框架
├── test_core.py / test_storage.py / test_retrieval.py
├── test_api.py / run_benchmark.py
├── requirements.txt             依赖清单
├── Dockerfile / docker-compose.yml / .env.example
└── SPEC.md                      技术规格说明书

# Hermes Agent 集成文件（安装在 ~/.hermes/hermes-agent/tools/）
~/.hermes/hermes-agent/
├── tools/
│   └── biomem_tool.py           4D-BioMem 工具（registry.register 注册）
└── toolsets.py                  已添加 biomem_remember/recall 到核心工具集
```

---

## 依赖

**仅 5 个必要依赖**（零外部 AI 依赖即可运行）：

```
fastapi>=0.138.0
uvicorn>=0.49.0
numpy>=1.24.0
openai>=1.0.0       # 可选（仅 LLM_BACKEND=openai 时需要）
httpx>=0.28.0       # 仅客户端工具需要
```

---

## 📜 许可证

MIT

---

## 🙏 致谢

本项目受生物突触可塑性（Synaptic Plasticity）和记忆衰退理论启发，旨在为大语言模型 Agent 提供一个更接近人脑记忆机制的长效记忆解决方案。
