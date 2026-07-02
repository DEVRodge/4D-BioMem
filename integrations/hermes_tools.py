"""Hermes Agent × 4D-BioMem 工具集。

向 Hermes Agent 的大模型暴露三个长效记忆工具，封装本地 FastAPI 服务的
HTTP 细节，让 LLM 只需关心"何时存、何时查、何时合成"：

  - remember_fact(content)  : 存入长效记忆（异步审计 + 风险锁定 + 实体提取 + 落盘）
  - recall_memory(query)     : 双通路检索历史记忆（硬反射 + 软语义 + 实体 boost）
  - synthesize_memory(...)   : 跨记忆合成问答

通信：httpx 同步客户端，默认 http://127.0.0.1:8000（可用 configure() 覆盖）。
LangChain 适配：函数签名 + docstring 已对齐 @tool 规范，可直接包裹：
    from langchain_core.tools import tool
    remember_fact_tool = tool(remember_fact)
    recall_memory_tool = tool(recall_memory)
    synthesize_memory_tool = tool(synthesize_memory)
随后注入 AgentExecutor / Hermes 工具列表即可。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = os.environ.get("BIOMEM_API_URL", "http://127.0.0.1:8000")
DEFAULT_USER = os.environ.get("BIOMEM_DEFAULT_USER", "hermes")
API_KEY = os.environ.get("BIOMEM_API_KEY", "")
TIMEOUT = 10.0


def configure(base_url: str | None = None, default_user: str | None = None,
              api_key: str | None = None) -> None:
    """运行时覆盖服务地址 / 默认用户 / API Key（测试或联调时用）。"""
    global BASE_URL, DEFAULT_USER, API_KEY
    if base_url:
        BASE_URL = base_url
    if default_user:
        DEFAULT_USER = default_user
    if api_key is not None:
        API_KEY = api_key


def _headers() -> dict[str, str]:
    """返回含 API Key 的请求头（未配置则空）。"""
    hdrs: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        hdrs["X-API-Key"] = API_KEY
    return hdrs


def remember_fact(content: str, user_id: str | None = None,
                  task_tags: dict | None = None) -> dict[str, Any]:
    """Store a piece of information into 4D-BioMem long-term memory.

    何时该调用（CALL THIS WHEN）:
      - 用户陈述了希望跨会话记住的事实：偏好、计划、项目细节、技术决策、
        约定、待办。
      - 用户透露安全 / 隐私 / 健康信息：过敏史、用药禁忌、密码、凭证、
        病史、机密。系统会自动识别并打上 is_risk=True 风险锁定标签，
        永不剪枝，未来检索时强制常驻上下文（生存本能轨）。
      - 任何"如果忘了会出错"的信息。

    何时不该调用（DO NOT CALL WHEN）:
      - 当前轮次的临时推理、寒暄、对用户即时问题的直接作答。
      - 用户只是闲聊天气 / 饮食 / 心情等无关紧要的琐事——这些会被系统
        当作低权重噪声，在剪枝时物理抹除。
      - 信息已在当前上下文窗口内，无需持久化。

    行为：异步写入——立即返回 {"status":"queued"}，后台 LLM 审计 +
    实体提取 + 向量化 + 落盘，绝不阻塞对话。同一内容重复存入会生成新记忆条目。

    Args:
        content: 要记忆的原始文本（中文 / 英文均可，建议完整句子）。
        user_id: 用户标识，默认 "hermes"。不同用户记忆隔离。
        task_tags: 可选，显式传入标签 dict（如 {"project": "Alpha", "type": "tech"}），
                   与 LLM 自动提取的标签合并。不传则完全由 LLM 自动提取。

    Returns:
        成功: {"status": "queued", "message": "...", "request_id": "..."}
        失败: {"error": "...", "status": "failed"}
    """
    uid = user_id or DEFAULT_USER
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as cli:
            body = {"user_id": uid, "content": content}
            if task_tags:
                body["task_tags"] = task_tags
            r = cli.post("/v1/memory/add", json=body, headers=_headers())
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001 — 工具失败不能击穿 Agent 主循环
        return {"error": f"remember_fact 失败: {exc}", "status": "failed"}


def recall_memory(query: str, user_id: str | None = None, top_k: int = 5,
                  query_entities: list[dict] | None = None,
                  agent_id: str | None = None) -> dict[str, Any]:
    """Retrieve relevant memories from 4D-BioMem via dual-pathway awakening.

    何时该调用（CALL THIS WHEN）:
      - 用户引用过去的讨论："上次说的那个 Bug"、"我之前提过的项目"、
        "上周的决定"。
      - 需要核对用户偏好 / 过敏 / 禁忌后再行动（如推荐药物、饮食、方案）。
      - 回答需要早期对话的上下文，而当前上下文窗口已不够。
      - 任何"我记得好像有过这么回事"的情境。

    何时不该调用（DO NOT CALL WHEN）:
      - 答案完全在当前上下文窗口内。
      - 用户问的是通用知识（无需个人记忆）。
      - 刚刚 remember_fact 存入、当前轮次已可见的信息。

    行为：双通路检索——
      通路 A（潜意识反射，毫秒级）：最近 5 条 + 所有风险记忆（强制常驻）。
      通路 B（显意识回忆，软匹配）：query 向量与记忆向量余弦相似度检索。
      若提供 query_entities，通路 B 按实体重叠做 score boost。
      若提供 agent_id，只检索该 agent 的记忆。
      去重融合后返回 top_k 条，并自动强化被回忆记忆的突触权重
      （access_count += 1，刷新时间戳，使其免于被过快剪枝）。

    Args:
        query: 检索查询文本（自然语言，描述想回忆什么）。
        user_id: 用户标识，默认 "hermes"。
        top_k: 返回条数上限，默认 5。风险记忆总是置顶返回。
        query_entities: 可选实体列表，如 [{"name": "Alpha", "type": "project"}]，
                       通路 B 命中实体时做 score boost。
        agent_id: 可选，指定只检索该 agent 的记忆。

    Returns:
        成功: {"user_id":..., "query":..., "hits":[...], "pathways":{"A":n,"B":m}}
              每个 hit 含 id / content / agent_id / is_risk / score / pathways /
              task_tags / entities。
        失败: {"error": "...", "status": "failed"}
    """
    uid = user_id or DEFAULT_USER
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as cli:
            body = {"user_id": uid, "query": query, "top_k": top_k}
            if query_entities:
                body["query_entities"] = query_entities
            if agent_id:
                body["agent_id"] = agent_id
            r = cli.post("/v1/memory/retrieve", json=body, headers=_headers())
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"recall_memory 失败: {exc}", "status": "failed"}


def synthesize_memory(question: str, user_id: str | None = None, top_k: int = 5,
                      query_tags: dict | None = None,
                      query_entities: list[dict] | None = None,
                      agent_id: str | None = None) -> dict[str, Any]:
    """Synthesize an answer across multiple memories.

    何时该调用（CALL THIS WHEN）:
      - 用户的问题需要组合多条记忆才能回答（如"关于项目 Alpha 我们有哪些信息？"）
      - 需要跨记忆总结而非单条检索（如"用户对我们这个产品有什么反馈？"）
      - 答案需要"连接不同记忆之间的点"

    何时不该调用（DO NOT CALL WHEN）:
      - 只需单条记忆检索即可回答——用 recall_memory 更高效。
      - 问题在当前上下文窗口内可直接回答。

    行为：内部调用双通路检索 Top-K，然后：
      - 启用 OpenAI 时：将检索结果喂给 LLM，生成综合回答
      - Mock 模式：返回拼接摘要 "Found N relevant memories: ..."

    Args:
        question: 需要综合回答的问题。
        user_id: 用户标识，默认 "hermes"。
        top_k: 检索条数上限，默认 5。
        query_tags: 可选标签过滤。
        query_entities: 可选实体 boost。
        agent_id: 可选 agent 过滤。

    Returns:
        成功: {"question":..., "answer":..., "hits":[...], "synthesis_mode":"mock"|"openai"}
        失败: {"error": "...", "status": "failed"}
    """
    uid = user_id or DEFAULT_USER
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as cli:
            body = {"user_id": uid, "question": question, "top_k": top_k}
            if query_tags:
                body["query_tags"] = query_tags
            if query_entities:
                body["query_entities"] = query_entities
            if agent_id:
                body["agent_id"] = agent_id
            r = cli.post("/v1/memory/synthesize", json=body, headers=_headers())
            r.raise_for_status()
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"synthesize_memory 失败: {exc}", "status": "failed"}


__all__ = ["remember_fact", "recall_memory", "synthesize_memory", "configure", "BASE_URL", "DEFAULT_USER"]
