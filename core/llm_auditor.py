"""4D-BioMem 生产级 LLM 审计 & Embedding 模块。

提供两个可替换后端：
  - OpenAI 后端：调用 OpenAI Chat Completions / Embeddings API
  - Mock 后端：keyword 匹配 + 4-gram hash（科研原型用）

自动降级：配置了 OPENAI_API_KEY 且 llm_backend=openai 时用 OpenAI，
否则用 Mock。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from config import settings

EMBED_DIM = 1536  # text-embedding-3-small 输出维度

# ── OpenAI 审计 ─────────────────────────────────────────────────


class OpenAILLMAuditor:
    """用 OpenAI 大模型做记忆审计：风险识别 + 强度评分 + 标签提取 + 实体提取。"""

    SYSTEM_PROMPT = """You are a memory auditor for an AI agent's long-term memory system.
Analyze the user's message and return a JSON object with exactly these fields:
{
  "is_risk": bool,       // true if content contains safety-critical info (allergies, passwords, secrets, health conditions)
  "base_intensity": int, // importance score 1-10: 10=life-critical, 7=project/technical, 4=general, 2=trivial/chat
  "task_tags": {          // structured tags for retrieval filtering
    "type": str,          // "medical" | "secret" | "tech" | "casual" | "general"
    "project": str | null // if a project name is mentioned, extract it
  },
  "entities": [           // extracted named entities (empty list if none)
    {"name": str, "type": str, "role": str}
  ]
}
Types: "person", "project", "version", "tech", "org", "location"
Roles: "subject", "owner", "tool", "reference"
Return ONLY the JSON object, no other text."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model = settings.openai_model
        self._mock = MockLLMAuditor()

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._client

    async def audit(self, content: str) -> dict[str, Any]:
        """审计一条记忆，返回 {is_risk, base_intensity, task_tags}。"""
        # 没有 API Key 时静默降级为 Mock
        if not settings.openai_api_key:
            return await self._mock.audit(content)
        try:
            client = self._ensure_client()
            resp = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            raw_entities = data.get("entities", [])
            if not isinstance(raw_entities, list):
                raw_entities = []
            return {
                "is_risk": bool(data.get("is_risk", False)),
                "base_intensity": int(data.get("base_intensity", 4)),
                "task_tags": data.get("task_tags", {"type": "general"}),
                "entities": raw_entities,
            }
        except Exception as exc:
            # OpenAI 调用失败时降级到 Mock，不中断服务
            import sys
            print(f"[OpenAILLMAuditor] 降级到 Mock: {exc}", file=sys.stderr)
            return await self._mock.audit(content)


# ── OpenAI Embedding ────────────────────────────────────────────


class OpenAIEmbedder:
    """用 OpenAI Embeddings API 生成文本向量。"""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model = settings.openai_embedding_model
        self._dim = EMBED_DIM
        self._mock = MockEmbedder()

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._client

    def _compute(self, text: str) -> list[float]:
        """同步计算向量（供 seed 等非异步路径调用）。"""
        if not settings.openai_api_key:
            return self._mock._compute(text)
        # 如果已有缓存的异步结果可直接返回，否则 fallback 到 mock
        return self._mock._compute(text)

    async def embed(self, text: str) -> list[float]:
        """异步计算向量。"""
        if not settings.openai_api_key:
            return await self._mock.embed(text)
        try:
            client = self._ensure_client()
            resp = await client.embeddings.create(
                model=self._model,
                input=text,
                dimensions=self._dim,
            )
            return resp.data[0].embedding
        except Exception as exc:
            import sys
            print(f"[OpenAIEmbedder] 降级到 Mock: {exc}", file=sys.stderr)
            return await self._mock.embed(text)

    async def embed_dim(self) -> int:
        """异步获取向量维度。"""
        return self._dim


# ── Mock 后备（原 api/main.py 中的实现，内联以避免循环依赖）────


class MockLLMAuditor:
    """关键词匹配的后备审计器（无外部依赖）。"""

    RISK_KEYWORDS = {"过敏", "青霉素", "阿司匹林", "药物过敏", "哮喘",
                     "密码", "口令", "secret", "password", "凭证",
                     "秘密", "机密", " confidential", "隐私",
                     "手术", "病史", "心脏病", "高血压"}

    async def audit(self, content: str) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        text = content.lower()
        is_risk = any(kw.lower() in text for kw in self.RISK_KEYWORDS)
        return {
            "is_risk": is_risk,
            "base_intensity": self._estimate_intensity(content, is_risk),
            "task_tags": self._extract_tags(content),
            "entities": self._extract_entities(content),
        }

    @staticmethod
    def _estimate_intensity(content: str, is_risk: bool) -> float:
        if is_risk:
            return 10.0
        tech_signals = ["项目", "bug", "修复", "方案", "架构", "api", "部署", "alpha", "beta"]
        if any(s in content.lower() for s in tech_signals):
            return 7.0
        trivial_signals = ["吃了", "今天", "中午", "天气", "闲聊"]
        if any(s in content for s in trivial_signals):
            return 2.0
        return 2.0

    @staticmethod
    def _extract_tags(content: str) -> dict[str, Any]:
        tags: dict[str, Any] = {}
        m = re.search(r"项目\s*([A-Za-z0-9_\-]+)", content)
        if m:
            tags["project"] = m.group(1)
        if any(k in content for k in ("过敏", "病", "药")):
            tags["type"] = "medical"
        elif any(k in content.lower() for k in ("bug", "项目", "修复", "方案")):
            tags["type"] = "tech"
        elif any(k in content for k in ("吃了", "今天", "天气")):
            tags["type"] = "casual"
        else:
            tags["type"] = "general"
        return tags

    @staticmethod
    def _extract_entities(content: str) -> list[dict[str, str]]:
        entities: list[dict[str, str]] = []
        # 项目名（中文或英文，跟在"项目"后面）
        for m in re.finditer(r"项目\s*([A-Za-z0-9_\-一-鿿]+)", content):
            entities.append({"name": m.group(1), "type": "project", "role": "subject"})
        # 人名（跟在"用户"或"开发者"或"负责人"后面）
        for m in re.finditer(r"(?:用户|开发者|owner|负责人)[：:\s]*([A-Za-z0-9_\-一-鿿]{2,})", content):
            entities.append({"name": m.group(1), "type": "person", "role": "owner"})
        # 版本号
        for m in re.finditer(r"v?(\d+\.\d+(?:\.\d+)?)", content):
            entities.append({"name": m.group(0), "type": "version", "role": "reference"})
        # 常见技术术语
        TECH_TERMS = {"redis", "postgresql", "k8s", "prometheus", "docker",
                      "kubernetes", "mysql", "nginx", "fastapi", "uvicorn"}
        for term in TECH_TERMS:
            if term in content.lower():
                entities.append({"name": term, "type": "tech", "role": "tool"})
        # agent 名
        for m in re.finditer(r"agent\s*[：:\s]*([A-Za-z0-9_\-]+)", content, re.IGNORECASE):
            entities.append({"name": m.group(1), "type": "agent", "role": "owner"})
        # 去重：同 (name.lower(), type) 只保留第一条
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, str]] = []
        for e in entities:
            key = (e["name"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        return deduped


class MockEmbedder:
    """4-gram hash 后备嵌入器（零额外依赖）。"""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _compute(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        grams = [text[i:i + 4] for i in range(max(1, len(text) - 3))]
        if not grams:
            grams = [text or "_"]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        h2 = int(hashlib.md5((text or "_").encode("utf-8")).hexdigest(), 16)
        vec[(h2 // 7) % self.dim] += 2.0
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec.tolist()

    async def embed(self, text: str) -> list[float]:
        await asyncio.sleep(0.01)
        return self._compute(text)
