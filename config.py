"""4D-BioMem 配置模块。

从环境变量 + .env 文件读取所有运行时配置。
零额外依赖（仅用 os.environ 和 dataclass）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class Settings:
    # ── LLM 后端 ─────────────────────────────────────────────────
    llm_backend: str = "mock"  # "openai" | "mock"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── 服务鉴权 ─────────────────────────────────────────────────
    api_key: str = ""  # 空 = 不启用鉴权

    # ── 存储路径 ─────────────────────────────────────────────────
    db_path: str = "/data/biomem.db"
    vector_path: str = "/data/vector_store"
    wiki_path: str = "/data/wiki"
    prefer_chroma: bool = False

    # ── 算法超参 ─────────────────────────────────────────────────
    lambda_: float = 0.05
    theta_prune: float = 0.5
    tau: float = 1.0

    # ── 日志 ────────────────────────────────────────────────────
    log_level: str = "info"

    # ── 演示数据 ─────────────────────────────────────────────────
    seed: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        llm = _env("LLM_BACKEND", "mock")
        return cls(
            llm_backend=llm,
            openai_api_key=_env("OPENAI_API_KEY", ""),
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=_env("API_KEY", ""),
            db_path=_env("DB_PATH", "/data/biomem.db"),
            vector_path=_env("VECTOR_PATH", "/data/vector_store"),
            wiki_path=_env("WIKI_PATH", "/data/wiki"),
            prefer_chroma=_env_bool("PREFER_CHROMA", False),
            lambda_=float(_env("LAMBDA", "0.05")),
            theta_prune=float(_env("THETA_PRUNE", "0.5")),
            tau=float(_env("TAU", "1.0")),
            log_level=_env("LOG_LEVEL", "info"),
            seed=_env_bool("SEED", True),
        )

    @property
    def use_openai(self) -> bool:
        return self.llm_backend == "openai" and bool(self.openai_api_key)


# 全局单例（模块导入时自动加载）
settings = Settings.from_env()
