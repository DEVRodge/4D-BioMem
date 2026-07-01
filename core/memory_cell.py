"""4D-BioMem 核心突触剪枝算法层。

实现 SPEC 第三部分 §2 的两个数学模型：
  - 突触记忆权重衰减公式 (Synaptic Weight Decay Model)
  - 突触剪枝算子 (Synaptic Pruning Operator)

权重公式：
    W_i(t) = ∞                                  if R_i = 1   (风险硬锁定轨)
    W_i(t) = I_i · ln(1 + C_i) · e^{-λ·Δt}      if R_i = 0   (动态遗忘轨)

剪枝算子：
    M_active(t) = { M_i | W_i(t) ≥ θ_prune }
    即 W_i(t) < θ_prune 的非风险记忆被物理抹除。
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

RISK_LOCKED_WEIGHT = float("inf")

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
_SECONDS_PER_DAY = 86400.0


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class MemoryCell:
    """一条记忆细胞，对应四维空间 M = (T, F, R, V) 中的一个点。

    字段对齐 SPEC 第二部分 §2 的 agent_memory_cells 表结构。
    """

    content: str
    user_id: str = "default"
    agent_id: str = "default"
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_tags: dict[str, Any] = field(default_factory=dict)
    is_risk: bool = False              # R_i ∈ {0, 1}
    base_intensity: float = 3.0        # I_i ∈ [1, 10]，LLM 评估的初始显性强度
    access_count: int = 1              # C_i，被成功检索唤醒的总次数（默认 1，对应 schema）
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed_at: datetime = field(default_factory=_utcnow)
    current_weight: float = 3.0        # 占位初值，对齐 schema DEFAULT 3.0；引擎 register/load 时重算
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not 1.0 <= self.base_intensity <= 10.0:
            raise ValueError(f"base_intensity I_i 必须落在 [1, 10]，得到 {self.base_intensity}")
        if self.is_risk:
            self.current_weight = RISK_LOCKED_WEIGHT

    def compute_weight(self, current_time: datetime, lambda_: float) -> float:
        """SPEC 2.1 突触权重衰减公式。

        - 风险轨 (R=1)：返回 ∞，永久豁免剪枝。
        - 动态轨 (R=0)：I·ln(1+C)·e^{-λ·Δt}，Δt 以"天"为单位。
        """
        if self.is_risk:
            return RISK_LOCKED_WEIGHT
        delta_seconds = (current_time - self.last_accessed_at).total_seconds()
        delta_days = max(delta_seconds, 0.0) / _SECONDS_PER_DAY
        frequency_reward = math.log(1 + self.access_count)
        decay = math.exp(-lambda_ * delta_days)
        return self.base_intensity * frequency_reward * decay

    def access(self, current_time: datetime, lambda_: float) -> float:
        """被双通路检索命中 → 突触强化：C_i += 1，刷新 last_accessed_at，重算权重。"""
        self.access_count += 1
        self.last_accessed_at = current_time
        self.current_weight = self.compute_weight(current_time, lambda_)
        return self.current_weight

    def is_prunable(self, current_time: datetime, lambda_: float, theta_prune: float) -> bool:
        """SPEC 2.2：W_i(t) < θ_prune 且非风险 → 可剪枝。"""
        if self.is_risk:
            return False
        return self.compute_weight(current_time, lambda_) < theta_prune

    def weight_display(self) -> str:
        return "∞" if self.current_weight == RISK_LOCKED_WEIGHT else f"{self.current_weight:.4f}"


@dataclass
class PruneRecord:
    """剪枝审计日志条目（不可逆抹除后保留元数据指纹，不含内容本体可选）。"""

    cell_id: str
    content: str
    final_weight: float
    access_count: int
    pruned_at: datetime


class SynapticPruningEngine:
    """突触剪枝引擎：管理记忆池、推进模拟时钟、执行批量剪枝守护作业。

    时间模型：引擎持有一座可推进的 sim_clock，所有写入/检索/剪枝都以该时钟
    为准，从而支持"加速衰减"实验——真实秒级时间内推进数百个模拟日。
    """

    def __init__(
        self,
        lambda_: float = 0.05,
        theta_prune: float = 0.5,
        start_time: datetime | None = None,
    ) -> None:
        if lambda_ <= 0:
            raise ValueError("lambda_ 必须为正")
        if theta_prune <= 0:
            raise ValueError("theta_prune 必须为正")
        self.lambda_ = lambda_
        self.theta_prune = theta_prune
        self.cells: dict[str, MemoryCell] = {}
        self.prune_log: list[PruneRecord] = []
        self.sim_clock: datetime = start_time or _EPOCH

    # ---- 时间推进 ----------------------------------------------------------

    def tick(self, delta: timedelta) -> datetime:
        """推进模拟时钟。"""
        self.sim_clock = self.sim_clock + delta
        return self.sim_clock

    def now(self) -> datetime:
        return self.sim_clock

    # ---- 记忆生命周期 ------------------------------------------------------

    def register(self, cell: MemoryCell) -> MemoryCell:
        """写入流水线：把 cell 挂载到当前模拟时钟，并计算初始权重。"""
        cell.created_at = self.sim_clock
        cell.last_accessed_at = self.sim_clock
        cell.current_weight = cell.compute_weight(self.sim_clock, self.lambda_)
        self.cells[cell.id] = cell
        return cell

    def load_cell(self, cell: MemoryCell) -> MemoryCell:
        """从持久化层载入一条记忆，保留其原始 created_at / last_accessed_at
        （不像 register 那样重置为 sim_clock），仅按当前 sim_clock 重算权重。

        供里程碑 2 存储层 load_all_active_cells 后重建引擎池使用。
        """
        cell.current_weight = cell.compute_weight(self.sim_clock, self.lambda_)
        self.cells[cell.id] = cell
        return cell

    def access(self, cell_id: str) -> float | None:
        """检索命中某条记忆 → 突触强化。"""
        cell = self.cells.get(cell_id)
        if cell is None:
            return None
        return cell.access(self.sim_clock, self.lambda_)

    def run_pruning(self) -> list[MemoryCell]:
        """SPEC 2.2 剪枝守护作业：扫描全池，物理抹除 W_i(t) < θ_prune 的非风险记忆。

        返回本次被剪枝的记忆列表，并在 prune_log 留下审计指纹。
        """
        pruned: list[MemoryCell] = []
        for cell_id in list(self.cells.keys()):
            cell = self.cells[cell_id]
            weight = cell.compute_weight(self.sim_clock, self.lambda_)
            cell.current_weight = weight
            if cell.is_prunable(self.sim_clock, self.lambda_, self.theta_prune):
                self.prune_log.append(
                    PruneRecord(
                        cell_id=cell.id,
                        content=cell.content,
                        final_weight=weight,
                        access_count=cell.access_count,
                        pruned_at=self.sim_clock,
                    )
                )
                pruned.append(cell)
                del self.cells[cell_id]
        return pruned

    # ---- 查询 --------------------------------------------------------------

    def active_cells(self) -> list[MemoryCell]:
        """返回当前有效记忆集合 M_active(t)。"""
        return list(self.cells.values())

    def get(self, cell_id: str) -> MemoryCell | None:
        return self.cells.get(cell_id)

    def snapshot(self) -> list[dict[str, Any]]:
        """带权重的快照，用于实验报告。"""
        rows: list[dict[str, Any]] = []
        for cell in self.cells.values():
            w = cell.compute_weight(self.sim_clock, self.lambda_)
            cell.current_weight = w
            rows.append({
                "id": cell.id,
                "content": cell.content,
                "is_risk": cell.is_risk,
                "I": cell.base_intensity,
                "C": cell.access_count,
                "weight": w,
                "last_accessed_at": cell.last_accessed_at,
            })
        return rows

    def __len__(self) -> int:
        return len(self.cells)


def format_weight(w: float) -> str:
    return "∞" if w == RISK_LOCKED_WEIGHT else f"{w:.4f}"


__all__ = [
    "RISK_LOCKED_WEIGHT",
    "MemoryCell",
    "SynapticPruningEngine",
    "PruneRecord",
    "format_weight",
]
