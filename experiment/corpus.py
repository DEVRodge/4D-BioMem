"""标注语料 + 地面真值查询。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CorpusItem:
    content: str
    category: str  # "risk" | "tech" | "casual"


@dataclass
class GroundTruthQuery:
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    top_k: int = 3
    expected_category: str = "tech"


# --- 50 条语料，扩展自 run_benchmark.py ---
CORPUS: list[CorpusItem] = [
    # 1-10
    CorpusItem("项目 Alpha 的 Bug 修复方案采用重试队列加幂等键", "tech"),
    CorpusItem("今天天气挺好，适合散步", "casual"),
    CorpusItem("中午吃了牛肉面，味道一般", "casual"),
    CorpusItem("项目 Alpha 的部署架构用 k8s 加双活", "tech"),
    CorpusItem("我对青霉素过敏，开药千万别用青霉素类", "risk"),
    CorpusItem("昨晚看了一部电影，挺无聊的", "casual"),
    CorpusItem("项目 Alpha 的 Bug 根因是并发竞态", "tech"),
    CorpusItem("早上喝了杯美式咖啡", "casual"),
    CorpusItem("周末想去爬山", "casual"),
    CorpusItem("项目 Alpha 的监控用 prometheus 加告警", "tech"),
    # 11-20
    CorpusItem("项目 Alpha 的 Bug 修复后压测通过", "tech"),
    CorpusItem("今天地铁有点挤", "casual"),
    CorpusItem("服务器 root 密码是 Alpha-Bug-2024 务必保密", "risk"),
    CorpusItem("晚饭吃了酸菜鱼", "casual"),
    CorpusItem("最近睡眠不太好", "casual"),
    CorpusItem("项目 Alpha 的回滚方案是灰度发布", "tech"),
    CorpusItem("今天有点累", "casual"),
    CorpusItem("中午和同事吃了火锅", "casual"),
    CorpusItem("项目 Alpha 的数据备份策略是每日全量", "tech"),
    CorpusItem("下班路上堵车了", "casual"),
    # 21-30
    CorpusItem("早上吃了油条和豆浆", "casual"),
    CorpusItem("项目 Alpha 的安全审计修复了 SQL 注入", "tech"),
    CorpusItem("今天风挺大", "casual"),
    CorpusItem("晚上打了会儿游戏", "casual"),
    CorpusItem("项目 Alpha 的性能优化用上了缓存", "tech"),
    CorpusItem("我的银行卡密码是 123456 请不要外传", "risk"),
    CorpusItem("今天心情不错", "casual"),
    CorpusItem("项目 Alpha 的文档已更新到 v2 版本", "tech"),
    CorpusItem("晚上看了会儿书", "casual"),
    CorpusItem("今天下班早点", "casual"),
    # 31-40
    CorpusItem("中午吃了酸菜鱼饭", "casual"),
    CorpusItem("项目 Alpha 的 CI/CD 集成到了 Jenkins", "tech"),
    CorpusItem("周末和朋友去打球", "casual"),
    CorpusItem("项目 Beta 的新功能开发优先级讨论", "tech"),
    CorpusItem("我有哮喘病史 剧烈运动要注意", "risk"),
    CorpusItem("今天路边的花开了", "casual"),
    CorpusItem("项目 Beta 的数据库迁移方案已评审", "tech"),
    CorpusItem("午睡了一会儿", "casual"),
    CorpusItem("项目 Alpha 的日志收集用了 ELK 栈", "tech"),
    CorpusItem("公司楼下新开了一家奶茶店", "casual"),
    # 41-50
    CorpusItem("注意：系统 API Key 是 sk-biomem-demo-2024", "risk"),
    CorpusItem("今天买了新衣服", "casual"),
    CorpusItem("项目 Alpha 的容器化方案用了 Docker Compose", "tech"),
    CorpusItem("晚上想吃点清淡的", "casual"),
    CorpusItem("下周末计划去郊游", "casual"),
    CorpusItem("项目 Beta 的单元测试覆盖率达到 85%", "tech"),
    CorpusItem("今天咖啡喝多了睡不着", "casual"),
    CorpusItem("项目 Alpha 的熔断机制用 Sentinel 实现", "tech"),
    CorpusItem("周末把房间打扫了一遍", "casual"),
    CorpusItem("我的社保卡号和身份证号关联要保密", "risk"),
]

GROUND_TRUTH_QUERIES: list[GroundTruthQuery] = [
    GroundTruthQuery(
        query="项目 Alpha 的 Bug 修复方案采用什么方式",
        expected_keywords=["重试队列", "幂等键"],
        expected_category="tech",
    ),
    GroundTruthQuery(
        query="项目 Alpha 的部署架构用 k8s 还是双活",
        expected_keywords=["k8s", "双活"],
        expected_category="tech",
    ),
    GroundTruthQuery(
        query="用户有哪些过敏史或病史需要关注",
        expected_keywords=["青霉素", "过敏", "哮喘"],
        expected_category="risk",
    ),
    GroundTruthQuery(
        query="服务器密码 API Key 是什么",
        expected_keywords=["密码", "API Key", "Alpha-Bug-2024"],
        expected_category="risk",
    ),
    GroundTruthQuery(
        query="项目 Alpha 安全审计 SQL 注入怎么修的",
        expected_keywords=["SQL 注入", "安全审计"],
        expected_category="tech",
    ),
]


def get_category_counts() -> dict[str, int]:
    counts: dict[str, int] = {"risk": 0, "tech": 0, "casual": 0}
    for item in CORPUS:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def get_contents_by_category(cat: str) -> list[str]:
    return [it.content for it in CORPUS if it.category == cat]
