"""VeriFin HTTP 服务层（FastAPI）。

路由：
- ``GET  /health``：存活探针 + 语料/会话规模
- ``POST /ask``：问答主入口（多轮会话经 conversation_id 复用 checkpoint）
- ``GET  /``：Web Demo 单页（存在 web/index.html 时托管）

装配复用 benchmark.runner 的 RetrievalEnvironment + 工具注册（与评测
同一条真实检索流水线：BM25 + Chroma + SQLite，非 stub）；图复用
``build_agent_graph`` 生产图。chromadb SQLite 线程亲和 → 服务用单线程
执行器承载装配与图执行（见 app.VerFiinService 文档字符串）。
"""

from verifin.api.app import VeriFinService, create_app

__all__ = ["VeriFinService", "create_app"]

