"""VeriFin MCP 工具包（Phase 12.4）。

将金融证据校验能力暴露为 MCP 服务，供任何 MCP 客户端复用。

注意：``verifin.mcp.server`` 在导入时会按需（首次工具调用时）加载索引，
因此本包可以安全导入而不要求索引存在。

启动服务::

    VERIFIN_INDEX_DIR=./bge_env python -m verifin.mcp.server
"""

__all__: list[str] = []
