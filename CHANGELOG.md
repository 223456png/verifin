# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格；
版本号在 `pyproject.toml` 中维护，与开发阶段（Phase）对应。

## [Unreleased]

### Added

- **工程化基线**：GitHub Actions CI（ruff 静态检查 + Python 3.10/3.11/3.12 测试矩阵 + 覆盖率）、`LICENSE`（MIT）、`.mailmap`（归并早期提交身份）、`.env.example`、`CHANGELOG.md`
- **容器化**：`Dockerfile`、`docker-compose.yml`、`.dockerignore`，`docker compose up --build` 一键启动 Web Demo
- **文档**：`docs/EVALUATION.md` —— 评测结论、逐阶段工程归因与已知局限的集中汇总
- `pyproject.toml`：补齐 `authors` / `keywords` / `classifiers` / `[project.urls]`，新增 `verifin-mcp` 入口点与 `[tool.ruff]` / `[tool.coverage]` 配置
- README：新增 TL;DR、MCP Server 章节（工具表 + 客户端配置 + 安全边界）、Docker 与 MCP 启动方式
- **核心消融数字链证据入库**（`results_v*/report.md` + LLM 调用摘要，`full/` 逐样本产物不入库）；`docs/EVALUATION.md` 回溯索引新增「证据状态」列

### Changed

- **MCP Server 从 `scripts/mcp_server.py` 迁移到 `src/verifin/mcp/server.py`**，成为包内一等能力，并支持 `python -m verifin.mcp.server` 与 `verifin-mcp` 两种启动方式
- MCP Server 索引加载改为**惰性**（首次工具调用时触发），导入模块不再要求索引存在，便于 CI 与测试
- `mcp>=1.0.0` 加入核心依赖 —— 此前 MCP Server 依赖未在 `pyproject.toml` 中声明，包安装后无法直接运行
- `scripts/test_mcp_server.py`：改用 `sys.executable` 启动子进程（原先硬编码 `.venv/bin/python`，Windows 下不可用），并注入 `PYTHONPATH` 支持源码模式运行
- `src/verifin/mcp/server.py` 文档中的「无 eval」表述修正为「AST 白名单 + 空 builtins 的受限 eval」，与实际实现保持一致

### Removed

- `results/resume_numbers.md` —— 内容改写为 `docs/EVALUATION.md`（中性化的评测与局限文档）

---

## [0.13.1] — 2026-09-29 · Structured Outputs + 消融实证

### Added

- **最小 Tool-Calling Harness（Phase 13）**：`verifin.harness.ToolCallingHarness` —— LLM 按 OpenAI function calling 协议自主调度 `ToolRegistry` 工具，tool 消息回喂自主纠正，三停机条件（finished / max_turns / llm_error）防死循环；17 项单测锁定调度 / 回喂 / 停机语义。定位为与 graph 确定性路由互补的**开放任务能力层**，不进 EM 评测口径（见 `docs/EVALUATION.md` Phase 13 章节）
- **LLM 调用接 Structured Outputs（Phase 13.1）**：`response_format` 透传（OpenAI `json_schema` strict 形态；planner schema `financial_plan` / programmer schema `dsl_program`），端点不支持时自动摘除回退（400/422 即摘、不吃退避预算），`--json-schema` 消融开关；8 项测试锁定透传 / 回退 / 接线语义
- **`--json-schema` 单变量消融实测**（FinQA test 100 样本、programmer 模式、DeepSeek 直连，vs v125 基线）：**EM 31.0% vs 30.0%**（+1pp = 多对 1 题，噪声量级）；LLM 调用 44/44 成功、attempt=1 零重试（基线 45/45 同为 100%——宽容解析在 DeepSeek 上本已零失败，json-schema 买到的是**确定性 schema 保证**而非实测解析救回）；代价 = 平均延迟 524.7ms → 713.2ms（**+36%**）。证据：`results_v13_js/`

### 副产物

- **无 LLM 降级基线 EM 22.0%**（key 失效轮：176 次调用全 401、系统静默降级）——与 v122_norank 规则基线 21.0% 吻合，反向验证降级架构：LLM 全挂时管线不崩、EM 回到确定性模板水平

---

## [0.12.5] — 2026-09-04 · 实体门控 + 实体链接负向结果

### Added

- **实体门控激活（Phase 12.5）**：claim 实体缺失的定量修复，EM 29.0% → 30.0%（v125）
- **LLM 辅助实体链接（Phase 12.5b，诚实负向结果）**：v126 实测 EM 29.0%（-1），链接器仅激活 1/100——内容实体候选稀疏（公司名多在 doc_id 而非正文），根因定性为信息论边界；代码保留（缓存 + 校验 + 降级完备）。证据：`results_v125/`、`results_v126/`

---

## [0.12.4] — 2026-09-04 · MCP Server + LLM-as-judge

### Added

- **MCP Server**（FastMCP stdio）：四工具只读暴露 —— `retrieve` / `expand_document` / `verify_claim` / `calculate`；`VERIFIN_INDEX_DIR` 指定索引；统一 JSON 序列化规避 FastMCP 裸 list 截断
- `scripts/test_mcp_server.py`：协议级冒烟 —— `initialize` / `tools/list` / 四工具全调用实测通过
- `scripts/llm_judge.py`：LLM-as-judge（faithfulness + relevancy），证据取四要素校验通过的 chunk，支持 bridge / api-key 双通道

### Changed

- `llm_provider.py`：`BridgedLLM` 补 `kind` 参数，支持 judge 等多实例场景

### Notes

- LLM-as-judge 分层发现：答错 27 题 faithfulness 0%（噪声证据放行后算错值的直接实证），答对 11 题 36%，与 EM 一致性 82%

---

## [0.12.3] — 2026-09-04 · 真实 LLM 程序生成接入

### Added

- `llm_provider.py`：`complete(prompt) -> str` 双通道 provider —— `BridgedLLM`（本地桥接）与 `OpenAICompatLLM`（OpenAI 兼容直连）。共同契约：永不抛异常、失败空串触发既有降级、逐调用统计落盘、3 次指数退避重试（3s/6s/12s）
- `scripts/summarize_llm_ablation.py`：LLM on/off 自动对比归档
- `run_benchmark.py`：`--llm-api-key` / `--llm-bridge` / `--llm-mode` 单变量消融开关

### Changed

- `runner` / `build_benchmark_graph` 透传 `llm_planner` / `llm_programmer`（仅生产图）

### Notes

- **EM 21.0% → 29.0%**（100 样本，同降级 reranker 口径）：LLM 程序生成接管模板未检出的 46 题，净 +8 题 / 零回归，46/46 调用成功（DeepSeek，均值 762ms）
- LLM planner 消融（24.0%，+5/-2）揭示「分解质量 ≠ 下游兼容性」，架构定为**规则 planner + LLM 程序生成**

---

## [0.12.2] — 2026-09-04 · 跨文档污染守卫

### Fixed

- **跨公司同构表污染**（回归根因）：CME 题召回 GPN 股权计划表，分母短语命中异公司行标签导致分母锚定错误。`_ratio_candidates` 新增守卫 —— ① chunk→doc 映射（检索 + expand 双来源）；② 分母候选与分子最优锚定（按 anchor_score 取优，非扫描序）；③ 不同文档时剔除异文档候选；④ 清空后 total 行回退限定分子同文档
- **reranker 失效伪影**：`rerank()` 每次调用重跑 `_load_model()`，HF SSL 阻断时每次检索卡约 40s（100 样本评测被拖至 34 分钟）。改为 `_load_attempted` 失败缓存（每进程只尝试一次），并新增 `RERANKER_MODEL` 环境变量支持本地模型目录离线加载

### Added

- `scripts/repro_regressions.py`：Phase 12.1 regression 逐样本复现
- 4 项 reranker 单测（失败缓存语义 / 成功只加载一次 / env 覆盖）+ 2 项跨文档守卫单测

### Notes

- 3/3 回归样本修复；同降级 reranker 口径 EM 13.0% → 21.0%；错误分布 calculation_error 38 → 33、verifier_reject 29 → 26

---

## [0.12] — 2026-09-04 · 程序覆盖扩展 + 锚定精度修复

### Added

- 5 类 ratio 句式正则（ratio of X to Y / X-to-Y / percent of X to Y / as a percentage of / represented）
- `average between Y1 and Y2` 均值模板 + `_execute_average` 执行器
- ratio 候选收集支持短语内年份锚定（purchase in dec 2012 vs jan 2013）
- `trace.extract_answer_value` 纳入 average alternates 的 value 刻度
- Phase 12.1 锚定精度修复：chunk 级指标门控、`_table_column_years` 跳过首列含日期文本的伪年份列、`collect_table_year_values` 垂直分支补扫首数据行 + roll-forward 行结构锚定

### Notes

- 全测 1147 六族句式未检出 122 → 9（-93%，设计目标 ≥60%）；程序可检测子集 38 → 57 题；EM 12.0% → 21.0%

---

## [0.11] — 2026-09-01 · 检索召回修复

### Fixed

- **排除语义修正（核心）**：`exclude_chunk_ids` 原实现把失败 chunk 从累积证据池整体移除，但 verifier 对每个子任务按各自 claim 重新校验。修正为只过滤本轮新检索结果、不驱逐历史证据
- **表格 chunk 豁免**：`verify_claim_batch` 富化 `is_table` 标记，`get_failed_chunk_ids` 跳过表格条目
- **replanner 原地打转终止**：新子任务与既有 sub_tasks 文本相同视为无新信号，走终止路径

### Added

- `scripts/diagnose_recall.py`：六组对照归因诊断
- `scripts/trace_gold_loss.py`：44 失败样本逐个跑图追踪
- Makefile `benchmark-finqa` 目标（bge 索引端到端路径）

### Notes

- 归因结论：检索层本身 88%，端到端 56%，损耗 59% 来自状态管理而非检索能力
- doc recall **56% → 88.0%**；EM 7.0% → 8.0%（净 +2/-1，诚实披露）；26 个 LOSS-BY-EXCLUDE 样本复跑 23/25 转为 gold 命中

---

## [0.10] — 2026-08-30 · 比率模板 + DSL 执行器 + LLM 程序生成

### Added

- ratio 短语键控模板（"what percentage of X are Y"，FinQA test 占 ~17%）
- `execute_dsl` 多步执行器：`#N` 步骤引用 / `vN`、`tN` 候选绑定 / `const_K` 常量，聚合算子展开后过 PoT 沙箱
- `LLMProgramGenerator`：可插拔 provider，编号候选供消歧，输出过 `validate_dsl` 结构校验（算子白名单 / 引用越界 / 幻觉候选拒绝）——**LLM 输出永不直接执行**；五类失败均降级回确定性模板

### Notes

- Phase 10.1 六项 ratio 锚定改进后，理想检索口径锚定精度 **2/13 → 7/13**
- Tool F1 95.6% → 94.8% 为**口径变化**（calc 触发词补齐 portion/fraction/percent of）

---

## [0.9.7] — 2026-08-30 · 实体键控计算流

### Added

- `cross_entity_diff` / `argmax_relay` 程序模板（检测 + 求值）
- `collect_entity_values`：实体提及 + 指标同义词 + 年份三信号最近邻的实体锚定数值扫描
- calculator 节点实体键控候选路径 + planner 实体子任务分解
- LLM 规划层、多跳 QA 合成管线、RAGAs 风格指标

### Notes

- 多跳合成基准 EM **37.5% → 100%**（跨实体差值 8 条、三实体接力 2 条此前为 0%）

---

## [0.9.5] — 2026-08-30 · 检索升级

### Added

- 可插拔 embedding 工厂 `get_embedding_function("hash" | "bge")`；索引 manifest 持久化嵌入选择
- `bge-small-en-v1.5` 语义 embedding 接入
- `scripts/compare_retrieval.py` 检索对比

### Changed

- rerank 接回默认路径（CrossEncoder 降级透传）；单测统一 reranker 降级夹具，不加载 200MB 模型

### Notes

- doc recall@20 **81.0% → 86.0%（+5.0 pp）**；通用 cross-encoder rerank 在此口径 **-1.0 pp**（如实报告）

---

## [0.1.0] — 2026-08-30 · 首次发布

### Added

- LangGraph 多智能体金融 RAG 骨架（Phase 1–9）：四要素校验（Entity / Period / Metric / Definition）、冲突仲裁与重规划、会话偏好记忆、PoT 安全计算器 + 程序模板执行器
- BM25 + Dense 混合检索（多查询合并 + small-to-big 父文档补全）
- 真实 FinQA / ConvFinQA 端到端评测与消融框架
- FastAPI 服务层 + Web Demo 轨迹可视化
- Makefile 快捷命令（demo / test / benchmark / index）

### Notes

- 首版实测：EM 1.0% → 6.0%，doc recall 48% → 56%
