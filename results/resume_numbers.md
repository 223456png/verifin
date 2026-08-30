# VeriFin — 简历数字（Phase 9 收官，2026-08-30）

> 本文件是从评测报告（results/synth_check3、results/finqa_v8、results/convfinqa_check）
> 提炼的面试/简历可引用数字与诚实边界说明。所有数字均可回溯到对应报告文件。

## 一句话定位

LangGraph 多智能体金融证据校验 RAG 系统：BM25+Dense 混合检索（多查询合并 +
small-to-big 父文档补全）、四要素规则校验、冲突仲裁与重规划、会话偏好记忆、
PoT 安全计算器 + 程序模板执行器；检索升级（语义 embedding 替换哈希向量）
与 126 项测试全绿，完成真实 FinQA / ConvFinQA 评测并提供 FastAPI 服务层与 Web Demo。

## 可引用数字

1. **端到端准确率（合成评测套件，20 样本 × 5 配置消融，多轮）**：完整系统 **100.0%**
2. **消融贡献（合成套件，多轮口径）**：Verifier **+25 pp**（100% vs 75%）、
   Replanner **+15 pp**（100% vs 85%）、偏好记忆 **+15 pp**（100% vs 85%）
3. **真实 FinQA test（1,147 题，评测 100 样本）**：
   - 答案 EM **1.0% → 6.0%（6 倍，Phase 9）**；模板可检测子集（25 题）**24%**
     （金标文档已召回子集 30%）
   - 文档级召回 recall@20：**48% → 56%**（多子任务检索合并 + top_k 25）
4. **真实 ConvFinQA dev 多轮（100 轮）**：文档级召回 **52% → 65%（+13 pp）**，
   多轮继承/偏好机制端到端跑通
5. **工程指标（真实数据）**：Tool F1 95.6%、平均轨迹步数 7.7、重规划有效率 46%
6. **检索升级（单查询直接检索口径，FinQA test 100 样本）**：语义 embedding
   bge-small 替换无语义 hash 向量，doc recall@20 **81.0% → 86.0%（+5.0 pp）**；
   cross-encoder rerank 在此口径下 **-1.0 pp**（通用精排对金融表格 doc 召回无益，
   如实报告）
7. **测试资产**：126 项全绿（含程序执行器 / 表格解析 / 检索合并回归）

## Phase 9 关键工程改动（面试可展开）

1. **多子任务检索合并（multi-query merge）**：计算流基期/比较期检索此前每轮
   **覆盖**上一轮结果（最终只剩最后一个子任务的文档，原始问题的强词法信号被
   丢弃）→ 按 chunk_id 去重合并（上限 80）+ top_k 20→25。这是 LangChain
   MultiQueryRetriever 的同思路实现。
2. **表格解析修复**：无分隔行（`|---|`）的 markdown 表格此前 `rows[2:]` 会把
   **首条数据行当分隔行丢掉**（如 "2014 net revenue" 行），修复后水平/垂直
   表格的年份候选不再缺行。
3. **候选锚定体系（anchor score）**：表格行 metric 命中+年份在行标签（0.0）<
   行标签×问题**双向**词重叠（0.5）< 散文证据 metric 命中（1.0）< 新闻源
   惩罚（+0.5）< 距离兜底（50+）；父文档补全候选仅 +0.25 来源惩罚（不抹平
   语义层级）；跨 chunk 组合 +1.0 惩罚（双期数值几乎总在同表）。
4. **数值抽取净化**："Form 10-K" 的 10 此前被抽为数值 10.0，与真实值 12000
   构成**假 major conflict** 导致重规划螺旋耗尽——数字后紧跟 `-字母` 的
   SEC 报表编号（10-K/10-Q/S-1）已剔除。
5. **FinQA 方向惯例**："percentage DECREASE from 2009 to 2010" 的金标 program
   为 `subtract(2009值, 2010值), divide(#0, 2009值)`（降幅取正），与标准
   (target-base)/base 相反——检测到 decrease 措辞时翻转求值方向。

## Phase 9.5 检索升级（面试可展开）

1. **语义 embedding 替换哈希向量**：Phase 1 的 HashEmbedding 是 SHA256 词袋
   哈希投影（无语义，仅"词重叠越多越相似"），在真实财报上的 Dense 路为负贡献。
   接入 `bge-small-en-v1.5`（sentence-transformers，384 维），通过
   `--embedding bge` 构建 + manifest 记录 + 读取时自动按 manifest 装配，
   单查询直接检索 doc recall@20 **81.0% → 86.0%（+5.0 pp）**。
2. **可插拔 embedding 工厂**：`get_embedding_function("hash"|"bge")` 统一入口，
   hash 为离线零依赖保底（测试/CI/demo 开箱即用），bge 为显式语义升级；
   索引 manifest 持久化嵌入选择，跨进程读取无需手工指定。
3. **rerank 接回默认路径但如实降级**：CrossEncoder 精排此前"写了但未接入主流程"，
   现从 manifest/参数贯通；但实测通用 `ms-marco-MiniLM` 对金融表格 doc 召回
   **-1.0 pp**（通用段落精排对数值表格敏感性低），保留为可选并在报告如实标注。

## 诚实边界（面试必答，已写入报告中）

- Agent 为**无 LLM 的确定性规则引擎**。FinQA test 答案类型分解：**提取型 4% /
  推导型（需 program 算术）93% / 布尔型 3%**——program 模板覆盖 growth_pct /
  difference 两类（占检测子集 25%），table_sum/exp_avg/greater 及多步 add 链
  未覆盖，属 LLM program 生成范畴（Phase 10 规划）。
- hash-Dense embedding 无语义，已在 Phase 9.5 用 bge-small 替换（单查询 doc
  recall 81%→86%，+5 pp）；但通用 `ms-marco` cross-encoder rerank 对金融表格
  doc 召回无益（-1 pp）——通用段落精排与金融数值表格的语义分布不匹配，
  金融域专用 rerank 留待 Phase 10。
- 真实大语料上整体 EM 仍低：检索（doc recall 56%）与 program 覆盖是两大瓶颈，
  均已在报告中按口径分解如实披露。

## 数字回溯索引

| 数字 | 报告 |
|------|------|
| 合成 100% / 消融贡献 | [results/synth_check3/report.md](synth_check3/report.md) |
| FinQA EM 6.0% / det 24% / recall 56% | [results/finqa_v8/report.md](finqa_v8/report.md) |
| ConvFinQA recall 65% | [results/convfinqa_check/report.md](convfinqa_check/report.md) |
| 检索升级 recall 81%→86% | `scripts/compare_retrieval.py`（FinQA test 100 样本实测） |
