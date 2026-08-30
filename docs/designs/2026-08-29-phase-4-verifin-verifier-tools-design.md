# Phase 4 — Verifier Tools（四要素校验 + PoT 计算器）设计文档

日期：2026-08-29
范围：VeriFin 结构化证据校验与数值计算能力（承接 Phase 3 Agent Core）

## 1. Intent（意图）

把 Phase 3 的占位校验/占位计算替换为 VeriFin 的核心差异化能力：

1. **Evidence 抽取**：把文档 chunk 规则化抽取为四要素 + 数值的结构化证据；
2. **四要素校验**：Entity / Period / Metric / Definition 逐项比对，拦截证据混淆、指标错配、口径不一；
3. **PoT 计算器**：受限沙箱内安全求值数值表达式（支持"增长了多少%"类问题）；
4. **Agent 集成**：verifier_node 接入真实校验，replanner 基于 mismatch 诊断定向重检索，synthesizer 仅绑定通过证据。

## 2. Constraints（约束）

- 领域 Tool（evidence/verifier/calculator）**不 import LangGraph**，保持纯函数可单测；
- 无新增外部依赖（ast/math/signal 均为标准库）；
- `verify_flags` 结构与 Phase 3 动态路径契约**向后兼容**（`flags[sub_task]["passed"]` 语义不变）；
- Phase 1-3 全部 25 项测试回归通过；
- 全部新增结构支持 JSON 序列化（dataclass + asdict）。

## 3. Key Decisions

| # | 决策 | 理由 |
|---|---|---|
| D1 | 结果结构用 `@dataclass`（Evidence/VerificationResult/ExpressionResult），工具包装函数返回 `dict` | 与 ToolCall/ToolResult 风格一致；registry 输出与 state 写入可直接 JSON 序列化 |
| D2 | 指标词典按「最长同义词优先」扫描，命中即返回 canonical key | 避免 `revenue` 词表中的 `income` 抢在 `net income` 之前误匹配 |
| D3 | Entity 抽取：先内容首字母大写短语（过滤问句首词/年份词），回退 doc_name 前缀解析 | 测试文档 doc_name 常为 `nova-fy2024`（小写），内容才是权威来源 |
| D4 | Entity 匹配：大小写不敏感精确匹配 → 词边界包含匹配（`Nova` vs `NovaTech` 因边界失败） | 规范明确要求避免过度匹配 |
| D5 | Period 匹配：抽取双方 4 位年份集合取交集（FY2024 ⊇ 2024，Q2 2023 ⊇ 2023） | 覆盖范围包含语义，规则可解释 |
| D6 | `verify_flags[sub_task]` 值 = {claim, results[], passed_count, total_count, decision, passed, confidence} | 与 Phase 3 `flags[task]["passed"]` 跳过逻辑/动态路径证明兼容；results 供 synthesizer 精确绑定 |
| D7 | 计算器三层防御（AST 白名单 → `__builtins__={}` + math 白名单 → SIGALRM），并前置决定性 guard（长度>400、常量指数>1000 即拒） | 白名单已可拦注入；guard 让拒绝**确定性、快速**，测试不依赖真实超时竞态 |
| D8 | 节点经 ToolRegistry 调 `verify_claim_batch`，未注册时回退直接调用领域函数 | 保持 Phase 3「薄编排层」，同时节点单测不强制注册 |
| D9 | Replanner 基于 claim 定向重建子任务：`retrieve {entity} {metric} {period} specifically` | 指标错配后第二跳直接命中正确指标，优于 Phase 3 的 "broader terms" |
| D10 | 单元与结果保留浮点原始精度，测试用 `pytest.approx` 断言 | 避免无依据的舍入规则污染验证数值 |

## 4. 组件与接口

### 4.1 tools/evidence.py

```python
METRIC_KEYWORDS  # 7 组：revenue/operating_margin/gross_margin/net_margin/EBITDA/cost/profit
class EvidenceExtractor:
    def __init__(self, llm_mode: bool = False): ...
    def extract(self, chunk: dict) -> Evidence: ...
def extract_entity_candidates(text: str) -> List[str]   # 供 claim 抽取复用
```

### 4.2 tools/verifier.py

```python
def extract_claim(text: str) -> dict          # {entity, period, metric, definition}
class Verifier:
    def __init__(self, mode="rule"): ...
    def verify(self, claim: dict, evidence: Evidence) -> VerificationResult
    def verify_batch(self, claim, evidence_list) -> List[VerificationResult]
def verify_claim(claim: dict, chunk: dict) -> dict       # Tool 包装（单条）
def verify_claim_batch(claim: dict, chunks: List[dict]) -> dict  # Tool 包装（批量+决策）
```

### 4.3 tools/calculator.py

```python
class ExpressionCalculator:
    def __init__(self, timeout_seconds: float = 1.0): ...
    def evaluate(self, expression: str, context: Optional[dict] = None) -> ExpressionResult
def calc_expression(expr: str) -> dict        # Tool 包装
```

### 4.4 校验细则

| 要素 | 规则 |
|---|---|
| Entity | 大小写不敏感精确匹配；否则词边界包含（`(?<![a-z0-9])candidate(?![a-z0-9])`） |
| Period | 双方年份集合交集非空 |
| Metric | canonical key 相等（词典同义词归一） |
| Definition | token 交集 / min(token 数) ≥ 0.5（仅 claim 提供时校验） |
| 未抽取 | claim 提供而 evidence 缺失 → mismatch + missing 记录 |
| confidence | 命中要素数 / claim 提供要素数 |

## 5. Success Criteria（可验证）

- 新增 19 项测试通过（evidence 4 + verifier 7 + calculator 5 + 集成 3）；
- Phase 1-3 回归 25 项通过（合计 44）；
- 端到端演示：指标错配（revenue vs gross margin）被拦截 → 触发 REPLAN → 二次检索 ACCEPT；
- ruff（F401/I001）与 compileall 干净。验证证据：`pytest -q` 输出与演示轨迹。