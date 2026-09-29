---
title: VeriFin — 金融证据校验 Agent
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VeriFin — 金融证据校验 Agent Demo

> 检索只是输入，输出是**可验证、可溯源的 Claim-Evidence 绑定**：证据在实体 / 时间 / 指标 / 口径四要素上不匹配时，自动诊断并重规划（Replan）重试。本 Demo 为确定性规则引擎（无 LLM 调用、无外部依赖）。

**内置合成语料**：NovaTech / Orion / Vertex / Quantum 四家公司财报，含一组专门设计的**口径冲突**（新闻稿 vs 年报合并报表）用于演示冲突仲裁与重规划。

## 可以问它什么

- `What was NovaTech's revenue in 2024?`（普通取证）
- `What is NovaTech's revenue growth rate from 2023 to 2024?`（跨年增长计算）
- `What is Orion's gross margin in 2023?`（指标口径校验）
- 追问 `and its operating margin?`（多轮偏好记忆）

页面会可视化 Agent 的完整轨迹：检索 → 四要素校验 → 证据卡片（含冲突仲裁）→ 计算过程 → 答案。

## 工程

- 镜像从 [GitHub 仓库](https://github.com/223456png/verifin) 按提交锁定构建（见 Dockerfile 的 `VERIFIN_REF`）
- 生产评测（FinQA 真实数据、LLM 程序生成、消融实验）见仓库 `docs/EVALUATION.md`
