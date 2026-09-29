# VeriFin 快捷命令（make demo / make test / make benchmark / ...）
PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8000
INDEX_DIR ?= ./bge_env

.PHONY: help install lint test cover demo demo-finqa mcp index benchmark benchmark-finqa \
        ablation multi-turn multihop multihop-synth mcp-smoke up down clean

help: ## 显示所有可用命令
        @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
                awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖（pip install -e ".[dev]"）
        $(PIP) install -e ".[dev]"

lint: ## 静态检查（与 CI 同款）
        ruff check src tests

test: ## 运行全部测试（225 项）
        $(PY) -m pytest tests/ -q

cover: ## 运行测试并输出覆盖率
        $(PY) -m pytest tests/ -q --cov=verifin --cov-report=term-missing

demo: ## 一键启动 Web Demo（http://127.0.0.1:$(PORT)）
        @echo ">>> 启动 VeriFin Web Demo: http://127.0.0.1:$(PORT)"
        @echo ">>> 内置合成语料（NovaTech/Orion/Vertex/Quantum），Ctrl+C 停止"
        $(PY) -m verifin.api.app

demo-finqa: ## 用 FinQA 索引启动（需先 make index）
        @echo ">>> 启动 VeriFin（FinQA 索引）: http://127.0.0.1:$(PORT)"
        VERIFIN_INDEX_DIR=./indexes $(PY) -m verifin.api.app

mcp: ## 启动 MCP Server（stdio，供 Claude Desktop / Cursor 等客户端连接）
        VERIFIN_INDEX_DIR=$(INDEX_DIR) $(PY) -m verifin.mcp.server

mcp-smoke: ## MCP Server 协议级冒烟测试（需已构建索引）
        VERIFIN_INDEX_DIR=$(INDEX_DIR) HF_HUB_OFFLINE=1 $(PY) scripts/test_mcp_server.py

index: ## 构建 FinQA 索引（data/finqa → indexes/）
        $(PY) scripts/build_index.py --dataset finqa --data-dir ./data/finqa

benchmark: ## FinQA 端到端评测（100 样本，hash 索引离线保底）
        $(PY) scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results

benchmark-finqa: ## FinQA 端到端评测（bge 索引，doc recall 最优路径）
# 需先构建 bge 索引（见 scripts/compare_retrieval.py 头部文档）：
#   HF_ENDPOINT=https://hf-mirror.com $(PY) scripts/build_index.py \
#     --dataset finqa --data-dir ./data/finqa --embedding bge --persist-dir ./bge_env
# 查询侧需 sentence-transformers 模型缓存；沙箱环境重置后首次运行用
# HF_ENDPOINT=https://hf-mirror.com make benchmark-finqa 重新拉取。
        HF_HUB_OFFLINE=1 $(PY) scripts/run_benchmark.py --dataset finqa \
                --data-dir ./data/finqa --max-samples 100 --index-dir ./bge_env \
                --output ./results

ablation: ## 消融实验（合成套件，5 配置 × 20 样本）
        $(PY) scripts/run_benchmark.py --dataset synthetic --ablation --multi-turn --max-samples 20 --output ./results

multi-turn: ## ConvFinQA 多轮评测
        $(PY) scripts/run_benchmark.py --dataset convfinqa --multi-turn --output ./results

multihop: ## 多跳合成基准评测（16 样本：种子抽取 + 四重校验产出）
        $(PY) scripts/run_benchmark.py --dataset synthetic_multihop --output ./results/multihop_check

multihop-synth: ## 多跳 QA 合成管线（产出基准 JSON + 校验统计）
        $(PY) scripts/synthesize_multihop.py --output ./results/multihop_synth

up: ## Docker Compose 启动（http://127.0.0.1:$(PORT)）
        docker compose up --build -d

down: ## 停止 Docker Compose 服务
        docker compose down

clean: ## 清理运行期产物（索引/缓存）
        rm -rf .verifin_api_env chroma_db indexes verifin_metadata.db .pytest_cache .ruff_cache .coverage htmlcov
