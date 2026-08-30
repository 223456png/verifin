# VeriFin 快捷命令（make demo / make test / make benchmark / ...）
PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8000

.PHONY: help install demo test benchmark ablation index multi-turn clean

help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖（pip install -e ".[dev]"）
	$(PIP) install -e ".[dev]" 2>/dev/null || pip install -e ".[dev]"

demo: ## 一键启动 Web Demo（http://127.0.0.1:$(PORT)）
	@echo ">>> 启动 VeriFin Web Demo: http://127.0.0.1:$(PORT)"
	@echo ">>> 内置合成语料（NovaTech/Orion/Vertex/Quantum），Ctrl+C 停止"
	$(PY) -m verifin.api.app

demo-finqa: ## 用 FinQA 索引启动（需先 make index）
	@echo ">>> 启动 VeriFin（FinQA 索引）: http://127.0.0.1:$(PORT)"
	VERIFIN_INDEX_DIR=./indexes $(PY) -m verifin.api.app

test: ## 运行全部测试（126 项）
	$(PY) -m pytest tests/ -q

index: ## 构建 FinQA 索引（data/finqa → indexes/）
	$(PY) scripts/build_index.py --dataset finqa --data-dir ./data/finqa

benchmark: ## FinQA 端到端评测（100 样本）
	$(PY) scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results

ablation: ## 消融实验（合成套件，5 配置 × 20 样本）
	$(PY) scripts/run_benchmark.py --dataset synthetic --ablation --multi-turn --max-samples 20 --output ./results

multi-turn: ## ConvFinQA 多轮评测
	$(PY) scripts/run_benchmark.py --dataset convfinqa --multi-turn --output ./results

clean: ## 清理运行期产物（索引/缓存）
	rm -rf .verifin_api_env chroma_db indexes verifin_metadata.db .pytest_cache
