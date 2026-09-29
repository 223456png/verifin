FROM python:3.12-slim

# 演示锁定到 verifin 仓库的具体提交（更新 demo 时改这里后重建 Space）
ARG VERIFIN_REF=f1a3eb9

# HF Spaces 惯例：非 root 用户（UID 1000）
RUN useradd -m -u 1000 user

# git 仅构建期需要（pip 从 git+https 安装），运行期不留在镜像里也无害
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 核心包来自公开 GitHub 仓库（chromadb / langgraph / rank-bm25 等由核心依赖拉齐；
# fastapi / uvicorn 在 dev extras，显式补装）。不装 [bge] extras——
# 合成语料模式零模型下载，reranker 无 sentence-transformers 时按契约自动降级。
RUN pip install --no-cache-dir \
    "git+https://github.com/223456png/verifin.git@${VERIFIN_REF}" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.30.0"

COPY --chown=user:user web/index.html /app/web/index.html

# /app 需对运行用户可写：RetrievalEnvironment 会在 cwd 下建 .verifin_api_env 工作区
RUN chown user:user /app

USER user

# HF Docker Space 对外端口固定 7860；应用读环境变量绑定
ENV VERIFIN_HOST=0.0.0.0 \
    VERIFIN_PORT=7860

EXPOSE 7860

# 无 VERIFIN_INDEX_DIR → 内置合成语料（NovaTech/Orion/Vertex/Quantum 四公司），
# 零外部数据、零模型下载、确定性规则引擎（无 LLM key 依赖）
CMD ["python", "-m", "verifin.api.app"]
