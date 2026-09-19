# syntax=docker/dockerfile:1
#
# VeriFin 运行镜像
#
# 构建：  docker build -t verifin:latest .
# 运行：  docker run --rm -p 8000:8000 verifin:latest
#
# 镜像内使用内置合成语料，无需挂载任何数据集即可启动 Web Demo。
# 若需使用真实 FinQA 索引，把索引目录挂载进容器并通过 VERIFIN_INDEX_DIR 指向它：
#   docker run --rm -p 8000:8000 -v $PWD/bge_env:/app/bge_env \
#     -e VERIFIN_INDEX_DIR=/app/bge_env verifin:latest

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 镜像内默认离线：不联网拉取 HuggingFace 模型（避免启动时 SSL 阻塞）
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VERIFIN_HOST=0.0.0.0 \
    VERIFIN_PORT=8000

WORKDIR /app

# curl 仅用于 HEALTHCHECK
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用层缓存：仅当 pyproject 变化时才重装）
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip \
 && pip install .

# 运行期需要的辅助文件
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY Makefile .env.example ./

# 非 root 运行
RUN useradd --create-home --shell /bin/bash verifin \
 && chown -R verifin:verifin /app
USER verifin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${VERIFIN_PORT}/health" || exit 1

CMD ["python", "-m", "verifin.api.app"]
