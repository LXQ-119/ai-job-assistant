# 后端服务的容器镜像。
#
# 构建并运行：
#   docker build -t ai-job-assistant .
#   docker run --rm -p 8000:8000 --env-file .env ai-job-assistant
#
# 注意：镜像里**不包含 .env**，密钥在运行时通过 --env-file 注入。
# 这是容器化的基本纪律——密钥永远不进镜像层，否则 push 到仓库就泄露了。

FROM python:3.12-slim

# 让 Python 不要写 .pyc、日志实时输出（不要缓冲）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先只拷依赖清单再安装，这样改代码不会让依赖层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

# 容器里必须监听 0.0.0.0，监听 127.0.0.1 的话宿主机访问不到
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
