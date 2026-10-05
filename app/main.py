"""FastAPI 应用入口。

启动方式：
    python -m app.main
或者：
    uvicorn app.main:app --reload

启动后打开 http://127.0.0.1:8000/docs 就能看到自动生成的接口文档。
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# 拦截「直接运行这个文件」的情况，给出人话提示
# ---------------------------------------------------------------------------
# app/ 是一个**包**（package），里面的模块用相对导入（from .xxx import yyy）。
# 相对导入要求**按包的方式运行**：python -m app.main
#
# 如果直接 `python app/main.py`，或者 VS Code 用「当前打开的文件」调试配置，
# Python 会把 main.py 当成独立脚本，此时 __package__ 是空的，相对导入就会报：
#     ImportError: attempted relative import with no known parent package
# 这个报错对新手极不友好，所以提前拦下来，直接告诉他该怎么做。
if __package__ in (None, ""):
    raise SystemExit(
        "\n"
        "请不要直接运行 app/main.py —— 它属于 app 包，必须按包的方式启动。\n"
        "\n"
        "正确做法（任选一种）：\n"
        "  1) 命令行：   python -m app.main\n"
        "  2) VS Code：  按 F5，选「1 后端 FastAPI（端口 8000）」\n"
        "                * 不要选「5 当前打开的文件」，那会把它当独立脚本跑\n"
        "\n"
    )

from .api.routes import router
from .config import get_settings, is_peak_now, mask_secret
from .llm import LLMError
from .services import rag
from .services.retriever import KB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ai-job-assistant")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI 求职助手 API",
        description=(
            "简历结构化 + 面经知识库问答。\n\n"
            "所有接口都可以在 /docs 里直接试跑。"
        ),
        version="0.1.0",
    )

    # 本地开发时 Streamlit(8501) 和 API(8000) 不同源，必须放开 CORS。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        # 启动时只打印必要信息，密钥永远打码
        logger.info(
            "服务启动 base_url=%s model=%s api_key=%s",
            settings.llm_base_url,
            settings.llm_model,
            mask_secret(settings.llm_api_key),
        )
        logger.info("当前计价时段：%s", "高峰（单价 x2）" if is_peak_now() else "非高峰")
        if not settings.llm_api_key:
            logger.warning(
                "没有配置 LLM_API_KEY，模型相关接口会失败。"
                "请把 .env.example 复制成 .env 并填入 Key。"
            )
        if rag.load_index():
            logger.info("已从磁盘恢复索引，共 %d 个块", len(KB))
        else:
            logger.info("未找到已有索引，可调用 POST /kb/index 建立")

    @app.exception_handler(LLMError)
    async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        """把模型异常统一转成 502，避免前端看到 500 堆栈。"""
        logger.warning("模型调用异常：%s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": "AI 求职助手 API",
            "docs": "/docs",
            "health": "/health",
            "ui": "python -m streamlit run ui/app.py",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    # reload=True 会让 uvicorn 额外 fork 一个 reloader 进程来做热重载。
    # 这个便利在受限环境下会直接失败——子进程创建被拦截（DSH 沙箱就是如此，
    # 表现为服务器根本没起来，而且看不到任何报错）。
    # 所以**默认关闭**，需要热重载时显式打开：
    #     $env:RELOAD = "1"; python -m app.main
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    port = int(os.getenv("PORT", "8000"))

    if reload_enabled:
        print("热重载已开启（会额外启动一个 reloader 进程）")
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=reload_enabled)
