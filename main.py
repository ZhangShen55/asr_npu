from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.models import load_models_if_needed
from app.api.routes.asr import router as asr_router
from app.api.routes.audio import router as audio_router
from app.api.routes.status import router as status_router
from app.api.routes.text import router as text_router
from app.api.routes.ws_online import router as ws_router
from app.utils.asr_stats import reset_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 日志初始化
    setup_logging(settings.log_path)

    # 2) 重置统计
    reset_stats()

    # 3) 记录启动时刻
    app.state.run_start_time = datetime.utcnow()

    # 4) 懒加载需要的模型（遵循配置开关）
    await load_models_if_needed()

    yield
    # 这里可按需释放资源（本项目暂不需要）


def create_app() -> FastAPI:
    app = FastAPI(title="SeaCraftASR", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由装配（路径与契约保持不变）
    app.include_router(status_router)
    app.include_router(asr_router)
    app.include_router(audio_router)
    app.include_router(text_router)
    app.include_router(ws_router)

    return app


app = create_app()

# 方便本地调试：容器里用 start.sh，多进程由 Nginx + 多 uvicorn 实例负责
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
    # sudo apt install ffmpeg

    # 启动方式
    # 格式：uvicorn 包名.文件名:实例名 参数
    # ASCEND_RT_VISIBLE_DEVICES=7 uvicorn app.main:app --host 0.0.0.0 --port 8083 --workers 1

