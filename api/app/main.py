from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.model_loader import get_writing_evaluator
from app.routers import writing


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 모델 로드
    get_writing_evaluator()
    yield


app = FastAPI(
    title="온점 AI API",
    description="국어 독해력 향상 서비스 AI 백엔드",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(writing.router, prefix="/api/writing", tags=["writing"])


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
