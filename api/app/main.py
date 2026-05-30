import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model import model_manager
from app.core.model_loader import get_writing_evaluator
from app.routers import korean_qa, writing, irt


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        print("모델 로딩 중...")
        model_manager.load()
        get_writing_evaluator()
        print("모델 로딩 완료!")
    else:
        print("모델 로딩 건너뜀 (SKIP_MODEL_LOAD=1)")
    yield


app = FastAPI(
    title="온점 AI API",
    description="국어 독해력 향상 서비스 AI 백엔드",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(korean_qa.router, prefix="/api")
app.include_router(writing.router, prefix="/api/writing", tags=["writing"])
app.include_router(irt.router, prefix="/api/writing/irt", tags=["irt"])


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
