import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model import model_manager
from app.core.model_loader import get_writing_evaluator
from app.korean_qa.router import router as korean_qa_router
from app.writing.router import router as writing_router
from app.writing.irt_router import router as irt_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        print("모델 로딩 중...")
        model_manager.load()       # korean_qa (Qwen2.5-3B + LoRA)
        get_writing_evaluator()    # writing (Llama-3.1-8B, MOCK_MODEL=true 시 mock)
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

app.include_router(korean_qa_router, prefix="/api")
app.include_router(writing_router, prefix="/api/writing", tags=["writing"])
app.include_router(irt_router, prefix="/api/writing/irt", tags=["irt"])


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
