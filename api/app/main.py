import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.model import model_manager
from app.routers import korean_qa, writing  # korean_qa: 이성진 / writing: 김우주


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("SKIP_MODEL_LOAD") != "1":
        print("모델 로딩 중...")
        model_manager.load()
        print("모델 로딩 완료!")
    else:
        print("모델 로딩 건너뜀 (SKIP_MODEL_LOAD=1)")
    yield


app = FastAPI(title="온점 AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(korean_qa.router, prefix="/api")  # 담당: 이성진
app.include_router(writing.router, prefix="/api")    # 담당: 김우주


@app.get("/health")
def health_check():
    return {"status": "ok"}
