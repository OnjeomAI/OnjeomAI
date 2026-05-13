import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.model import model_manager
from app.api import grading, tutor, curriculum, indexing


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

app.include_router(grading.router, prefix="/api")
app.include_router(tutor.router, prefix="/api")
app.include_router(curriculum.router, prefix="/api")
app.include_router(indexing.router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}
