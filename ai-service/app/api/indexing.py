from fastapi import APIRouter, BackgroundTasks
from app.schemas.indexing import IndexRequest, IndexResponse
from app.services.rag_service import rag_service

router = APIRouter(prefix="/indexing", tags=["indexing"])


@router.post("/index", response_model=IndexResponse)
async def index_content(req: IndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        rag_service.index_content,
        content_id=req.content_id,
        passage=req.passage,
        question=req.question,
        answer=req.model_answer,
    )
    return IndexResponse(status="indexing", content_id=req.content_id, chunks_indexed=0)
