from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from app.schemas.grading import GradeRequest, GradeResponse
from app.services.grading_service import grading_service

router = APIRouter(prefix="/grading", tags=["grading"])


@router.post("/grade", response_model=GradeResponse)
async def grade_answer(req: GradeRequest):
    return await run_in_threadpool(
        grading_service.grade,
        passage=req.passage,
        question=req.question,
        model_answer=req.model_answer,
        keywords=[k.model_dump() for k in req.keywords],
        student_answer=req.student_answer,
    )
