from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from app.schemas.curriculum import CurriculumRequest, CurriculumResponse
from app.services.curriculum_service import curriculum_service

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


@router.post("/generate", response_model=CurriculumResponse)
async def generate_curriculum(req: CurriculumRequest):
    return await run_in_threadpool(
        curriculum_service.generate,
        theta=req.theta,
        daily_goal=req.daily_goal,
        weak_areas=req.weak_areas,
    )
