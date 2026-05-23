from fastapi import APIRouter, HTTPException

from app.schemas.writing import WritingEvaluateRequest, WritingEvaluateResponse
from app.services.writing_service import evaluate_writing

router = APIRouter()


@router.post("/evaluate", response_model=WritingEvaluateResponse, summary="서술형 답안 자동 채점")
def evaluate(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    """
    학생의 서술형 답안을 분석하여 1~4점 척도로 채점하고 피드백을 반환합니다.

    - **raw_score**: 원점수 1~4점
    - **normalized_score**: 정규화 점수 (25 / 50 / 75 / 100)
    - **feedback**: 개선 방향 및 피드백 텍스트
    """
    try:
        return evaluate_writing(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
