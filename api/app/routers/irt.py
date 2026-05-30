from fastapi import APIRouter, HTTPException

from app.schemas.irt import IrtEstimateRequest, IrtEstimateResponse
from app.services.irt_service import estimate_theta

router = APIRouter()


@router.post(
    "/estimate",
    response_model=IrtEstimateResponse,
    summary="IRT 기반 학생 능력 추정",
)
def irt_estimate(req: IrtEstimateRequest) -> IrtEstimateResponse:
    """
    학생의 문제 응답 이력을 바탕으로 **1PL IRT (Rasch 모델)**로 능력 수준을 추정합니다.

    - difficulty 1→b=-2, 2→b=-1, 3→b=0, 4→b=1, 5→b=2
    - score ≥ 50 → 정답(1), 미만 → 오답(0)
    - EAP(Expected A Posteriori): θ 격자(-4~4) × N(0,1) 사전분포 → 사후 기댓값
    """
    try:
        result = estimate_theta([r.model_dump() for r in req.responses])
        return IrtEstimateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
