from typing import Optional

from pydantic import BaseModel, Field


class IrtResponseItem(BaseModel):
    difficulty: int = Field(..., ge=1, le=5, description="문제 난이도 (1~5)")
    score: int = Field(..., ge=0, le=100, description="학생 점수 (0~100)")
    a_param: Optional[float] = Field(None, ge=0.1, le=4.0, description="3PL 변별도 (미입력 시 1.0 고정)")
    b_param: Optional[float] = Field(None, ge=-4.0, le=4.0, description="3PL 난이도 파라미터 (미입력 시 difficulty로 자동 매핑)")
    c_param: Optional[float] = Field(None, ge=0.0, le=0.5, description="3PL 추측도 (미입력 시 0.0 고정)")


class IrtEstimateRequest(BaseModel):
    responses: list[IrtResponseItem] = Field(
        ..., min_length=1, description="응답 이력 (최소 1개)"
    )


class IrtEstimateResponse(BaseModel):
    theta: float = Field(..., description="능력 추정치 (EAP, 표준화 척도 μ=0 σ=1)")
    se: float = Field(..., description="추정 표준오차")
    ability_level: str = Field(..., description="능력 수준 (하/중하/중/중상/상)")
    next_difficulty: int = Field(..., ge=1, le=5, description="권장 다음 문제 난이도 (1~5)")
