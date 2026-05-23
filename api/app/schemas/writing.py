from pydantic import BaseModel, Field


class WritingEvaluateRequest(BaseModel):
    passage_text: str = Field(..., description="지문 텍스트")
    question_text: str = Field(..., description="문제 지시문")
    model_answer: str = Field(..., description="모범 답안")
    user_answer: str = Field(..., max_length=700, description="학생 답안 (최대 700자)")


class WritingEvaluateResponse(BaseModel):
    raw_score: int = Field(..., ge=1, le=4, description="원점수 (1~4점)")
    normalized_score: int = Field(..., description="정규화 점수 (25/50/75/100)")
    feedback: str = Field(..., description="채점 피드백")
