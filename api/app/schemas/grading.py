from pydantic import BaseModel


class Keyword(BaseModel):
    keyword: str
    weight: int  # 배점 (합계 100 권장)


class GradeRequest(BaseModel):
    passage: str
    question: str
    model_answer: str
    keywords: list[Keyword]
    student_answer: str


class GradeResponse(BaseModel):
    score: int
    stage1_score: int
    found_keywords: list[str]
    missing_keywords: list[Keyword]
    feedback: str
    grade_reason: str
