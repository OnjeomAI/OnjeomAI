from pydantic import BaseModel
from typing import Optional


class GeneratedKeyword(BaseModel):
    keyword: str
    weight: int


class ProblemGenerateRequest(BaseModel):
    difficulty: int
    reading_type: str
    topic: Optional[str] = None


class ProblemGenerateResponse(BaseModel):
    passage_text: str
    question_text: str
    model_answer: str
    reading_type: str
    difficulty: int
    keywords: list[GeneratedKeyword] = []
