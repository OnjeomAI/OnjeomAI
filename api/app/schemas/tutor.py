from pydantic import BaseModel
from typing import Optional


class TutorRequest(BaseModel):
    question: str
    context: Optional[str] = None


class TutorResponse(BaseModel):
    answer: str
    sources: list[str]


class ExplainRequest(BaseModel):
    term: str
    context: Optional[str] = None


class ExplainResponse(BaseModel):
    explanation: str
