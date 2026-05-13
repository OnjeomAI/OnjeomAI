from pydantic import BaseModel


class IndexRequest(BaseModel):
    content_id: str
    passage: str
    question: str
    model_answer: str
    keywords: list[str]


class IndexResponse(BaseModel):
    status: str
    content_id: str
    chunks_indexed: int
