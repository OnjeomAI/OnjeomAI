from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from app.schemas.tutor import TutorRequest, TutorResponse, ExplainRequest, ExplainResponse
from app.services.rag_service import rag_service
from app.core.model import model_manager

router = APIRouter(prefix="/tutor", tags=["tutor"])


def _ask(req: TutorRequest) -> TutorResponse:
    results = rag_service.search(req.question)
    context = "\n\n".join(r["text"] for r in results)
    sources = list({r["metadata"].get("content_id", "") for r in results})

    context_block = f"\n\n[참고 자료]\n{context}" if context else ""
    prompt = f"""국어 독해 전문가로서 질문에 답하세요.{context_block}

[질문]
{req.question}"""

    messages = [
        {"role": "system", "content": "당신은 국어 독해 전문 AI 튜터입니다. 주어진 자료에 근거하여 답변하세요."},
        {"role": "user", "content": prompt},
    ]
    answer = model_manager.generate(messages, max_new_tokens=512)
    return TutorResponse(answer=answer, sources=sources)


def _explain(req: ExplainRequest) -> ExplainResponse:
    context_block = f"\n\n[문맥]\n{req.context}" if req.context else ""
    prompt = f"""중학생도 이해할 수 있게 쉽게 설명해주세요.{context_block}

[설명 요청]
{req.term}"""

    messages = [
        {"role": "system", "content": "당신은 국어를 가르치는 친절한 선생님입니다."},
        {"role": "user", "content": prompt},
    ]
    explanation = model_manager.generate(messages, max_new_tokens=256)
    return ExplainResponse(explanation=explanation)


@router.post("/ask", response_model=TutorResponse)
async def ask_tutor(req: TutorRequest):
    return await run_in_threadpool(_ask, req)


@router.post("/explain", response_model=ExplainResponse)
async def explain_term(req: ExplainRequest):
    return await run_in_threadpool(_explain, req)
