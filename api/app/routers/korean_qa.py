# 담당: 이성진
from fastapi import APIRouter, BackgroundTasks
from app.schemas.grading import GradeRequest, GradeResponse
from app.schemas.tutor import TutorRequest, TutorResponse, ExplainRequest, ExplainResponse
from app.schemas.curriculum import CurriculumRequest, CurriculumResponse
from app.schemas.indexing import IndexRequest, IndexResponse
from app.services.grading_service import grading_service
from app.services.rag_service import rag_service
from app.services.curriculum_service import curriculum_service
from app.core.model import model_manager

router = APIRouter(tags=["korean_qa"])


# ── 채점 ──────────────────────────────────────────────────────────
@router.post("/grading/grade", response_model=GradeResponse)
def grade_answer(req: GradeRequest):
    return grading_service.grade(
        passage=req.passage,
        question=req.question,
        model_answer=req.model_answer,
        keywords=[k.model_dump() for k in req.keywords],
        student_answer=req.student_answer,
    )


# ── AI 튜터 ───────────────────────────────────────────────────────
@router.post("/tutor/ask", response_model=TutorResponse)
def ask_tutor(req: TutorRequest):
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


@router.post("/tutor/explain", response_model=ExplainResponse)
def explain_term(req: ExplainRequest):
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


# ── 커리큘럼 ──────────────────────────────────────────────────────
@router.post("/curriculum/generate", response_model=CurriculumResponse)
def generate_curriculum(req: CurriculumRequest):
    return curriculum_service.generate(
        theta=req.theta,
        daily_goal=req.daily_goal,
        weak_areas=req.weak_areas,
    )


# ── 벡터 인덱싱 ───────────────────────────────────────────────────
@router.post("/indexing/index", response_model=IndexResponse)
def index_content(req: IndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        rag_service.index_content,
        content_id=req.content_id,
        passage=req.passage,
        question=req.question,
        answer=req.model_answer,
    )
    return IndexResponse(status="indexing", content_id=req.content_id, chunks_indexed=0)
