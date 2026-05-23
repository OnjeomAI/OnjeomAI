from fastapi import APIRouter, HTTPException

from app.schemas.writing import WritingEvaluateRequest, WritingEvaluateResponse
from app.services.writing_service import evaluate_writing

router = APIRouter()


@router.post("/evaluate", response_model=WritingEvaluateResponse, summary="서술형 답안 자동 채점")
def evaluate(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    """
    **전체 채점 파이프라인** (2단계 방식)

    **1단계** — 키워드 매칭 채점 (`keywords` 입력 시)
    - 관리자 등록 핵심 키워드 포함 여부로 기본 점수 산출
    - 누락/포함 키워드 목록 반환 (프론트 하이라이트용)

    **2단계** — LLM 채점 (`Onjeom/writing-ai`)
    - 지문 + 모범답안 + 학생 답안을 함께 분석하여 1~4점 부여
    - 문맥·논리 흐름 기반 최종 점수 조정

    **점수 구간별 피드백**
    - 80~100점: 심화 학습 추천
    - 50~79점: 보완 포인트 제시
    - 0~49점: 관련 학습 콘텐츠 링크 안내

    **오답 심층 분석** (final_score < 50 시 자동 제공)
    - Chain-of-Thought로 오류 유형 분류: 개념 혼동 / 어휘 부족 / 논리 비약 / 내용 누락
    """
    try:
        return evaluate_writing(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
