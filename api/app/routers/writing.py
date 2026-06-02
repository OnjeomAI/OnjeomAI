# 담당: 김우주
from fastapi import APIRouter, HTTPException

from app.schemas.writing import (
    CompareAnswersRequest,
    CompareAnswersResponse,
    CurriculumAdjustRequest,
    CurriculumAdjustResponse,
    CurriculumPlanRequest,
    CurriculumPlanResponse,
    TermExplainRequest,
    TermExplainResponse,
    WeaknessReportRequest,
    WeaknessReportResponse,
    WritingEvaluateRequest,
    WritingEvaluateResponse,
)
from app.services.writing_service import (
    adjust_curriculum,
    compare_answers,
    evaluate_writing,
    explain_term,
    generate_curriculum_plan,
    generate_weakness_report,
)

router = APIRouter(tags=["writing"])


@router.post("/evaluate", response_model=WritingEvaluateResponse, summary="서술형 답안 자동 채점")
def evaluate(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    """
    **전체 채점 파이프라인** (2단계 방식)

    **1단계** — 키워드 매칭 채점 (`keywords` 입력 시)
    - 관리자 등록 핵심 키워드 포함 여부로 기본 점수 산출

    **2단계** — LLM 채점 (`Onjeom/writing-ai`)
    - 지문 + 모범답안 + 학생 답안을 함께 분석하여 1~4점 부여

    **점수 구간별 피드백**
    - 80~100점: 심화 학습 추천
    - 50~79점: 보완 포인트 제시
    - 0~49점: 관련 학습 콘텐츠 안내 + 심층 분석 자동 제공
    """
    try:
        return evaluate_writing(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/curriculum/adjust",
    response_model=CurriculumAdjustResponse,
    summary="동적 학습 경로 재조정",
)
def curriculum_adjust(req: CurriculumAdjustRequest) -> CurriculumAdjustResponse:
    """
    역량별 최근 점수 이력을 분석하여 취약 역량을 탐지하고 커리큘럼 재조정 메시지를 생성합니다.

    - 3회 연속 50점 미만인 역량을 취약 역량으로 판정
    """
    try:
        return adjust_curriculum(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/compare",
    response_model=CompareAnswersResponse,
    summary="답변 변화 추적",
)
def compare(req: CompareAnswersRequest) -> CompareAnswersResponse:
    """
    이전 답변과 현재 답변을 비교하여 성장 메시지와 키워드 변화를 반환합니다.

    - 새로 포함된 키워드 / 여전히 누락된 키워드 목록 제공
    """
    try:
        return compare_answers(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/weakness-report",
    response_model=WeaknessReportResponse,
    summary="약점 분석 리포트",
)
def weakness_report(req: WeaknessReportRequest) -> WeaknessReportResponse:
    """
    역량별 평균 점수를 분석하여 약점 리포트와 개선 권장사항을 생성합니다.

    - 50점 미만 역량: 취약 / 50~69점: 보통으로 분류
    """
    try:
        return generate_weakness_report(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/curriculum-plan",
    response_model=CurriculumPlanResponse,
    summary="커리큘럼 플랜 생성",
)
def curriculum_plan(req: CurriculumPlanRequest) -> CurriculumPlanResponse:
    """
    theta 기반 스테이지를 결정하고 취약 역량(50점 미만) reading_type 문제를 우선 배치합니다.

    - theta < -0.5 → 스테이지 [1]
    - -0.5 ≤ theta < 0.0 → 스테이지 [1, 2]
    - 0.0 ≤ theta < 0.5 → 스테이지 [2, 3]
    - theta ≥ 0.5 → 스테이지 [3, 4]
    """
    try:
        return generate_curriculum_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/explain-term",
    response_model=TermExplainResponse,
    summary="용어/문장 설명 (도움 기능)",
)
def explain(req: TermExplainRequest) -> TermExplainResponse:
    """
    이해가 안 되는 용어나 문장을 입력하면 AI가 쉬운 말로 설명합니다.

    - `passage_text` 제공 시 지문 맥락을 반영하여 더 정확한 설명 생성
    """
    try:
        return explain_term(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
