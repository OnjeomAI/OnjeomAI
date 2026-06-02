from fastapi import APIRouter, HTTPException

from app.writing.schemas.writing import (
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
from app.writing.services.writing_service import (
    adjust_curriculum,
    compare_answers,
    evaluate_writing,
    explain_term,
    generate_curriculum_plan,
    generate_weakness_report,
)

router = APIRouter()


@router.post("/evaluate", response_model=WritingEvaluateResponse, summary="서술형 답안 자동 채점")
def evaluate(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    try:
        return evaluate_writing(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/curriculum/adjust", response_model=CurriculumAdjustResponse, summary="동적 학습 경로 재조정")
def curriculum_adjust(req: CurriculumAdjustRequest) -> CurriculumAdjustResponse:
    try:
        return adjust_curriculum(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareAnswersResponse, summary="답변 변화 추적")
def compare(req: CompareAnswersRequest) -> CompareAnswersResponse:
    try:
        return compare_answers(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weakness-report", response_model=WeaknessReportResponse, summary="약점 분석 리포트")
def weakness_report(req: WeaknessReportRequest) -> WeaknessReportResponse:
    try:
        return generate_weakness_report(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/curriculum-plan", response_model=CurriculumPlanResponse, summary="커리큘럼 플랜 생성")
def curriculum_plan(req: CurriculumPlanRequest) -> CurriculumPlanResponse:
    try:
        return generate_curriculum_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-term", response_model=TermExplainResponse, summary="용어/문장 설명")
def explain(req: TermExplainRequest) -> TermExplainResponse:
    try:
        return explain_term(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
