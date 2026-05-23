from fastapi import APIRouter, HTTPException

from app.schemas.writing import (
    CompareAnswersRequest,
    CompareAnswersResponse,
    CurriculumAdjustRequest,
    CurriculumAdjustResponse,
    IrtEstimateRequest,
    IrtEstimateResponse,
    WeaknessReportRequest,
    WeaknessReportResponse,
    WritingEvaluateRequest,
    WritingEvaluateResponse,
)
from app.services.irt_service import estimate_theta
from app.services.writing_service import (
    adjust_curriculum,
    compare_answers,
    evaluate_writing,
    generate_weakness_report,
)

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


@router.post(
    "/curriculum/adjust",
    response_model=CurriculumAdjustResponse,
    summary="동적 학습 경로 재조정",
)
def curriculum_adjust(req: CurriculumAdjustRequest) -> CurriculumAdjustResponse:
    """
    역량별 최근 점수 이력을 분석하여 취약 역량을 탐지하고 커리큘럼 재조정 메시지를 생성합니다.

    - 3회 연속 50점 미만인 역량을 취약 역량으로 판정
    - 취약 역량이 있으면 LLM이 맞춤형 재조정 메시지와 개선 방향을 생성
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
    - LLM이 두 답변의 변화를 분석하여 격려 메시지 생성
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
    - 가장 점수가 낮은 역량을 우선 집중 역량으로 지정
    - LLM이 2~3문장의 약점 분석 리포트 생성
    """
    try:
        return generate_weakness_report(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/irt/estimate",
    response_model=IrtEstimateResponse,
    summary="IRT 기반 학생 능력 추정",
)
def irt_estimate(req: IrtEstimateRequest) -> IrtEstimateResponse:
    """
    학생의 문제 응답 이력을 바탕으로 **1PL IRT (Rasch 모델)**로 능력 수준을 추정합니다.

    **파라미터 매핑 (DB 스키마 변경 없음)**
    - difficulty 1→b=-2, 2→b=-1, 3→b=0, 4→b=1, 5→b=2
    - a=1.0 고정 (변별도), c=0.0 고정 (서술형이므로 추측 없음)

    **채점 이진화**
    - score ≥ 50 → 정답(1), 미만 → 오답(0)

    **알고리즘**
    - EAP(Expected A Posteriori): θ 격자(-4~4) × N(0,1) 사전분포 → 사후 기댓값

    **반환값**
    - `theta`: EAP 추정치 (표준화 척도, 평균 0·표준편차 1)
    - `se`: 추정 표준오차 (낮을수록 정확)
    - `ability_level`: 하/중하/중/중상/상
    - `next_difficulty`: 현재 능력에 맞는 권장 난이도 (1~5)
    """
    try:
        result = estimate_theta([r.model_dump() for r in req.responses])
        return IrtEstimateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
