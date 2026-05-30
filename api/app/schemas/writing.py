from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── 공통 ──────────────────────────────────────────────────────────────────────

class FeedbackType(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"


class ErrorType(str, Enum):
    CONCEPT_CONFUSION = "개념 혼동"
    VOCABULARY_LACK = "어휘 부족"
    LOGICAL_LEAP = "논리 비약"
    CONTENT_OMISSION = "내용 누락"


class Competency(str, Enum):
    FACTUAL = "factual"
    INFERENTIAL = "inferential"
    CRITICAL = "critical"
    VOCABULARY = "vocabulary"
    LOGICAL = "logical"


COMPETENCY_KO = {
    Competency.FACTUAL: "사실적 독해",
    Competency.INFERENTIAL: "추론적 독해",
    Competency.CRITICAL: "비판적 독해",
    Competency.VOCABULARY: "어휘력",
    Competency.LOGICAL: "논리 구조 파악",
}

class KeywordItem(BaseModel):
    keyword: str = Field(..., description="핵심 키워드")
    weight: int = Field(..., ge=1, le=100, description="배점 비중 (1~100)")


# ── POST /api/writing/evaluate ────────────────────────────────────────────────

class WritingEvaluateRequest(BaseModel):
    passage_text: str = Field(..., description="지문 텍스트")
    question_text: str = Field(..., description="문제 지시문")
    model_answer: str = Field(..., description="모범 답안")
    user_answer: str = Field(..., max_length=700, description="학생 답안 (최대 700자)")
    keywords: list[KeywordItem] = Field(
        default_factory=list,
        description="1단계 채점용 핵심 키워드 목록 (미입력 시 LLM 단독 채점)",
    )


class DeepAnalysis(BaseModel):
    error_types: list[ErrorType] = Field(..., description="오류 유형 분류")
    analysis: str = Field(..., description="오류 원인 상세 분석")
    improvement: str = Field(..., description="구체적 개선 방향")


class WritingEvaluateResponse(BaseModel):
    keyword_score: Optional[int] = Field(None, description="1단계 키워드 기반 점수 (0~100). 미입력 시 null")
    raw_score: int = Field(..., ge=1, le=4, description="LLM 원점수 (1~4점)")
    normalized_score: int = Field(..., description="LLM 정규화 점수 (25/50/75/100)")
    final_score: int = Field(..., ge=0, le=100, description="최종 점수 (0~100)")
    feedback: str = Field(..., description="LLM 채점 피드백")
    feedback_type: FeedbackType = Field(..., description="점수 구간 유형")
    score_feedback: str = Field(..., description="점수 구간별 안내 메시지")
    matched_keywords: list[str] = Field(default_factory=list, description="포함된 키워드 (초록 하이라이트용)")
    missing_keywords: list[str] = Field(default_factory=list, description="누락된 키워드 (빨강 하이라이트용)")
    deep_analysis: Optional[DeepAnalysis] = Field(None, description="오답 심층 분석 (50점 미만 시에만 반환)")


# ── POST /api/writing/curriculum/adjust ──────────────────────────────────────

class CompetencyHistory(BaseModel):
    competency: Competency = Field(..., description="역량 유형")
    scores: list[int] = Field(..., description="최근 점수 이력 (오래된 순)")


class CurriculumAdjustRequest(BaseModel):
    competency_history: list[CompetencyHistory] = Field(
        ..., description="역량별 최근 점수 이력"
    )


class CurriculumAdjustResponse(BaseModel):
    needs_adjustment: bool = Field(..., description="커리큘럼 재조정 필요 여부")
    weak_competencies: list[str] = Field(
        default_factory=list, description="취약 역량 목록 (3회 연속 50점 미만)"
    )
    adjustment_message: str = Field(..., description="사용자에게 보여줄 알림 메시지")
    recommended_focus: str = Field(..., description="LLM 생성 개선 방향")


# ── POST /api/writing/compare ────────────────────────────────────────────────

class CompareAnswersRequest(BaseModel):
    question_text: str = Field(..., description="문제 지시문")
    model_answer: str = Field(..., description="모범 답안")
    previous_answer: str = Field(..., max_length=700, description="이전 답변")
    previous_score: int = Field(..., ge=0, le=100, description="이전 점수")
    current_answer: str = Field(..., max_length=700, description="현재 답변")
    current_score: int = Field(..., ge=0, le=100, description="현재 점수")
    keywords: list[KeywordItem] = Field(default_factory=list, description="핵심 키워드")


class CompareAnswersResponse(BaseModel):
    score_diff: int = Field(..., description="점수 변화 (양수=향상, 음수=하락)")
    is_improved: bool = Field(..., description="성장 여부")
    growth_message: str = Field(..., description="성장 메시지 (예: 저번엔 이 키워드가 없었는데 이번엔 포함됐어요!)")
    newly_included_keywords: list[str] = Field(default_factory=list, description="새로 포함된 키워드")
    still_missing_keywords: list[str] = Field(default_factory=list, description="여전히 누락된 키워드")
    analysis: str = Field(..., description="LLM 생성 비교 분석")


# ── POST /api/writing/weakness-report ────────────────────────────────────────

class CompetencyScoreItem(BaseModel):
    competency: Competency = Field(..., description="역량 유형")
    score: int = Field(..., ge=0, le=100, description="평균 점수")


class WeaknessReportRequest(BaseModel):
    competency_scores: list[CompetencyScoreItem] = Field(
        ..., description="역량별 평균 점수 목록"
    )


class WeakCompetencyDetail(BaseModel):
    competency: str = Field(..., description="역량명 (한국어)")
    score: int = Field(..., description="평균 점수")
    level: str = Field(..., description="수준 (취약/보통/양호)")


class WeaknessReportResponse(BaseModel):
    weak_competencies: list[WeakCompetencyDetail] = Field(
        ..., description="취약 역량 목록 (50점 미만)"
    )
    report: str = Field(..., description="LLM 생성 약점 분석 리포트")
    recommendations: list[str] = Field(..., description="역량별 개선 권장사항")
    priority_competency: Optional[str] = Field(None, description="가장 집중해야 할 역량")


# ── POST /api/writing/irt/estimate ────────────────────────────────────────────

class IrtResponseItem(BaseModel):
    difficulty: int = Field(..., ge=1, le=5, description="문제 난이도 (1~5)")
    score: int = Field(..., ge=0, le=100, description="학생 점수 (0~100)")
    a_param: Optional[float] = Field(None, ge=0.1, le=4.0, description="3PL 변별도 (미입력 시 1.0 고정)")
    b_param: Optional[float] = Field(None, ge=-4.0, le=4.0, description="3PL 난이도 파라미터 (미입력 시 difficulty로 자동 매핑)")
    c_param: Optional[float] = Field(None, ge=0.0, le=0.5, description="3PL 추측도 (미입력 시 0.0 고정)")


class IrtEstimateRequest(BaseModel):
    responses: list[IrtResponseItem] = Field(
        ..., min_length=1, description="응답 이력 (최소 1개)"
    )


class IrtEstimateResponse(BaseModel):
    theta: float = Field(..., description="능력 추정치 (EAP, 표준화 척도 μ=0 σ=1)")
    se: float = Field(..., description="추정 표준오차")
    ability_level: str = Field(..., description="능력 수준 (하/중하/중/중상/상)")
    next_difficulty: int = Field(..., ge=1, le=5, description="권장 다음 문제 난이도 (1~5)")


# ── POST /api/writing/explain-term ───────────────────────────────────────────

class TermExplainRequest(BaseModel):
    term: str = Field(..., min_length=1, description="설명이 필요한 용어 또는 문장")
    passage_text: Optional[str] = Field(None, description="지문 텍스트 (맥락 기반 설명에 활용)")


class TermExplainResponse(BaseModel):
    term: str = Field(..., description="입력된 용어")
    explanation: str = Field(..., description="AI가 생성한 쉬운 설명")


# ── POST /api/writing/curriculum-plan ────────────────────────────────────────

class AvailableProblem(BaseModel):
    id: int = Field(..., description="문제 ID")
    difficulty: int = Field(..., ge=1, le=5, description="문제 난이도 (1~5)")
    reading_type: str = Field(..., description="독해 유형 (FACTUAL/INFERENTIAL/CRITICAL/CREATIVE)")


class CurriculumPlanRequest(BaseModel):
    theta: float = Field(..., description="IRT 능력 추정치")
    daily_goal: int = Field(..., ge=1, description="일일 목표 문제 수")
    competency_scores: dict[str, int] = Field(
        ..., description="역량별 점수 {역량명: 점수(0~100)}"
    )
    available_problems: list[AvailableProblem] = Field(
        ..., description="배정 가능한 전체 문제 목록"
    )


class CurriculumPlanResponse(BaseModel):
    plan: dict[int, list[int]] = Field(
        ..., description="스테이지별 문제 ID 목록 {stage: [problemId, ...]}"
    )
