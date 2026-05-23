from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    EXCELLENT = "EXCELLENT"           # 80~100점
    GOOD = "GOOD"                     # 50~79점
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"  # 0~49점


class ErrorType(str, Enum):
    CONCEPT_CONFUSION = "개념 혼동"
    VOCABULARY_LACK = "어휘 부족"
    LOGICAL_LEAP = "논리 비약"
    CONTENT_OMISSION = "내용 누락"


class KeywordItem(BaseModel):
    keyword: str = Field(..., description="핵심 키워드")
    weight: int = Field(..., ge=1, le=100, description="배점 비중 (1~100)")


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
    # ── 점수 ──────────────────────────────────────────
    keyword_score: Optional[int] = Field(
        None, description="1단계 키워드 기반 점수 (0~100). 키워드 미입력 시 null"
    )
    raw_score: int = Field(..., ge=1, le=4, description="LLM 원점수 (1~4점)")
    normalized_score: int = Field(..., description="LLM 정규화 점수 (25/50/75/100)")
    final_score: int = Field(..., ge=0, le=100, description="최종 점수 (0~100)")

    # ── 피드백 ────────────────────────────────────────
    feedback: str = Field(..., description="LLM 채점 피드백")
    feedback_type: FeedbackType = Field(..., description="점수 구간 유형")
    score_feedback: str = Field(..., description="점수 구간별 안내 메시지")

    # ── 키워드 분석 ───────────────────────────────────
    matched_keywords: list[str] = Field(
        default_factory=list, description="포함된 핵심 키워드 (초록 하이라이트용)"
    )
    missing_keywords: list[str] = Field(
        default_factory=list, description="누락된 핵심 키워드 (빨강 하이라이트용)"
    )

    # ── 심층 분석 (final_score < 50 시에만 반환) ──────
    deep_analysis: Optional[DeepAnalysis] = Field(
        None, description="오답 심층 분석 (50점 미만 답변에만 제공)"
    )
