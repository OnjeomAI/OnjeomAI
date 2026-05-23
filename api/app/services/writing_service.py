from app.core.model_loader import get_writing_evaluator
from app.schemas.writing import (
    DeepAnalysis,
    ErrorType,
    FeedbackType,
    KeywordItem,
    WritingEvaluateRequest,
    WritingEvaluateResponse,
)


# ── 1단계: 키워드 채점 ────────────────────────────────────────────────────────

def _calc_keyword_score(
    user_answer: str,
    keywords: list[KeywordItem],
) -> tuple[int, list[str], list[str]]:
    """
    핵심 키워드 포함 여부로 기본 점수 산출.

    Returns:
        (score 0~100, matched_keywords, missing_keywords)
    """
    if not keywords:
        return 0, [], []

    total_weight = sum(k.weight for k in keywords)
    if total_weight == 0:
        return 0, [], []

    matched, missing = [], []
    earned = 0

    for kw in keywords:
        if kw.keyword in user_answer:
            matched.append(kw.keyword)
            earned += kw.weight
        else:
            missing.append(kw.keyword)

    score = round(earned / total_weight * 100)
    return score, matched, missing


# ── 점수 구간별 피드백 ────────────────────────────────────────────────────────

def _score_feedback(final_score: int) -> tuple[FeedbackType, str]:
    """요구사항: 80~100 / 50~79 / 0~49 구간별 안내 메시지."""
    if final_score >= 80:
        return (
            FeedbackType.EXCELLENT,
            "핵심을 잘 파악했어요! 더 심화된 내용을 학습해볼까요?",
        )
    elif final_score >= 50:
        return (
            FeedbackType.GOOD,
            "방향은 맞지만 일부 내용이 빠졌어요. 보완 포인트를 확인해보세요.",
        )
    else:
        return (
            FeedbackType.NEEDS_IMPROVEMENT,
            "이 개념부터 다시 보세요. 관련 학습 콘텐츠를 추천해드릴게요.",
        )


# ── 최종 점수 산출 ────────────────────────────────────────────────────────────

def _calc_final_score(keyword_score: int | None, normalized_score: int) -> int:
    """
    키워드 점수가 있으면 키워드(40%) + LLM(60%) 가중 평균.
    없으면 LLM 점수 그대로 사용.
    """
    if keyword_score is None:
        return normalized_score
    return round(keyword_score * 0.4 + normalized_score * 0.6)


# ── 메인 서비스 ───────────────────────────────────────────────────────────────

def evaluate_writing(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    """
    전체 채점 파이프라인:
    1단계 키워드 채점 → 2단계 LLM 채점 → 최종 점수 산출
    → 점수 구간별 피드백 → 50점 미만 심층 분석
    """
    evaluator = get_writing_evaluator()

    # 1단계: 키워드 채점
    has_keywords = bool(req.keywords)
    if has_keywords:
        kw_score, matched, missing = _calc_keyword_score(req.user_answer, req.keywords)
    else:
        kw_score, matched, missing = None, [], []

    # 2단계: LLM 채점
    llm_result = evaluator.evaluate(
        passage_text=req.passage_text,
        question_text=req.question_text,
        model_answer=req.model_answer,
        user_answer=req.user_answer,
    )

    # 최종 점수
    final_score = _calc_final_score(kw_score, llm_result["normalized_score"])
    feedback_type, score_feedback = _score_feedback(final_score)

    # 심층 분석 (50점 미만)
    deep = None
    if final_score < 50:
        raw = evaluator.deep_analysis(
            question_text=req.question_text,
            model_answer=req.model_answer,
            user_answer=req.user_answer,
            score=final_score,
        )
        valid_types = {e.value for e in ErrorType}
        deep = DeepAnalysis(
            error_types=[
                ErrorType(t) for t in raw["error_types"] if t in valid_types
            ] or [ErrorType.CONTENT_OMISSION],
            analysis=raw["analysis"],
            improvement=raw["improvement"],
        )

    return WritingEvaluateResponse(
        keyword_score=kw_score,
        raw_score=llm_result["raw_score"],
        normalized_score=llm_result["normalized_score"],
        final_score=final_score,
        feedback=llm_result["feedback"],
        feedback_type=feedback_type,
        score_feedback=score_feedback,
        matched_keywords=matched,
        missing_keywords=missing,
        deep_analysis=deep,
    )
