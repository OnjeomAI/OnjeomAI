from app.core.model_loader import get_writing_evaluator
from app.writing.schemas.writing import (
    COMPETENCY_KO,
    AvailableProblem,
    Competency,
    CompareAnswersRequest,
    CompareAnswersResponse,
    CurriculumAdjustRequest,
    CurriculumAdjustResponse,
    CurriculumPlanRequest,
    CurriculumPlanResponse,
    DeepAnalysis,
    ErrorType,
    FeedbackType,
    KeywordItem,
    TermExplainRequest,
    TermExplainResponse,
    WeakCompetencyDetail,
    WeaknessReportRequest,
    WeaknessReportResponse,
    WritingEvaluateRequest,
    WritingEvaluateResponse,
)


# ── 1단계: 키워드 채점 ────────────────────────────────────────────────────────

def _calc_keyword_score(
    user_answer: str,
    keywords: list[KeywordItem],
) -> tuple[int, list[str], list[str]]:
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
    if keyword_score is None:
        return normalized_score
    return round(keyword_score * 0.4 + normalized_score * 0.6)


# ── 메인 서비스 ───────────────────────────────────────────────────────────────

def evaluate_writing(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    evaluator = get_writing_evaluator()

    has_keywords = bool(req.keywords)
    if has_keywords:
        kw_score, matched, missing = _calc_keyword_score(req.user_answer, req.keywords)
    else:
        kw_score, matched, missing = None, [], []

    llm_result = evaluator.evaluate(
        passage_text=req.passage_text,
        question_text=req.question_text,
        model_answer=req.model_answer,
        user_answer=req.user_answer,
    )

    final_score = _calc_final_score(kw_score, llm_result["normalized_score"])
    feedback_type, score_feedback = _score_feedback(final_score)

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


# ── 동적 학습 경로 재조정 ─────────────────────────────────────────────────────

def adjust_curriculum(req: CurriculumAdjustRequest) -> CurriculumAdjustResponse:
    evaluator = get_writing_evaluator()

    weak: list[str] = []
    for history in req.competency_history:
        recent = history.scores[-3:] if len(history.scores) >= 3 else history.scores
        if recent and all(s < 50 for s in recent):
            weak.append(COMPETENCY_KO[history.competency])

    needs_adjustment = bool(weak)

    if needs_adjustment:
        result = evaluator.curriculum_adjust(weak)
        adjustment_message = result["adjustment_message"]
        recommended_focus = result["recommended_focus"]
    else:
        adjustment_message = "현재 학습 경로가 적절합니다. 꾸준히 학습을 이어나가세요!"
        recommended_focus = "현재 역량 수준에 맞는 문제를 계속 풀어보세요."

    return CurriculumAdjustResponse(
        needs_adjustment=needs_adjustment,
        weak_competencies=weak,
        adjustment_message=adjustment_message,
        recommended_focus=recommended_focus,
    )


# ── 답변 변화 추적 ────────────────────────────────────────────────────────────

def compare_answers(req: CompareAnswersRequest) -> CompareAnswersResponse:
    evaluator = get_writing_evaluator()

    newly_included: list[str] = []
    still_missing: list[str] = []

    if req.keywords:
        for kw in req.keywords:
            in_prev = kw.keyword in req.previous_answer
            in_curr = kw.keyword in req.current_answer
            if in_curr and not in_prev:
                newly_included.append(kw.keyword)
            elif not in_curr:
                still_missing.append(kw.keyword)

    result = evaluator.compare_answers(
        question_text=req.question_text,
        model_answer=req.model_answer,
        previous_answer=req.previous_answer,
        previous_score=req.previous_score,
        current_answer=req.current_answer,
        current_score=req.current_score,
        newly_included=newly_included,
        still_missing=still_missing,
    )

    score_diff = req.current_score - req.previous_score
    return CompareAnswersResponse(
        score_diff=score_diff,
        is_improved=score_diff > 0,
        growth_message=result["growth_message"],
        newly_included_keywords=newly_included,
        still_missing_keywords=still_missing,
        analysis=result["analysis"],
    )


# ── 약점 분석 리포트 ──────────────────────────────────────────────────────────

_COMPETENCY_TO_READING_TYPE = {
    "factual": "FACTUAL",
    "inferential": "INFERENTIAL",
    "critical": "CRITICAL",
}


def generate_curriculum_plan(req: CurriculumPlanRequest) -> CurriculumPlanResponse:
    theta = req.theta
    if theta < -0.5:
        stages = [1]
    elif theta < 0.0:
        stages = [1, 2]
    elif theta < 0.5:
        stages = [2, 3]
    else:
        stages = [3, 4]

    _stage_diffs: dict[int, list[int]] = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 5],
    }

    weak_types: set[str] = {
        _COMPETENCY_TO_READING_TYPE[comp.lower()]
        for comp, score in req.competency_scores.items()
        if score < 50 and comp.lower() in _COMPETENCY_TO_READING_TYPE
    }

    by_diff: dict[int, list[AvailableProblem]] = {}
    for p in req.available_problems:
        by_diff.setdefault(p.difficulty, []).append(p)

    per_stage = req.daily_goal * 7
    plan: dict[int, list[int]] = {}

    for stage in stages:
        candidates: list[AvailableProblem] = []
        for d in _stage_diffs.get(stage, [stage]):
            candidates.extend(by_diff.get(d, []))

        priority = [p for p in candidates if p.reading_type.upper() in weak_types]
        others = [p for p in candidates if p.reading_type.upper() not in weak_types]
        selected = (priority + others)[:per_stage]
        plan[stage] = [p.id for p in selected]

    return CurriculumPlanResponse(plan=plan)


def explain_term(req: TermExplainRequest) -> TermExplainResponse:
    evaluator = get_writing_evaluator()
    explanation = evaluator.explain(req.term, req.passage_text)
    return TermExplainResponse(term=req.term, explanation=explanation)


def generate_weakness_report(req: WeaknessReportRequest) -> WeaknessReportResponse:
    evaluator = get_writing_evaluator()

    competency_scores: dict[str, int] = {
        COMPETENCY_KO[item.competency]: item.score for item in req.competency_scores
    }

    result = evaluator.weakness_report(competency_scores)

    weak_details: list[WeakCompetencyDetail] = []
    priority: str | None = None
    lowest_score = 101

    for item in req.competency_scores:
        ko_name = COMPETENCY_KO[item.competency]
        if item.score < 50:
            level = "취약"
            weak_details.append(WeakCompetencyDetail(
                competency=ko_name, score=item.score, level=level
            ))
            if item.score < lowest_score:
                lowest_score = item.score
                priority = ko_name
        elif item.score < 70:
            weak_details.append(WeakCompetencyDetail(
                competency=ko_name, score=item.score, level="보통"
            ))

    return WeaknessReportResponse(
        weak_competencies=weak_details,
        report=result["report"],
        recommendations=result["recommendations"],
        priority_competency=priority,
    )
