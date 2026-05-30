from app.core.model_loader import get_writing_evaluator
from app.schemas.writing import (
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


# ── 동적 학습 경로 재조정 ─────────────────────────────────────────────────────

def adjust_curriculum(req: CurriculumAdjustRequest) -> CurriculumAdjustResponse:
    """3회 연속 50점 미만 역량을 취약 역량으로 판정하여 재조정 메시지 생성."""
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
    """이전·현재 답변을 비교하여 성장 메시지와 키워드 변화를 반환."""
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

_LEVEL_LABELS = {True: "취약", False: "보통"}  # True = score < 50

# 역량명 → ReadingType 매핑 (대소문자 무관)
_COMPETENCY_TO_READING_TYPE = {
    "factual": "FACTUAL",
    "inferential": "INFERENTIAL",
    "critical": "CRITICAL",
}


# ── 커리큘럼 플랜 ──────────────────────────────────────────────────────────────

def generate_curriculum_plan(req: CurriculumPlanRequest) -> CurriculumPlanResponse:
    """
    theta 기반 스테이지 결정 후 취약 역량 reading_type 우선 배치.
    - 스테이지별 배정 문제 수: daily_goal × 7
    """
    theta = req.theta
    if theta < -0.5:
        stages = [1]
    elif theta < 0.0:
        stages = [1, 2]
    elif theta < 0.5:
        stages = [2, 3]
    else:
        stages = [3, 4]

    # stage → 적합 difficulty 범위
    _stage_diffs: dict[int, list[int]] = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 5],
    }

    # 취약 역량(50점 미만)에 대응하는 reading_type 집합
    weak_types: set[str] = {
        _COMPETENCY_TO_READING_TYPE[comp.lower()]
        for comp, score in req.competency_scores.items()
        if score < 50 and comp.lower() in _COMPETENCY_TO_READING_TYPE
    }

    # difficulty → 문제 목록 색인
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
    """용어/문장에 대해 쉬운 말로 설명. passageText가 있으면 지문 맥락을 활용."""
    evaluator = get_writing_evaluator()

    if req.passage_text:
        prompt = (
            f"다음 지문을 참고하여 '{req.term}'을(를) 중학생도 이해할 수 있는 쉬운 말로 2~3문장으로 설명해.\n\n"
            f"[지문]\n{req.passage_text[:500]}\n\n"
            f"설명:"
        )
    else:
        prompt = (
            f"'{req.term}'을(를) 중학생도 이해할 수 있는 쉬운 말로 2~3문장으로 설명해.\n\n"
            f"설명:"
        )

    explanation = evaluator._generate(prompt, max_new_tokens=200).strip()
    return TermExplainResponse(term=req.term, explanation=explanation)


def generate_weakness_report(req: WeaknessReportRequest) -> WeaknessReportResponse:
    """역량별 평균 점수를 분석하여 약점 리포트와 개선 권장사항 생성."""
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
