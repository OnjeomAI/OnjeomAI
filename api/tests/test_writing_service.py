"""
글쓰기 AI 서비스 로직 품질 테스트
- 모델 로드 없이 순수 로직만 검증
- 품질 기준: 점수 계산, 취약 역량 탐지, 스테이지 선택, 키워드 추적
"""

import pytest
from unittest.mock import patch, MagicMock

from app.writing.services.writing_service import (
    _calc_keyword_score,
    _calc_final_score,
    _score_feedback,
    adjust_curriculum,
    compare_answers,
    generate_curriculum_plan,
    generate_weakness_report,
    evaluate_writing,
    explain_term,
)
from app.writing.schemas.writing import (
    KeywordItem,
    Competency,
    CompetencyHistory,
    CurriculumAdjustRequest,
    CompareAnswersRequest,
    CurriculumPlanRequest,
    AvailableProblem,
    CompetencyScoreItem,
    WeaknessReportRequest,
    WritingEvaluateRequest,
    TermExplainRequest,
    FeedbackType,
)


# ── 키워드 채점 ────────────────────────────────────────────────────────────────

class TestKeywordScore:
    def test_all_matched(self):
        keywords = [KeywordItem(keyword="주제", weight=50), KeywordItem(keyword="근거", weight=50)]
        score, matched, missing = _calc_keyword_score("주제와 근거가 명확합니다.", keywords)
        assert score == 100
        assert set(matched) == {"주제", "근거"}
        assert missing == []

    def test_partial_match(self):
        keywords = [KeywordItem(keyword="주제", weight=60), KeywordItem(keyword="근거", weight=40)]
        score, matched, missing = _calc_keyword_score("주제만 있습니다.", keywords)
        assert score == 60
        assert matched == ["주제"]
        assert missing == ["근거"]

    def test_no_match(self):
        keywords = [KeywordItem(keyword="주제", weight=100)]
        score, matched, missing = _calc_keyword_score("관계없는 내용입니다.", keywords)
        assert score == 0
        assert matched == []
        assert missing == ["주제"]

    def test_empty_keywords(self):
        score, matched, missing = _calc_keyword_score("어떤 답변", [])
        assert score == 0
        assert matched == []
        assert missing == []


# ── 최종 점수 계산 ─────────────────────────────────────────────────────────────

class TestFinalScore:
    def test_with_keyword_score(self):
        # 키워드 40% + LLM 60%
        assert _calc_final_score(100, 75) == 85   # 100*0.4 + 75*0.6
        assert _calc_final_score(0, 75) == 45     # 0*0.4 + 75*0.6
        assert _calc_final_score(60, 50) == 54    # 60*0.4 + 50*0.6

    def test_without_keyword_score(self):
        assert _calc_final_score(None, 75) == 75
        assert _calc_final_score(None, 25) == 25


# ── 점수 구간별 피드백 ─────────────────────────────────────────────────────────

class TestScoreFeedback:
    def test_excellent(self):
        fb_type, msg = _score_feedback(80)
        assert fb_type == FeedbackType.EXCELLENT
        assert "심화" in msg

    def test_excellent_boundary(self):
        fb_type, _ = _score_feedback(100)
        assert fb_type == FeedbackType.EXCELLENT

    def test_good(self):
        fb_type, msg = _score_feedback(65)
        assert fb_type == FeedbackType.GOOD
        assert "보완" in msg

    def test_good_boundary(self):
        fb_type, _ = _score_feedback(50)
        assert fb_type == FeedbackType.GOOD

    def test_needs_improvement(self):
        fb_type, msg = _score_feedback(49)
        assert fb_type == FeedbackType.NEEDS_IMPROVEMENT
        assert "다시" in msg or "개념" in msg

    def test_zero(self):
        fb_type, _ = _score_feedback(0)
        assert fb_type == FeedbackType.NEEDS_IMPROVEMENT


# ── 커리큘럼 플랜 스테이지 선택 ───────────────────────────────────────────────

class TestCurriculumPlanStages:
    """theta 기반 스테이지 선택 품질 테스트."""

    def _make_problems(self, diffs: list[int], reading_type: str = "FACTUAL") -> list[AvailableProblem]:
        return [AvailableProblem(id=i + 1, difficulty=d, reading_type=reading_type)
                for i, d in enumerate(diffs)]

    def _make_request(self, theta: float, problems: list[AvailableProblem],
                      competency_scores: dict | None = None) -> CurriculumPlanRequest:
        return CurriculumPlanRequest(
            theta=theta,
            daily_goal=2,
            competency_scores=competency_scores or {"factual": 80, "inferential": 80},
            available_problems=problems,
        )

    def test_low_theta_stage1_only(self):
        problems = self._make_problems([1, 2, 3, 4])
        req = self._make_request(-1.0, problems)
        result = generate_curriculum_plan(req)
        assert set(result.plan.keys()) == {1}

    def test_medium_low_theta_stages_1_2(self):
        problems = self._make_problems([1, 2, 2, 3])
        req = self._make_request(-0.3, problems)
        result = generate_curriculum_plan(req)
        assert set(result.plan.keys()) == {1, 2}

    def test_medium_high_theta_stages_2_3(self):
        problems = self._make_problems([2, 3, 3, 4])
        req = self._make_request(0.3, problems)
        result = generate_curriculum_plan(req)
        assert set(result.plan.keys()) == {2, 3}

    def test_high_theta_stages_3_4(self):
        problems = self._make_problems([3, 4, 4, 5])
        req = self._make_request(1.0, problems)
        result = generate_curriculum_plan(req)
        assert set(result.plan.keys()) == {3, 4}

    def test_weak_competency_prioritized(self):
        """취약 역량(score<50)의 reading_type 문제가 앞에 배치되는지 검증."""
        problems = [
            AvailableProblem(id=1, difficulty=2, reading_type="INFERENTIAL"),  # 취약
            AvailableProblem(id=2, difficulty=2, reading_type="INFERENTIAL"),  # 취약
            AvailableProblem(id=3, difficulty=2, reading_type="FACTUAL"),
            AvailableProblem(id=4, difficulty=2, reading_type="FACTUAL"),
        ]
        req = CurriculumPlanRequest(
            theta=-0.3,
            daily_goal=2,
            competency_scores={"factual": 80, "inferential": 30},  # inferential 취약
            available_problems=problems,
        )
        result = generate_curriculum_plan(req)
        stage1_ids = result.plan.get(1, [])
        # inferential(id=1,2)이 factual(id=3,4)보다 앞에 있어야 함
        inferential_ids = {1, 2}
        factual_ids = {3, 4}
        first_two = set(stage1_ids[:2])
        assert first_two == inferential_ids

    def test_empty_problems_returns_empty_stage(self):
        req = self._make_request(-1.0, [])
        result = generate_curriculum_plan(req)
        assert result.plan.get(1, []) == []


# ── 커리큘럼 재조정 ───────────────────────────────────────────────────────────

class TestAdjustCurriculum:
    def _make_request(self, histories: list[tuple]) -> CurriculumAdjustRequest:
        return CurriculumAdjustRequest(
            competency_history=[
                CompetencyHistory(competency=Competency(comp), scores=scores)
                for comp, scores in histories
            ]
        )

    def test_needs_adjustment_when_3_consecutive_below_50(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            req = self._make_request([("inferential", [40, 45, 30])])
            result = adjust_curriculum(req)
        assert result.needs_adjustment is True
        assert "추론적 독해" in result.weak_competencies

    def test_no_adjustment_when_scores_above_50(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            req = self._make_request([("factual", [60, 70, 80])])
            result = adjust_curriculum(req)
        assert result.needs_adjustment is False
        assert result.weak_competencies == []

    def test_no_adjustment_when_not_3_consecutive(self, fake_evaluator):
        """3회 미만이면 취약으로 판정 안 함."""
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            req = self._make_request([("critical", [40, 80])])
            result = adjust_curriculum(req)
        assert result.needs_adjustment is False

    def test_mixed_competencies(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            req = self._make_request([
                ("factual", [80, 90, 85]),
                ("inferential", [30, 40, 45]),
            ])
            result = adjust_curriculum(req)
        assert result.needs_adjustment is True
        assert "추론적 독해" in result.weak_competencies
        assert "사실적 독해" not in result.weak_competencies


# ── 답변 비교 ─────────────────────────────────────────────────────────────────

class TestCompareAnswers:
    def _base_req(self, **kwargs) -> CompareAnswersRequest:
        defaults = dict(
            question_text="작가의 주장을 서술하시오.",
            model_answer="작가는 자연보호의 필요성을 주장한다.",
            previous_answer="작가는 환경에 대해 말한다.",
            previous_score=40,
            current_answer="작가는 자연보호의 필요성을 강조한다.",
            current_score=75,
            keywords=[],
        )
        defaults.update(kwargs)
        return CompareAnswersRequest(**defaults)

    def test_score_diff_positive_is_improved(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = compare_answers(self._base_req(previous_score=40, current_score=75))
        assert result.score_diff == 35
        assert result.is_improved is True

    def test_score_diff_negative_not_improved(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = compare_answers(self._base_req(previous_score=75, current_score=40))
        assert result.score_diff == -35
        assert result.is_improved is False

    def test_newly_included_keyword_detected(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = compare_answers(self._base_req(
                previous_answer="환경을 보호해야 한다.",
                current_answer="자연보호와 생태계 보전이 필요하다.",
                keywords=[
                    KeywordItem(keyword="자연보호", weight=50),
                    KeywordItem(keyword="생태계", weight=50),
                ],
            ))
        assert "자연보호" in result.newly_included_keywords
        assert "생태계" in result.newly_included_keywords

    def test_still_missing_keyword_detected(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = compare_answers(self._base_req(
                previous_answer="환경을 보호해야 한다.",
                current_answer="환경을 보호해야 한다.",  # 동일
                keywords=[KeywordItem(keyword="자연보호", weight=100)],
            ))
        assert "자연보호" in result.still_missing_keywords
        assert result.newly_included_keywords == []


# ── 약점 분석 ─────────────────────────────────────────────────────────────────

class TestWeaknessReport:
    def _make_request(self, scores: dict) -> WeaknessReportRequest:
        return WeaknessReportRequest(
            competency_scores=[
                CompetencyScoreItem(competency=Competency(k), score=v)
                for k, v in scores.items()
            ]
        )

    def test_critical_competency_classified_as_weak(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = generate_weakness_report(self._make_request({
                "factual": 40, "inferential": 80,
            }))
        weak = {w.competency: w.level for w in result.weak_competencies}
        assert weak.get("사실적 독해") == "취약"
        assert "추론적 독해" not in weak

    def test_moderate_competency_classified_as_normal(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = generate_weakness_report(self._make_request({
                "factual": 60, "inferential": 80,
            }))
        weak = {w.competency: w.level for w in result.weak_competencies}
        assert weak.get("사실적 독해") == "보통"

    def test_priority_competency_is_lowest_scorer(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = generate_weakness_report(self._make_request({
                "factual": 30, "inferential": 45, "critical": 80,
            }))
        assert result.priority_competency == "사실적 독해"  # 30점이 최저

    def test_no_weak_competency(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = generate_weakness_report(self._make_request({
                "factual": 80, "inferential": 75,
            }))
        assert result.priority_competency is None
        assert result.weak_competencies == []


# ── 글쓰기 채점 (전체 파이프라인) ─────────────────────────────────────────────

class TestEvaluateWriting:
    def _base_req(self, **kwargs) -> WritingEvaluateRequest:
        defaults = dict(
            passage_text="환경 보호는 현대 사회의 중요한 과제이다.",
            question_text="작가의 주장을 서술하시오.",
            model_answer="작가는 자연보호의 필요성을 주장한다.",
            user_answer="작가는 자연보호가 중요하다고 말한다.",
            keywords=[],
        )
        defaults.update(kwargs)
        return WritingEvaluateRequest(**defaults)

    def test_response_structure(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = evaluate_writing(self._base_req())
        assert result.final_score >= 0
        assert result.feedback != ""
        assert result.feedback_type is not None

    def test_deep_analysis_triggered_below_50(self, fake_evaluator):
        low_score_evaluator = MagicMock()
        low_score_evaluator.evaluate.return_value = {
            "raw_score": 1, "normalized_score": 25, "feedback": "핵심 내용이 빠졌습니다."
        }
        low_score_evaluator.deep_analysis.return_value = {
            "error_types": ["내용 누락"],
            "analysis": "주요 키워드가 없습니다.",
            "improvement": "지문을 다시 읽어보세요.",
        }
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=low_score_evaluator):
            result = evaluate_writing(self._base_req())
        assert result.final_score < 50
        assert result.deep_analysis is not None
        assert result.deep_analysis.improvement != ""
        low_score_evaluator.deep_analysis.assert_called_once()

    def test_no_deep_analysis_above_50(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = evaluate_writing(self._base_req())
        assert result.final_score >= 50
        assert result.deep_analysis is None

    def test_keyword_blending(self, fake_evaluator):
        """키워드 점수 40% + LLM 60% 블렌딩 검증."""
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = evaluate_writing(self._base_req(
                user_answer="자연보호와 생태계 보전이 중요하다.",
                keywords=[
                    KeywordItem(keyword="자연보호", weight=50),
                    KeywordItem(keyword="생태계", weight=50),
                ],
            ))
        # keyword_score=100, normalized_score=75 → final=100*0.4+75*0.6=85
        assert result.keyword_score == 100
        assert result.final_score == 85


# ── 용어 설명 ─────────────────────────────────────────────────────────────────

class TestExplainTerm:
    def test_returns_explanation(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = explain_term(TermExplainRequest(term="은유법"))
        assert result.term == "은유법"
        assert result.explanation != ""

    def test_with_passage_context(self, fake_evaluator):
        with patch("app.writing.services.writing_service.get_writing_evaluator", return_value=fake_evaluator):
            result = explain_term(TermExplainRequest(
                term="은유법",
                passage_text="달은 나의 마음이다."
            ))
        assert result.term == "은유법"
        assert result.explanation != ""
