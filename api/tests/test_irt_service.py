"""
IRT 능력 추정 서비스 품질 테스트
- 실제 수학 계산 검증 (모델 불필요)
- 품질 기준: 정답률 높으면 theta 양수, 낮으면 음수
"""

import pytest
from app.writing.services.irt_service import estimate_theta


class TestEstimateTheta:
    def _response(self, difficulty: int, score: int) -> dict:
        return {"difficulty": difficulty, "score": score}

    def test_all_correct_high_difficulty_returns_positive_theta(self):
        responses = [self._response(d, 80) for d in [4, 5, 4, 5, 4]]
        result = estimate_theta(responses)
        assert result["theta"] > 0.5, f"고난도 전정답이면 theta > 0.5 기대, 실제: {result['theta']}"

    def test_all_wrong_low_difficulty_returns_negative_theta(self):
        responses = [self._response(d, 20) for d in [1, 2, 1, 2, 1]]
        result = estimate_theta(responses)
        assert result["theta"] < -0.5, f"저난도 전오답이면 theta < -0.5 기대, 실제: {result['theta']}"

    def test_mixed_responses_returns_middle_theta(self):
        responses = [
            self._response(3, 70),  # 정답
            self._response(3, 30),  # 오답
            self._response(3, 80),  # 정답
            self._response(3, 20),  # 오답
        ]
        result = estimate_theta(responses)
        assert -1.5 < result["theta"] < 1.5

    def test_response_structure(self):
        responses = [self._response(3, 60)]
        result = estimate_theta(responses)
        assert "theta" in result
        assert "se" in result
        assert "ability_level" in result
        assert "next_difficulty" in result

    def test_ability_level_label(self):
        high = estimate_theta([self._response(5, 90)] * 5)
        low = estimate_theta([self._response(1, 10)] * 5)
        assert high["ability_level"] in ["중상", "상"]
        assert low["ability_level"] in ["하", "중하"]

    def test_next_difficulty_range(self):
        result = estimate_theta([self._response(3, 60)])
        assert 1 <= result["next_difficulty"] <= 5

    def test_se_positive(self):
        result = estimate_theta([self._response(3, 60)])
        assert result["se"] > 0

    def test_binarization_threshold_at_50(self):
        """score=50은 정답(1), score=49는 오답(0)으로 처리."""
        above = estimate_theta([self._response(3, 50)] * 5)
        below = estimate_theta([self._response(3, 49)] * 5)
        assert above["theta"] > below["theta"]

    def test_optional_irt_params(self):
        """a_param, b_param, c_param 직접 지정 시 정상 동작."""
        responses = [
            {"difficulty": 3, "score": 70, "a_param": 1.5, "b_param": 0.0, "c_param": 0.1}
        ]
        result = estimate_theta(responses)
        assert isinstance(result["theta"], float)
