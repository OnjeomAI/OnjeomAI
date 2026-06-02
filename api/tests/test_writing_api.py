"""
글쓰기 AI API 엔드포인트 계약 테스트
- FastAPI TestClient로 실제 HTTP 요청/응답 검증
- 품질 기준: 올바른 상태 코드, 필수 필드 존재, 타입 정합성
"""

import pytest


EVALUATE_PAYLOAD = {
    "passage_text": "환경 보호는 현대 사회의 중요한 과제이다. 우리는 자연을 아끼고 보전해야 한다.",
    "question_text": "작가의 주장을 서술하시오.",
    "model_answer": "작가는 자연보호의 필요성을 주장한다.",
    "user_answer": "작가는 자연보호와 생태계 보전이 중요하다고 주장한다.",
    "keywords": [
        {"keyword": "자연보호", "weight": 60},
        {"keyword": "생태계", "weight": 40},
    ],
    "reading_type": "FACTUAL",
}

ADJUST_PAYLOAD = {
    "competency_history": [
        {"competency": "inferential", "scores": [30, 40, 45]},
        {"competency": "factual", "scores": [80, 75, 85]},
    ]
}

COMPARE_PAYLOAD = {
    "question_text": "작가의 주장을 서술하시오.",
    "model_answer": "작가는 자연보호의 필요성을 주장한다.",
    "previous_answer": "환경을 보호해야 한다.",
    "previous_score": 40,
    "current_answer": "작가는 자연보호와 생태계 보전의 필요성을 강조한다.",
    "current_score": 75,
    "keywords": [{"keyword": "자연보호", "weight": 100}],
}

WEAKNESS_PAYLOAD = {
    "competency_scores": [
        {"competency": "factual", "score": 40},
        {"competency": "inferential", "score": 65},
        {"competency": "critical", "score": 85},
    ]
}

CURRICULUM_PLAN_PAYLOAD = {
    "theta": 0.3,
    "daily_goal": 3,
    "competency_scores": {"factual": 80, "inferential": 35, "critical": 70},
    "available_problems": [
        {"id": i, "difficulty": (i % 5) + 1, "reading_type": rt}
        for i, rt in enumerate(["FACTUAL", "INFERENTIAL", "CRITICAL"] * 7, start=1)
    ],
}

EXPLAIN_PAYLOAD = {
    "term": "은유법",
    "passage_text": "달은 나의 마음이다.",
}

IRT_PAYLOAD = {
    "responses": [
        {"difficulty": 3, "score": 70},
        {"difficulty": 4, "score": 80},
        {"difficulty": 3, "score": 60},
    ]
}


class TestEvaluateEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/evaluate", json=EVALUATE_PAYLOAD)
        assert res.status_code == 200

    def test_response_fields(self, test_client):
        data = test_client.post("/api/writing/evaluate", json=EVALUATE_PAYLOAD).json()
        assert "final_score" in data
        assert "feedback" in data
        assert "feedback_type" in data
        assert "matched_keywords" in data
        assert "missing_keywords" in data

    def test_final_score_range(self, test_client):
        data = test_client.post("/api/writing/evaluate", json=EVALUATE_PAYLOAD).json()
        assert 0 <= data["final_score"] <= 100

    def test_feedback_not_empty(self, test_client):
        data = test_client.post("/api/writing/evaluate", json=EVALUATE_PAYLOAD).json()
        assert data["feedback"] != ""

    def test_without_keywords(self, test_client):
        payload = {**EVALUATE_PAYLOAD, "keywords": []}
        data = test_client.post("/api/writing/evaluate", json=payload).json()
        assert data["keyword_score"] is None
        assert data["final_score"] == data["normalized_score"]

    def test_missing_required_field_returns_422(self, test_client):
        res = test_client.post("/api/writing/evaluate", json={"passage_text": "지문"})
        assert res.status_code == 422


class TestCurriculumAdjustEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/curriculum/adjust", json=ADJUST_PAYLOAD)
        assert res.status_code == 200

    def test_weak_competency_detected(self, test_client):
        data = test_client.post("/api/writing/curriculum/adjust", json=ADJUST_PAYLOAD).json()
        assert data["needs_adjustment"] is True
        assert len(data["weak_competencies"]) > 0

    def test_no_adjustment_when_all_scores_high(self, test_client):
        payload = {
            "competency_history": [
                {"competency": "factual", "scores": [80, 85, 90]},
            ]
        }
        data = test_client.post("/api/writing/curriculum/adjust", json=payload).json()
        assert data["needs_adjustment"] is False

    def test_adjustment_message_not_empty(self, test_client):
        data = test_client.post("/api/writing/curriculum/adjust", json=ADJUST_PAYLOAD).json()
        assert data["adjustment_message"] != ""
        assert data["recommended_focus"] != ""


class TestCompareAnswersEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/compare", json=COMPARE_PAYLOAD)
        assert res.status_code == 200

    def test_score_diff_calculated(self, test_client):
        data = test_client.post("/api/writing/compare", json=COMPARE_PAYLOAD).json()
        assert data["score_diff"] == 35  # 75 - 40
        assert data["is_improved"] is True

    def test_growth_message_not_empty(self, test_client):
        data = test_client.post("/api/writing/compare", json=COMPARE_PAYLOAD).json()
        assert data["growth_message"] != ""

    def test_keyword_tracking(self, test_client):
        data = test_client.post("/api/writing/compare", json=COMPARE_PAYLOAD).json()
        # "자연보호"가 이전엔 없고 현재엔 있으므로 newly_included
        assert "자연보호" in data["newly_included_keywords"]


class TestWeaknessReportEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/weakness-report", json=WEAKNESS_PAYLOAD)
        assert res.status_code == 200

    def test_weak_competencies_classified(self, test_client):
        data = test_client.post("/api/writing/weakness-report", json=WEAKNESS_PAYLOAD).json()
        levels = {w["competency"]: w["level"] for w in data["weak_competencies"]}
        assert levels.get("사실적 독해") == "취약"    # score=40
        assert levels.get("추론적 독해") == "보통"    # score=65
        assert "비판적 독해" not in levels             # score=85, 양호

    def test_priority_competency_is_lowest(self, test_client):
        data = test_client.post("/api/writing/weakness-report", json=WEAKNESS_PAYLOAD).json()
        assert data["priority_competency"] == "사실적 독해"

    def test_recommendations_not_empty(self, test_client):
        data = test_client.post("/api/writing/weakness-report", json=WEAKNESS_PAYLOAD).json()
        assert len(data["recommendations"]) > 0


class TestCurriculumPlanEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/curriculum-plan", json=CURRICULUM_PLAN_PAYLOAD)
        assert res.status_code == 200

    def test_plan_stages_match_theta(self, test_client):
        data = test_client.post("/api/writing/curriculum-plan", json=CURRICULUM_PLAN_PAYLOAD).json()
        # theta=0.3 → 스테이지 [2, 3]
        assert set(map(int, data["plan"].keys())) == {2, 3}

    def test_plan_problem_ids_are_valid(self, test_client):
        data = test_client.post("/api/writing/curriculum-plan", json=CURRICULUM_PLAN_PAYLOAD).json()
        valid_ids = {p["id"] for p in CURRICULUM_PLAN_PAYLOAD["available_problems"]}
        for ids in data["plan"].values():
            for pid in ids:
                assert pid in valid_ids


class TestExplainTermEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/explain-term", json=EXPLAIN_PAYLOAD)
        assert res.status_code == 200

    def test_term_echoed(self, test_client):
        data = test_client.post("/api/writing/explain-term", json=EXPLAIN_PAYLOAD).json()
        assert data["term"] == "은유법"

    def test_explanation_not_empty(self, test_client):
        data = test_client.post("/api/writing/explain-term", json=EXPLAIN_PAYLOAD).json()
        assert data["explanation"] != ""

    def test_without_passage(self, test_client):
        data = test_client.post("/api/writing/explain-term", json={"term": "직유법"}).json()
        assert data["term"] == "직유법"


class TestIrtEstimateEndpoint:
    def test_status_200(self, test_client):
        res = test_client.post("/api/writing/irt/estimate", json=IRT_PAYLOAD)
        assert res.status_code == 200

    def test_response_fields(self, test_client):
        data = test_client.post("/api/writing/irt/estimate", json=IRT_PAYLOAD).json()
        assert "theta" in data
        assert "se" in data
        assert "ability_level" in data
        assert "next_difficulty" in data

    def test_empty_responses_returns_422(self, test_client):
        res = test_client.post("/api/writing/irt/estimate", json={"responses": []})
        assert res.status_code == 422
