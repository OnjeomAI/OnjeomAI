import os
import sys
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

os.environ["MOCK_MODEL"] = "1"
os.environ["SKIP_MODEL_LOAD"] = "1"

# 다른 팀원 의존성(chromadb 등) 로드 차단 — writing 모델 테스트에는 불필요
from fastapi import APIRouter as _APIRouter

_empty_korean_qa = MagicMock()
_empty_korean_qa.router = _APIRouter()   # MagicMock 대신 실제 빈 라우터

sys.modules["chromadb"] = MagicMock()
sys.modules["app.services.rag_service"] = MagicMock()
sys.modules["app.routers.korean_qa"] = _empty_korean_qa


class FakeEvaluator:
    """테스트용 WritingEvaluator 대체 객체 — 실제 모델 로드 없이 고정 응답 반환."""

    def evaluate(self, passage_text, question_text, model_answer, user_answer,
                 reading_type=None):
        return {"raw_score": 3, "normalized_score": 75, "feedback": "핵심을 잘 파악했습니다."}

    def deep_analysis(self, passage_text, question_text, model_answer, user_answer, score):
        return {
            "error_types": ["내용 누락"],
            "analysis": "주요 개념이 누락됐습니다.",
            "improvement": "지문의 핵심 키워드를 포함해보세요.",
        }

    def curriculum_adjust(self, weak_competencies):
        joined = ", ".join(weak_competencies)
        return {
            "adjustment_message": f"{joined} 점수가 낮아요. 관련 문제를 먼저 배치했어요.",
            "recommended_focus": f"{joined} 문제를 집중적으로 연습하는 것을 권장합니다.",
        }

    def compare_answers(self, question_text, model_answer, previous_answer, previous_score,
                        current_answer, current_score, newly_included, still_missing):
        diff = current_score - previous_score
        msg = "성장했어요!" if diff > 0 else "다시 도전해보세요."
        return {"growth_message": msg, "analysis": "이전 답변보다 구체성이 향상됐습니다."}

    def weakness_report(self, competency_scores):
        return {
            "report": "추론적 독해 영역에서 집중적인 학습이 필요합니다.",
            "recommendations": ["추론 문제를 매일 2문제씩 풀어보세요.", "지문을 꼼꼼히 읽는 습관을 기르세요."],
        }

    def explain(self, term, passage_text=None):
        return f"'{term}'은(는) 쉽게 말해 이런 뜻입니다."


@pytest.fixture
def fake_evaluator():
    return FakeEvaluator()


@pytest.fixture
def test_client(fake_evaluator):
    from app.main import app
    import app.services.writing_service as ws
    import app.core.model_loader as ml

    ml._writing_evaluator = fake_evaluator

    with TestClient(app) as client:
        yield client
