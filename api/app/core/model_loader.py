"""
모델 싱글톤 로더 — 앱 기동 시 한 번만 로드
"""

import sys
import os

# models/ 경로를 PYTHONPATH에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../models"))

from writing.inference import WritingEvaluator
from .config import settings

_writing_evaluator: WritingEvaluator | None = None


class _MockWritingEvaluator(WritingEvaluator):
    """MOCK_MODEL=true 시 실제 모델 없이 고정 응답을 반환하는 목 평가기."""

    def _load(self):
        pass  # 모델 로드 생략

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        return (
            "이 답안은 핵심 개념을 잘 파악하고 있으나 일부 근거가 부족합니다. "
            "논리 구조를 좀 더 명확히 해보세요. [최종 점수: 3]"
        )


def get_writing_evaluator() -> WritingEvaluator:
    global _writing_evaluator
    if _writing_evaluator is None:
        if settings.mock_model:
            _writing_evaluator = _MockWritingEvaluator(
                model_name=settings.writing_model_name,
                max_seq_length=settings.writing_max_seq_length,
            )
        else:
            _writing_evaluator = WritingEvaluator(
                model_name=settings.writing_model_name,
                max_seq_length=settings.writing_max_seq_length,
            )
            _writing_evaluator._load()
    return _writing_evaluator
