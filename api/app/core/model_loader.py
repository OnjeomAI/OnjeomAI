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


def get_writing_evaluator() -> WritingEvaluator:
    global _writing_evaluator
    if _writing_evaluator is None:
        _writing_evaluator = WritingEvaluator(
            model_name=settings.writing_model_name,
            max_seq_length=settings.writing_max_seq_length,
        )
        _writing_evaluator._load()
    return _writing_evaluator
