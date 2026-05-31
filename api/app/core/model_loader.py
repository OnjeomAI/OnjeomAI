"""
모델 싱글톤 로더 — 앱 기동 시 한 번만 로드
"""

import os
import sys

# models/ 경로를 PYTHONPATH에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../models"))

from writing.inference import WritingEvaluator
from .config import settings

_writing_evaluator: WritingEvaluator | None = None

# HuggingFace에서 다운로드된 모델을 저장할 로컬 디렉터리
_MODEL_LOCAL_DIR = os.environ.get(
    "WRITING_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "../../../../models/writing/weights"),
)


def _ensure_model_downloaded(repo_id: str, local_dir: str) -> str:
    """로컬에 모델이 없으면 HuggingFace Hub에서 최초 1회 다운로드 후 로컬 경로 반환."""
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir
    print(f"[model_loader] Writing 모델 다운로드 시작: {repo_id} → {local_dir}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("[model_loader] 다운로드 완료")
    return local_dir


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
            try:
                local_path = _ensure_model_downloaded(
                    repo_id=settings.writing_model_name,
                    local_dir=_MODEL_LOCAL_DIR,
                )
                _writing_evaluator = WritingEvaluator(
                    model_name=local_path,
                    max_seq_length=settings.writing_max_seq_length,
                )
                _writing_evaluator._load()
            except ModuleNotFoundError as e:
                print(f"[model_loader] Writing 모델 로드 실패 ({e}) → Mock으로 대체")
                _writing_evaluator = _MockWritingEvaluator(
                    model_name=settings.writing_model_name,
                    max_seq_length=settings.writing_max_seq_length,
                )
    return _writing_evaluator
