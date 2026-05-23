"""
글쓰기 채점 모델 추론 모듈
Hub: Onjeom/writing-ai (Llama-3.1-8B QLoRA, unsloth)
"""

import re
import torch
from unsloth import FastLanguageModel


ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

RELAXED_INSTRUCTION = (
    "주어진 지시문과 학생의 답안을 분석하여, 부족한 점과 개선 방향을 포함한 피드백을 작성하고 "
    "맨 마지막에 1점부터 4점 사이의 최종 점수를 부여하시오.\n\n"
    "[유연하고 관대한 채점 기준]\n"
    "- 4점: 지시문의 핵심 요구사항을 잘 파악하였고 전반적인 흐름이 우수한 답안 "
    "(사소한 결함은 너그럽게 만점 처리)\n"
    "- 3점: 지시문은 이해했으나 근거가 다소 평이하거나 논리의 깊이가 아쉬운 일반적인 답안\n"
    "- 2점: 지시문의 키워드만 겨우 나열했거나 주장의 근거가 심각하게 부족한 답안\n"
    "- 1점 (최하점): 같은 말을 무의미하게 반복하거나 꼼수가 명백한 답안"
)

# 1~4점 → 정규화 점수 (백엔드 int 반환용)
SCORE_MAP = {1: 25, 2: 50, 3: 75, 4: 100}


class WritingEvaluator:
    """서술형 답안 자동 채점 및 피드백 생성."""

    def __init__(self, model_name: str = "Onjeom/writing-ai", max_seq_length: int = 1536):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(self._model)

    def _build_prompt(self, question_text: str, model_answer: str, user_answer: str) -> str:
        instruction = (
            f"{RELAXED_INSTRUCTION}\n\n"
            f"[문제]\n{question_text}\n\n"
            f"[모범 답안]\n{model_answer}"
        )
        return ALPACA_PROMPT.format(
            instruction=instruction,
            input=user_answer[:700],
        )

    @staticmethod
    def _parse_score(text: str) -> int:
        """응답 텍스트에서 [최종 점수: X] 파싱, 실패 시 2 반환."""
        match = re.search(r"\[최종\s*점수\s*:\s*([1-4])\]", text)
        if match:
            return int(match.group(1))
        # 숫자만 있는 경우 fallback
        digits = re.findall(r"\b([1-4])\b", text[-50:])
        return int(digits[-1]) if digits else 2

    @staticmethod
    def _parse_feedback(text: str) -> str:
        """점수 태그 이전 텍스트를 피드백으로 반환."""
        parts = re.split(r"\[최종\s*점수\s*:", text)
        return parts[0].strip()

    def evaluate(
        self,
        passage_text: str,
        question_text: str,
        model_answer: str,
        user_answer: str,
    ) -> dict:
        """
        Args:
            passage_text: 지문 (참고용, 프롬프트에 포함 가능)
            question_text: 문제 지시문
            model_answer: 모범 답안
            user_answer: 학생 답안

        Returns:
            {
                "raw_score": 1~4,
                "normalized_score": 25/50/75/100,
                "feedback": "...",
                "full_response": "...",
            }
        """
        self._load()

        prompt = self._build_prompt(question_text, model_answer, user_answer)
        inputs = self._tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=True,
            )

        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("### Response:\n")[-1].strip()

        raw_score = self._parse_score(response)
        return {
            "raw_score": raw_score,
            "normalized_score": SCORE_MAP.get(raw_score, 50),
            "feedback": self._parse_feedback(response),
            "full_response": response,
        }


# ──────────────────────────────────────────────
# 간단 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = WritingEvaluator()
    result = evaluator.evaluate(
        passage_text="선인장은 사막에 사는 식물이다.",
        question_text="선인장이 사막에서 살 수 있는 이유를 서술하시오.",
        model_answer="선인장은 줄기에 물을 저장하고 잎이 가시로 변해 수분 손실을 줄이기 때문에 사막에서 살 수 있다.",
        user_answer="선인장은 물을 저장해서 살 수 있다.",
    )
    print(f"점수: {result['raw_score']}점 ({result['normalized_score']}점)")
    print(f"피드백: {result['feedback']}")
