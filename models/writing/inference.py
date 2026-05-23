"""
글쓰기 채점 모델 추론 모듈
Hub: Onjeom/writing-ai (Llama-3.1-8B QLoRA, unsloth)
"""

import json
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

DEEP_ANALYSIS_INSTRUCTION = (
    "학생 답안의 오류를 분석하고 반드시 아래 JSON 형식으로만 응답하시오.\n\n"
    "오류 유형 목록: 개념 혼동, 어휘 부족, 논리 비약, 내용 누락\n\n"
    "응답 형식:\n"
    '{"error_types": ["오류유형1", "오류유형2"], '
    '"analysis": "오류 원인 상세 분석", '
    '"improvement": "구체적 개선 방향"}'
)

# 1~4점 → 정규화 점수
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

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self._tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                do_sample=True,
            )
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("### Response:\n")[-1].strip()

    @staticmethod
    def _parse_score(text: str) -> int:
        match = re.search(r"\[최종\s*점수\s*:\s*([1-4])\]", text)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\b([1-4])\b", text[-50:])
        return int(digits[-1]) if digits else 2

    @staticmethod
    def _parse_feedback(text: str) -> str:
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
        2단계 LLM 채점 수행.

        Returns:
            raw_score (1~4), normalized_score (25/50/75/100), feedback, full_response
        """
        self._load()

        instruction = (
            f"{RELAXED_INSTRUCTION}\n\n"
            f"[문제]\n{question_text}\n\n"
            f"[모범 답안]\n{model_answer}"
        )
        prompt = ALPACA_PROMPT.format(
            instruction=instruction,
            input=user_answer[:700],
        )
        response = self._generate(prompt)
        raw_score = self._parse_score(response)

        return {
            "raw_score": raw_score,
            "normalized_score": SCORE_MAP.get(raw_score, 50),
            "feedback": self._parse_feedback(response),
            "full_response": response,
        }

    def deep_analysis(
        self,
        question_text: str,
        model_answer: str,
        user_answer: str,
        score: int,
    ) -> dict:
        """
        50점 미만 답안에 대한 Chain-of-Thought 심층 분석.

        Returns:
            error_types (list), analysis (str), improvement (str)
        """
        self._load()

        context = (
            f"[문제]\n{question_text}\n\n"
            f"[모범 답안]\n{model_answer}\n\n"
            f"[학생 답안]\n{user_answer[:700]}\n\n"
            f"[받은 점수] {score}점 / 100점"
        )
        prompt = ALPACA_PROMPT.format(
            instruction=DEEP_ANALYSIS_INSTRUCTION,
            input=context,
        )
        response = self._generate(prompt, max_new_tokens=300)

        # JSON 파싱 시도
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "error_types": data.get("error_types", ["내용 누락"]),
                    "analysis": data.get("analysis", response),
                    "improvement": data.get("improvement", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # 파싱 실패 시 전문을 분석으로 반환
        return {
            "error_types": ["내용 누락"],
            "analysis": response,
            "improvement": "",
        }


# ── 간단 테스트 ──────────────────────────────────────

if __name__ == "__main__":
    ev = WritingEvaluator()

    result = ev.evaluate(
        passage_text="선인장은 사막에 사는 식물이다.",
        question_text="선인장이 사막에서 살 수 있는 이유를 서술하시오.",
        model_answer="선인장은 줄기에 물을 저장하고 잎이 가시로 변해 수분 손실을 줄이기 때문이다.",
        user_answer="선인장은 물을 저장해서 살 수 있다.",
    )
    print(f"점수: {result['raw_score']}점 ({result['normalized_score']}점)")
    print(f"피드백: {result['feedback']}")

    if result["normalized_score"] < 50:
        analysis = ev.deep_analysis(
            question_text="선인장이 사막에서 살 수 있는 이유를 서술하시오.",
            model_answer="선인장은 줄기에 물을 저장하고 잎이 가시로 변해 수분 손실을 줄이기 때문이다.",
            user_answer="선인장은 물을 저장해서 살 수 있다.",
            score=result["normalized_score"],
        )
        print(f"오류 유형: {analysis['error_types']}")
        print(f"분석: {analysis['analysis']}")
