"""
글쓰기 채점 모델 추론 모듈
Hub: Onjeom/writing-ai (Llama-3.1-8B QLoRA, unsloth)
"""

import json
import re


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

CURRICULUM_ADJUST_INSTRUCTION = (
    "학생의 역량별 점수 이력을 분석하여 반드시 아래 JSON 형식으로만 응답하시오.\n\n"
    "응답 형식:\n"
    '{"adjustment_message": "사용자에게 보여줄 짧은 알림 메시지 (예: 추론적 독해 점수가 낮아요. 관련 문제를 먼저 배치했어요.)", '
    '"recommended_focus": "개선 방향 상세 설명"}'
)

COMPARE_ANSWERS_INSTRUCTION = (
    "학생의 이전 답변과 현재 답변을 비교하여 반드시 아래 JSON 형식으로만 응답하시오.\n\n"
    "응답 형식:\n"
    '{"growth_message": "성장 메시지 (예: 저번엔 이 키워드가 없었는데 이번엔 포함됐어요. 성장했어요!)", '
    '"analysis": "두 답변의 비교 분석"}'
)

WEAKNESS_REPORT_INSTRUCTION = (
    "학생의 역량별 점수를 분석하여 반드시 아래 JSON 형식으로만 응답하시오.\n\n"
    "응답 형식:\n"
    '{"report": "약점 분석 리포트 (2~3문장)", '
    '"recommendations": ["권장사항1", "권장사항2"]}'
)

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
        import torch  # noqa: F401  — 실제 로드 시에만 임포트
        from unsloth import FastLanguageModel
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(self._model)

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
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

    def _parse_json(self, text: str) -> dict:
        """응답에서 JSON 파싱. 실패 시 빈 dict 반환."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

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

    # ── 채점 ──────────────────────────────────────────────────────────────────

    def evaluate(self, passage_text: str, question_text: str,
                 model_answer: str, user_answer: str) -> dict:
        """2단계 LLM 채점. raw_score(1~4), normalized_score, feedback 반환."""
        self._load()
        instruction = (
            f"{RELAXED_INSTRUCTION}\n\n"
            f"[문제]\n{question_text}\n\n[모범 답안]\n{model_answer}"
        )
        prompt = ALPACA_PROMPT.format(instruction=instruction, input=user_answer[:700])
        response = self._generate(prompt)
        raw_score = self._parse_score(response)
        return {
            "raw_score": raw_score,
            "normalized_score": SCORE_MAP.get(raw_score, 50),
            "feedback": self._parse_feedback(response),
        }

    def deep_analysis(self, question_text: str, model_answer: str,
                      user_answer: str, score: int) -> dict:
        """50점 미만 답안 CoT 심층 분석."""
        self._load()
        context = (
            f"[문제]\n{question_text}\n\n[모범 답안]\n{model_answer}\n\n"
            f"[학생 답안]\n{user_answer[:700]}\n\n[받은 점수] {score}점 / 100점"
        )
        prompt = ALPACA_PROMPT.format(instruction=DEEP_ANALYSIS_INSTRUCTION, input=context)
        response = self._generate(prompt, max_new_tokens=300)
        data = self._parse_json(response)
        return {
            "error_types": data.get("error_types", ["내용 누락"]),
            "analysis": data.get("analysis", response),
            "improvement": data.get("improvement", ""),
        }

    # ── 동적 학습 경로 재조정 ─────────────────────────────────────────────────

    def curriculum_adjust(self, weak_competencies: list[str]) -> dict:
        """
        취약 역량 목록(한국어)을 받아 재조정 메시지와 권장 방향을 생성.

        Args:
            weak_competencies: 3회 연속 50점 미만인 역량 한국어 이름 목록
        Returns:
            adjustment_message, recommended_focus
        """
        self._load()
        context = f"취약 역량: {', '.join(weak_competencies)}"
        prompt = ALPACA_PROMPT.format(instruction=CURRICULUM_ADJUST_INSTRUCTION, input=context)
        response = self._generate(prompt, max_new_tokens=200)
        data = self._parse_json(response)

        weak_str = ', '.join(weak_competencies)
        return {
            "adjustment_message": data.get(
                "adjustment_message",
                f"{weak_str} 점수가 낮아요. 관련 문제를 먼저 배치했어요.",
            ),
            "recommended_focus": data.get(
                "recommended_focus",
                f"{weak_str} 문제를 집중적으로 연습하는 것을 권장합니다.",
            ),
        }

    # ── 답변 변화 추적 ────────────────────────────────────────────────────────

    def compare_answers(self, question_text: str, model_answer: str,
                        previous_answer: str, previous_score: int,
                        current_answer: str, current_score: int,
                        newly_included: list[str], still_missing: list[str]) -> dict:
        """
        이전·현재 답변을 비교하여 성장 메시지와 분석을 생성.

        Returns:
            growth_message, analysis
        """
        self._load()
        context = (
            f"[문제]\n{question_text}\n\n"
            f"[모범 답안]\n{model_answer}\n\n"
            f"[이전 답변] ({previous_score}점)\n{previous_answer[:700]}\n\n"
            f"[현재 답변] ({current_score}점)\n{current_answer[:700]}\n\n"
            f"[새로 포함된 키워드] {', '.join(newly_included) if newly_included else '없음'}\n"
            f"[여전히 누락된 키워드] {', '.join(still_missing) if still_missing else '없음'}"
        )
        prompt = ALPACA_PROMPT.format(instruction=COMPARE_ANSWERS_INSTRUCTION, input=context)
        response = self._generate(prompt, max_new_tokens=250)
        data = self._parse_json(response)

        score_diff = current_score - previous_score
        if newly_included:
            default_msg = (
                f"저번엔 '{newly_included[0]}' 키워드가 없었는데 이번엔 포함됐어요. 성장했어요!"
                if score_diff > 0
                else f"'{newly_included[0]}' 키워드를 새로 포함했지만 아직 보완할 부분이 있어요."
            )
        else:
            default_msg = (
                f"이번엔 점수가 {abs(score_diff)}점 올랐어요. 꾸준히 성장하고 있어요!"
                if score_diff > 0
                else "이번엔 점수가 내려갔어요. 다시 한번 개념을 확인해보세요."
            )

        return {
            "growth_message": data.get("growth_message", default_msg),
            "analysis": data.get("analysis", response),
        }

    # ── 약점 분석 리포트 ──────────────────────────────────────────────────────

    def weakness_report(self, competency_scores: dict[str, int]) -> dict:
        """
        역량별 점수를 받아 약점 분석 리포트와 권장사항을 생성.

        Args:
            competency_scores: {"사실적 독해": 75, "추론적 독해": 42, ...}
        Returns:
            report, recommendations
        """
        self._load()
        score_lines = "\n".join(
            f"- {name}: {score}점" for name, score in competency_scores.items()
        )
        context = f"역량별 평균 점수:\n{score_lines}"
        prompt = ALPACA_PROMPT.format(instruction=WEAKNESS_REPORT_INSTRUCTION, input=context)
        response = self._generate(prompt, max_new_tokens=300)
        data = self._parse_json(response)

        weak = [n for n, s in competency_scores.items() if s < 50]
        default_recommendations = [f"{n} 문제를 집중 연습하세요." for n in weak] or ["전반적인 학습을 꾸준히 이어나가세요."]

        return {
            "report": data.get("report", f"{', '.join(weak)} 영역에서 집중적인 학습이 필요합니다." if weak else "전반적으로 양호합니다."),
            "recommendations": data.get("recommendations", default_recommendations),
        }
