import re
from app.core.model import model_manager


READING_TYPE_KO = {
    "FACTUAL":     "사실적 이해",
    "INFERENTIAL": "추론적 이해",
    "CRITICAL":    "비판적 이해",
    "CREATIVE":    "창의적 이해",
}

DIFFICULTY_DESC = {
    1: "초등학생 수준의 쉬운",
    2: "중학교 1~2학년 수준의",
    3: "중학교 3학년 수준의",
    4: "고등학교 1~2학년 수준의",
    5: "고등학교 3학년 수준의 심화",
}

_NON_KOREAN = re.compile(r"[^가-힣ᄀ-ᇿ㄰-㆏0-9a-zA-Z\s。．、，,\.\!\?\(\)\[\]\{\}\'\"·…「」『』""''-]")
_CHOICE_LINE = re.compile(r"^\s*[①②③④⑤]\s|^\s*[1-5][\.）)]\s")


def _clean(text: str) -> str:
    return _NON_KOREAN.sub("", text).strip()


def _strip_choices(text: str) -> str:
    """선택지 행(①②③④⑤, 1. 2. 등)을 제거한다."""
    lines = [l for l in text.split("\n") if not _CHOICE_LINE.match(l)]
    return "\n".join(lines).strip()


_TYPE_HINT = {
    "CREATIVE": "어휘 이해와 논리적 사고를 동시에 요구하는 주관식 서술형 문제 (예: '~의 의미를 문맥 속에서 파악하고, 글의 논리 흐름과 연결지어 서술하시오.')",
}


GENERATABLE_TYPES = {"FACTUAL", "INFERENTIAL", "CRITICAL", "CREATIVE"}


class ProblemService:
    def generate(self, difficulty: int, reading_type: str, topic: str | None) -> dict:
        if reading_type not in GENERATABLE_TYPES:
            raise ValueError(f"문제 생성 불가 유형: {reading_type}. 생성 가능 유형: {sorted(GENERATABLE_TYPES)}")
        reading_ko = READING_TYPE_KO.get(reading_type, "사실적 이해")
        diff_desc = DIFFICULTY_DESC.get(difficulty, "중학교 수준의")
        topic_hint = f"주제: {topic}\n" if topic else ""
        type_hint = _TYPE_HINT.get(reading_type, "지문을 바탕으로 한 주관식 서술형 질문 1개")

        prompt = f"""{topic_hint}반드시 한국어로만 작성하세요. 다음 조건에 맞는 국어 독해 문제를 만들어주세요.
- 난이도: {diff_desc}
- 독해 유형: {reading_ko}
- 문제 유형: 주관식 서술형 (①②③④⑤ 같은 선택지를 절대 포함하지 마세요)

[문제] 작성 지침: {type_hint}

아래 형식을 정확히 지켜주세요. 각 항목은 반드시 해당 태그 다음 줄에 작성하세요.

[지문]
(200~400자 분량의 한국어 지문)

[문제]
(주관식 서술형 질문 1개 — 선택지 없이 질문문만 작성)

[모범답안]
(2~4문장의 완전한 한국어 답변)"""

        if not model_manager.is_loaded:
            return {
                "passage_text": f"[MOCK] {reading_ko} 난이도 {difficulty} 지문입니다.",
                "question_text": "[MOCK] 지문의 핵심 내용을 서술하시오.",
                "model_answer": "[MOCK] 모범답안입니다.",
                "reading_type": reading_type,
                "difficulty": difficulty,
            }

        messages = [
            {"role": "system", "content": "당신은 한국어로 국어 독해 문제를 출제하는 전문 교사입니다. 반드시 한국어로만 답하세요."},
            {"role": "user", "content": prompt},
        ]
        output = model_manager.generate(messages, max_new_tokens=600)
        return self._parse_output(output, difficulty, reading_type)

    def _parse_output(self, output: str, difficulty: int, reading_type: str) -> dict:
        passage, question, answer = "", "", ""
        current = None

        for line in output.split("\n"):
            stripped = line.strip()
            if stripped in ("[지문]", "지문") or stripped.startswith("[지문]"):
                current = "passage"
                continue
            elif stripped in ("[문제]", "문제") or stripped.startswith("[문제]"):
                current = "question"
                continue
            elif stripped in ("[모범답안]", "모범답안") or stripped.startswith("[모범답안]"):
                current = "answer"
                continue

            if not stripped:
                continue

            if current == "passage":
                passage += stripped + "\n"
            elif current == "question":
                question += stripped + "\n"
            elif current == "answer":
                answer += stripped + "\n"

        passage = _clean(passage.strip())
        question = _clean(_strip_choices(question.strip()))
        answer = _clean(answer.strip())

        if not passage:
            m = re.search(r"\[지문\]\s*(.+?)(?=\[문제\]|\[모범답안\]|$)", output, re.S)
            passage = _clean(m.group(1)) if m else ""
        if not question:
            m = re.search(r"\[문제\]\s*(.+?)(?=\[모범답안\]|$)", output, re.S)
            question = _clean(m.group(1)) if m else ""
        if not answer:
            m = re.search(r"\[모범답안\]\s*(.+?)$", output, re.S)
            answer = _clean(m.group(1)) if m else ""

        if not passage:
            passage = _clean(output[:400])
        if not question:
            question = "지문의 핵심 내용을 서술하시오."
        if not answer:
            answer = "지문에서 핵심 개념을 찾아 두 문장 이상으로 서술하시오."

        return {
            "passage_text": passage,
            "question_text": question,
            "model_answer": answer,
            "reading_type": reading_type,
            "difficulty": difficulty,
        }


problem_service = ProblemService()
