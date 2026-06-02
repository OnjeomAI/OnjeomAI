import re
import random
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
_OBJECTIVE_PATTERN = re.compile(r"(것은|않은\s*것은|알맞은\s*것은|적절한\s*것은|옳은\s*것은|틀린\s*것은)\s*\??")

_FALLBACK_QUESTIONS = {
    "FACTUAL":     "지문에서 확인할 수 있는 핵심 내용을 두 가지 이상 서술하시오.",
    "INFERENTIAL": "지문을 바탕으로 글쓴이의 의도나 숨겨진 의미를 추론하여 서술하시오.",
    "CRITICAL":    "지문의 주장에 대한 자신의 의견을 근거와 함께 서술하시오.",
    "CREATIVE":    "지문에 나온 핵심 어휘의 의미를 문맥에서 파악하고, 글의 논리 흐름과 연결지어 서술하시오.",
}

_TYPE_HINT = {
    "CREATIVE": "어휘 이해와 논리적 사고를 동시에 요구하는 주관식 서술형 문제",
}

GENERATABLE_TYPES = {"FACTUAL", "INFERENTIAL", "CRITICAL", "CREATIVE"}

_DEFAULT_TOPICS = [
    "환경 보호와 기후 변화",
    "독서의 중요성",
    "진로와 직업 선택",
    "우정과 갈등 해결",
    "디지털 기기와 청소년",
    "건강한 식습관",
    "봉사 활동의 의미",
    "전통문화와 현대 생활",
    "동물 보호와 생태계",
    "가족 간의 대화와 소통",
    "학교폭력과 예방",
    "과학 기술의 발전과 윤리",
    "미디어 리터러시",
    "꿈과 목표를 향한 노력",
    "다문화 사회와 배려",
]


def _clean(text: str) -> str:
    return _NON_KOREAN.sub("", text).strip()


def _strip_choices(text: str) -> str:
    lines = [l for l in text.split("\n") if not _CHOICE_LINE.match(l)]
    return "\n".join(lines).strip()


_SUBJECTIVE_ENDINGS = re.compile(r"(하시오|서술하시오|설명하시오|쓰시오|작성하시오|논하시오|분석하시오|비교하시오|주시오)\s*\.?\s*$")


def _fix_objective_question(question: str, reading_type: str) -> str:
    if _OBJECTIVE_PATTERN.search(question):
        return _FALLBACK_QUESTIONS.get(reading_type, "지문의 핵심 내용을 서술하시오.")
    if question and not _SUBJECTIVE_ENDINGS.search(question):
        return _FALLBACK_QUESTIONS.get(reading_type, "지문의 핵심 내용을 서술하시오.")
    return question


def _strip_answer_prefix(text: str) -> str:
    return re.sub(r"^(해설|정답)\s*[:：]?\s*", "", text).strip()


MIN_PASSAGE_LEN = 100
MAX_PASSAGE_LEN = 400
MIN_ANSWER_LEN = {1: 20, 2: 30, 3: 50, 4: 70, 5: 90}


class ProblemService:
    def generate(self, difficulty: int, reading_type: str, topic: str | None) -> dict:
        if reading_type not in GENERATABLE_TYPES:
            raise ValueError(f"문제 생성 불가 유형: {reading_type}. 생성 가능 유형: {sorted(GENERATABLE_TYPES)}")
        reading_ko = READING_TYPE_KO.get(reading_type, "사실적 이해")
        diff_desc = DIFFICULTY_DESC.get(difficulty, "중학교 수준의")
        if not topic:
            topic = random.choice(_DEFAULT_TOPICS)
        topic_hint = f"주제: {topic}\n"
        type_hint = _TYPE_HINT.get(reading_type, "지문을 바탕으로 한 주관식 서술형 질문 1개")

        prompt = f"""{topic_hint}반드시 한국어로만 작성하세요. 다음 조건에 맞는 국어 독해 문제를 만들어주세요.
- 난이도: {diff_desc}
- 독해 유형: {reading_ko}
- 문제 유형: 주관식 서술형
  * 질문은 반드시 "~하시오", "~설명하시오", "~서술하시오" 중 하나로 끝나야 합니다
  * "~것은?", "~않은 것은?", "~알맞은 것은?", "~적절한 것은?" 형식은 절대 금지합니다
  * ①②③④⑤ 같은 선택지를 절대 포함하지 마세요

[문제] 작성 지침: {type_hint}

아래 형식을 정확히 지켜주세요. 각 항목은 반드시 해당 태그 다음 줄에 작성하세요.

[지문]
(200~400자 분량의 한국어 지문)

[문제]
(주관식 서술형 질문 1개 — "~하시오"로 끝나는 문장)

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
        output = model_manager.generate(messages, max_new_tokens=700)
        result = self._parse_output(output, difficulty, reading_type)

        # 지문이 너무 짧으면 1회 재생성
        if len(result["passage_text"]) < MIN_PASSAGE_LEN:
            output = model_manager.generate(messages, max_new_tokens=700)
            result = self._parse_output(output, difficulty, reading_type)

        # 지문이 너무 길면 400자로 자르기
        if len(result["passage_text"]) > MAX_PASSAGE_LEN:
            result["passage_text"] = result["passage_text"][:MAX_PASSAGE_LEN].rsplit(" ", 1)[0]

        # 질문이 fallback이면 2차 호출로 생성
        if result["question_text"] in _FALLBACK_QUESTIONS.values():
            q_messages = [
                {"role": "system", "content": "다음 지문을 읽고 주관식 서술형 질문을 1개만 만드시오. 질문은 반드시 '~하시오', '~서술하시오', '~설명하시오', '~논하시오' 중 하나로 끝나야 합니다. 질문 외에 다른 말은 쓰지 마시오. 예시: '지문에서 ~의 의미를 설명하시오.' / '~에 대한 글쓴이의 관점을 서술하시오.'"},
                {"role": "user", "content": f"[지문]\n{result['passage_text']}"},
            ]
            q_output = model_manager.generate(q_messages, max_new_tokens=100)
            q_lines = [l.strip() for l in q_output.strip().split("\n") if l.strip()]
            q_text = q_lines[-1] if q_lines else ""
            question = _fix_objective_question(_clean(q_text), reading_type)
            if question and question not in _FALLBACK_QUESTIONS.values():
                result["question_text"] = question

        # 모범답안이 없거나 너무 짧으면 2차 QA 호출로 생성
        min_len = MIN_ANSWER_LEN.get(difficulty, 50)
        if not result["model_answer"] or len(result["model_answer"]) < min_len:
            qa_messages = [
                {"role": "system", "content": "다음 지문을 읽고 문항에 대한 모범답안을 반드시 3문장 이상으로 구체적으로 작성하시오. 짧게 쓰지 마시오."},
                {"role": "user", "content": f"[지문]\n{result['passage_text']}\n\n[문항]\n{result['question_text']}"},
            ]
            qa_output = model_manager.generate(qa_messages, max_new_tokens=400)
            m = re.search(r"해설\s*[:：]\s*(.+?)$", qa_output, re.S)
            if not m:
                m = re.search(r"정답\s*[:：]\s*(.+?)$", qa_output, re.S)
            answer = _clean(m.group(1)) if m else _clean(qa_output.strip())
            if answer:
                result["model_answer"] = _strip_answer_prefix(answer)
        else:
            result["model_answer"] = _strip_answer_prefix(result["model_answer"])

        return result

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
        question = _fix_objective_question(_clean(_strip_choices(question.strip())), reading_type)
        answer = _clean(answer.strip())

        if not passage:
            m = re.search(r"\[지문\]\s*(.+?)(?=\[문제\]|\[모범답안\]|$)", output, re.S)
            passage = _clean(m.group(1)) if m else ""
        if not question:
            m = re.search(r"\[문제\]\s*(.+?)(?=\[모범답안\]|$)", output, re.S)
            question = _fix_objective_question(_clean(m.group(1)), reading_type) if m else ""
        if not answer:
            m = re.search(r"\[모범답안\]\s*(.+?)$", output, re.S)
            answer = _clean(m.group(1)) if m else ""

        if not passage:
            passage = _clean(output[:400])
        if not question:
            question = _FALLBACK_QUESTIONS.get(reading_type, "지문의 핵심 내용을 서술하시오.")

        return {
            "passage_text": passage,
            "question_text": question,
            "model_answer": answer,
            "reading_type": reading_type,
            "difficulty": difficulty,
        }


problem_service = ProblemService()
