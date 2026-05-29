from app.core.model import model_manager


READING_TYPE_KO = {
    "FACTUAL": "사실적 이해",
    "INFERENTIAL": "추론적 이해",
    "CRITICAL": "비판적 이해",
    "CREATIVE": "창의적 이해",
}

DIFFICULTY_DESC = {
    1: "초등학생 수준의 쉬운",
    2: "중학교 1~2학년 수준의",
    3: "중학교 3학년 수준의",
    4: "고등학교 1~2학년 수준의",
    5: "고등학교 3학년 수준의 심화",
}


class ProblemService:
    def generate(self, difficulty: int, reading_type: str, topic: str | None) -> dict:
        reading_ko = READING_TYPE_KO.get(reading_type, "사실적 이해")
        diff_desc = DIFFICULTY_DESC.get(difficulty, "중학교 수준의")
        topic_hint = f"주제: {topic}\n" if topic else ""

        prompt = f"""{topic_hint}다음 조건에 맞는 국어 독해 문제를 만들어주세요.
- 난이도: {diff_desc}
- 독해 유형: {reading_ko}

아래 형식을 정확히 지켜주세요.

[지문]
(200~400자 분량의 지문)

[문제]
(지문을 바탕으로 한 주관식 질문 1개)

[모범답안]
(2~4문장의 완전한 답변)"""

        messages = [
            {"role": "system", "content": "당신은 국어 독해 문제를 출제하는 전문 교사입니다."},
            {"role": "user", "content": prompt},
        ]
        output = model_manager.generate(messages, max_new_tokens=512)
        return self._parse_output(output, difficulty, reading_type)

    def _parse_output(self, output: str, difficulty: int, reading_type: str) -> dict:
        passage, question, answer = "", "", ""
        current = None

        for line in output.split("\n"):
            line = line.strip()
            if line == "[지문]":
                current = "passage"
            elif line == "[문제]":
                current = "question"
            elif line == "[모범답안]":
                current = "answer"
            elif current == "passage":
                passage += line + "\n"
            elif current == "question":
                question += line + "\n"
            elif current == "answer":
                answer += line + "\n"

        passage = passage.strip()
        question = question.strip()
        answer = answer.strip()

        if not passage or not question or not answer:
            passage = passage or output[:300]
            question = question or "지문의 핵심 내용을 서술하시오."
            answer = answer or "지문을 참고하여 답하시오."

        return {
            "passage_text": passage,
            "question_text": question,
            "model_answer": answer,
            "reading_type": reading_type,
            "difficulty": difficulty,
        }


problem_service = ProblemService()
