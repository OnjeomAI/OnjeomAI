from app.core.model import model_manager

# 유형별 키워드 점수 반영 비중 (나머지는 LLM 점수)
KEYWORD_WEIGHTS: dict[str, float] = {
    "FACTUAL":     0.60,
    "CREATIVE":    0.50,
    "INFERENTIAL": 0.30,
    "CRITICAL":    0.20,
}


class GradingService:
    def grade(
        self,
        passage: str,
        question: str,
        model_answer: str,
        keywords: list[dict],
        student_answer: str,
        reading_type: str = "FACTUAL",
    ) -> dict:
        stage1_score, missing_keywords, found_keywords = self._keyword_grade(
            student_answer, keywords
        )
        llm_score = self._llm_grade(
            passage, question, model_answer, student_answer,
            stage1_score, missing_keywords,
        )
        kw_weight = KEYWORD_WEIGHTS.get(reading_type, 0.5)
        final_score = max(0, min(100, int(stage1_score * kw_weight + llm_score * (1 - kw_weight))))
        feedback = self._score_based_feedback(final_score, missing_keywords)
        return {
            "score": final_score,
            "stage1_score": stage1_score,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "feedback": feedback,
        }

    def _keyword_grade(self, student_answer: str, keywords: list[dict]):
        total_weight = sum(k["weight"] for k in keywords)
        earned_weight = 0
        found, missing = [], []

        for kw in keywords:
            if kw["keyword"] in student_answer:
                earned_weight += kw["weight"]
                found.append(kw["keyword"])
            else:
                missing.append(kw)

        score = int((earned_weight / total_weight) * 100) if total_weight > 0 else 50
        return score, missing, found

    def _score_based_feedback(self, score: int, missing: list) -> str:
        if score >= 80:
            return "핵심을 잘 파악했어요! 심화 학습을 추천해요."
        elif score >= 50:
            missing_str = ", ".join(f'"{k["keyword"]}"' for k in missing)
            return f"방향은 맞지만 {missing_str} 부분이 빠졌어요."
        else:
            missing_str = ", ".join(f'"{k["keyword"]}"' for k in missing)
            return f"이 개념부터 다시 보세요. 누락된 키워드: {missing_str}"

    def _llm_grade(
        self, passage, question, model_answer, student_answer,
        stage1_score, missing_keywords,
    ) -> int:
        missing_str = ", ".join(f'"{k["keyword"]}"' for k in missing_keywords) or "없음"

        prompt = f"""다음 주관식 문제의 학습자 답변을 채점해주세요.

[지문]
{passage}

[문제]
{question}

[모범답안]
{model_answer}

[학습자 답변]
{student_answer}

[키워드 채점 결과]
점수: {stage1_score}점 / 누락 키워드: {missing_str}

문맥과 논리 흐름을 고려해 최종 점수를 조정하고 아래 형식으로만 답하세요.
최종점수: [숫자]"""

        messages = [
            {"role": "system", "content": "당신은 국어 독해 채점 전문 AI입니다."},
            {"role": "user", "content": prompt},
        ]
        output = model_manager.generate(messages, max_new_tokens=64)
        return self._parse_output(output, stage1_score)

    def _parse_output(self, output: str, fallback: int) -> int:
        for line in output.split("\n"):
            if line.startswith("최종점수:"):
                try:
                    return max(0, min(100, int(line.replace("최종점수:", "").strip())))
                except ValueError:
                    pass
        return fallback


grading_service = GradingService()
