import json
from app.core.model import model_manager


class CurriculumService:
    def generate(self, theta: float, daily_goal: int, weak_areas: list[str]) -> dict:
        start_stage = self._start_stage(theta)
        weak_str = ", ".join(weak_areas) if weak_areas else "없음"

        prompt = f"""국어 독해 28일 학습 커리큘럼을 생성해주세요.

[학습자 정보]
- 독해 수준 (θ): {theta:.2f}
- 시작 단계: {start_stage}단계 (1:사실적 → 2:추론적 → 3:비판적 → 4:창의적)
- 일일 목표: {daily_goal}문제
- 취약 영역: {weak_str} (이 영역 비중을 높일 것)

아래 JSON 형식으로만 답하세요:
{{
  "total_days": 28,
  "summary": "커리큘럼 요약",
  "plans": [
    {{"day": 1, "topic": "주제", "problem_count": {daily_goal}, "focus_area": "사실적 이해"}},
    ...
  ]
}}"""

        messages = [
            {"role": "system", "content": "당신은 국어 독해 커리큘럼 설계 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        output = model_manager.generate(messages, max_new_tokens=1024)
        return self._parse(output, daily_goal)

    def _start_stage(self, theta: float) -> int:
        if theta < -1:
            return 1
        elif theta < 0:
            return 2
        elif theta < 1:
            return 3
        return 4

    def _parse(self, output: str, daily_goal: int) -> dict:
        try:
            start, end = output.find("{"), output.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(output[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "total_days": 28,
            "summary": "기본 커리큘럼",
            "plans": [
                {"day": i + 1, "topic": f"{i + 1}일차 학습", "problem_count": daily_goal, "focus_area": "사실적 이해"}
                for i in range(28)
            ],
        }


curriculum_service = CurriculumService()
