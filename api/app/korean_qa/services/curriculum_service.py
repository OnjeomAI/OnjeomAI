import json
from app.core.model import model_manager

STAGES = ["사실적 이해", "추론적 이해", "비판적 이해", "창의적 이해"]
TOTAL_DAYS = 28


class CurriculumService:
    def generate(self, theta: float, daily_goal: int, weak_areas: list[str]) -> dict:
        stage_plan = self._build_stage_plan(theta, weak_areas)
        topics = self._generate_topics(theta, weak_areas, stage_plan)
        plans = self._build_plans(stage_plan, topics, daily_goal)

        return {
            "total_days": TOTAL_DAYS,
            "summary": self._summary(theta, weak_areas),
            "plans": plans,
        }

    def _build_stage_plan(self, theta: float, weak_areas: list[str]) -> list[str]:
        start_idx = self._start_stage_idx(theta)
        available = STAGES[start_idx:]

        weights = {s: 1.0 for s in available}
        for area in weak_areas:
            if area in weights:
                weights[area] = 1.5

        total_weight = sum(weights.values())
        plan = []
        remaining = TOTAL_DAYS

        for i, stage in enumerate(available):
            if i == len(available) - 1:
                days = remaining
            else:
                days = round(TOTAL_DAYS * weights[stage] / total_weight)
                remaining -= days
            plan.extend([stage] * days)

        return plan[:TOTAL_DAYS]

    def _generate_topics(self, theta: float, weak_areas: list[str], stage_plan: list[str]) -> dict:
        stage_counts = {}
        for s in stage_plan:
            stage_counts[s] = stage_counts.get(s, 0) + 1

        weak_str = ", ".join(weak_areas) if weak_areas else "없음"
        stage_str = "\n".join(f"- {s}: {d}일" for s, d in stage_counts.items())

        prompt = f"""국어 독해 학습 주제를 각 단계별로 생성하세요.

[학습자 수준] theta={theta:.2f}, 취약영역={weak_str}
[단계별 일수]
{stage_str}

각 단계의 일별 학습 주제를 JSON으로 답하세요. 주제는 간결하게 (예: "중심 내용 파악", "글쓴이 의도 분석"):
{{
  "사실적 이해": ["주제1", "주제2", ...],
  "추론적 이해": ["주제1", ...],
  "비판적 이해": ["주제1", ...],
  "창의적 이해": ["주제1", ...]
}}"""

        messages = [
            {"role": "system", "content": "당신은 국어 교육 전문가입니다. JSON만 출력하세요."},
            {"role": "user", "content": prompt},
        ]
        output = model_manager.generate(messages, max_new_tokens=512)

        try:
            start, end = output.find("{"), output.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(output[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _build_plans(self, stage_plan: list[str], topics: dict, daily_goal: int) -> list[dict]:
        stage_counters = {}
        plans = []

        for day, stage in enumerate(stage_plan, start=1):
            idx = stage_counters.get(stage, 0)
            topic_list = topics.get(stage, [])
            topic = topic_list[idx] if idx < len(topic_list) else f"{stage} 심화"

            plans.append({
                "day": day,
                "topic": topic,
                "problem_count": daily_goal,
                "focus_area": stage,
            })
            stage_counters[stage] = idx + 1

        return plans

    def _start_stage_idx(self, theta: float) -> int:
        if theta < -1:
            return 0
        elif theta < 0:
            return 1
        elif theta < 1:
            return 2
        return 3

    def _summary(self, theta: float, weak_areas: list[str]) -> str:
        stage = STAGES[self._start_stage_idx(theta)]
        weak = ", ".join(weak_areas) if weak_areas else "없음"
        return f"{stage}부터 시작하는 28일 커리큘럼 (취약영역: {weak})"


curriculum_service = CurriculumService()
