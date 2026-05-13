from pydantic import BaseModel


class CurriculumRequest(BaseModel):
    theta: float        # IRT 능력 추정치 (-3 ~ 3)
    daily_goal: int     # 일일 목표 문제 수 (5 / 10 / 20)
    weak_areas: list[str]  # 취약 역량 태그


class DailyPlan(BaseModel):
    day: int
    topic: str
    problem_count: int
    focus_area: str


class CurriculumResponse(BaseModel):
    total_days: int
    summary: str
    plans: list[DailyPlan]
