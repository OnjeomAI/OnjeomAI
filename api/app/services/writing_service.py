from app.core.model_loader import get_writing_evaluator
from app.schemas.writing import WritingEvaluateRequest, WritingEvaluateResponse


def evaluate_writing(req: WritingEvaluateRequest) -> WritingEvaluateResponse:
    evaluator = get_writing_evaluator()
    result = evaluator.evaluate(
        passage_text=req.passage_text,
        question_text=req.question_text,
        model_answer=req.model_answer,
        user_answer=req.user_answer,
    )
    return WritingEvaluateResponse(
        raw_score=result["raw_score"],
        normalized_score=result["normalized_score"],
        feedback=result["feedback"],
    )
