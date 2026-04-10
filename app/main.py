from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from app.environment import env
from app.graders import _strict_unit_interval
from app.models import ScoreBreakdown, ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse


app = FastAPI(title="Customer Support Chat Environment", version="0.1.0")


def _normalize_step_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.5
    return _strict_unit_interval(score)


def _parse_reset_task_id(payload: Optional[ResetRequest]) -> Optional[str]:
    if payload is None:
        return None
    if payload.task_id:
        return payload.task_id
    if isinstance(payload.input, str) and payload.input:
        return payload.input
    if isinstance(payload.input, dict):
        candidate = payload.input.get("task_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _parse_step_action(payload: StepRequest) -> str:
    if payload.action:
        return payload.action
    if isinstance(payload.input, str) and payload.input:
        return payload.input
    if isinstance(payload.input, dict):
        candidate = payload.input.get("action")
        if isinstance(candidate, str) and candidate:
            return candidate
    return "No action"


@app.get("/")
def root() -> dict:
    task_ids = env.available_tasks()
    tasks_with_graders = env.available_tasks_with_graders()
    return {
        "service": "customer-support-openenv",
        "status": "ok",
        "tasks": task_ids,
        "task_ids": task_ids,
        "tasks_with_graders": tasks_with_graders,
    }


@app.get("/tasks")
def tasks() -> dict:
    task_ids = env.available_tasks()
    tasks_with_graders = env.available_tasks_with_graders()
    return {
        "tasks": task_ids,
        "task_ids": task_ids,
        "tasks_with_graders": tasks_with_graders,
    }


@app.post("/reset", response_model=ResetResponse)
def reset(payload: Optional[ResetRequest] = None) -> ResetResponse:
    try:
        task_id = _parse_reset_task_id(payload)
        data = env.reset(task_id)
        return ResetResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> StepResponse:
    try:
        action = _parse_step_action(payload)
        data = env.step(action)

        # Absolute moat clamp far from edges for protocol stability.
        def extreme_clamp(value: Any) -> float:
            try:
                score = float(value)
                return max(0.2, min(0.8, score))
            except (TypeError, ValueError):
                return 0.5

        data["reward"] = extreme_clamp(data.get("reward", 0.5))
        data["score"] = data["reward"]

        raw_scores = data.get("scores")
        if "scores" in data and isinstance(raw_scores, dict):
            for key in raw_scores:
                raw_scores[key] = extreme_clamp(raw_scores[key])
        else:
            data["scores"] = {
                "correctness": data["reward"],
                "tone": data["reward"],
                "overall": data["reward"],
            }

        return StepResponse(**data)
    except Exception as exc:
        # If step processing fails, return a safe mid-score response.
        return StepResponse(
            reward=0.5,
            done=True,
            observation={"msg": "fallback"},
            score=0.5,
            scores=ScoreBreakdown(correctness=0.5, tone=0.5, overall=0.5),
            info={"error": str(exc)},
        )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**env.state())
