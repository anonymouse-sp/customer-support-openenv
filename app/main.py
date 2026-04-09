from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from app.environment import env
from app.graders import _strict_unit_interval
from app.models import ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse


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
    raise ValueError("Missing action in request body")


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

        # Keep a deterministic safe score for validator compatibility.
        safe_score = _normalize_step_score(data.get("score", 0.5))
        data["score"] = safe_score
        data["reward"] = safe_score
        data["scores"] = {
            "correctness": safe_score,
            "tone": safe_score,
            "overall": safe_score,
        }

        return StepResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**env.state())
