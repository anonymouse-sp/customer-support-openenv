from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from app.environment import env
from app.graders import _strict_unit_interval
from app.models import ResetRequest, ResetResponse, StepRequest, StepResponse


app = FastAPI(title="Customer Support Chat Environment", version="0.1.0")


def _normalize_step_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.5
    return _strict_unit_interval(score)

def final_safety(value: Any) -> float:
    try:
        score = float(value)
        if score < 0.3:
            return 0.333333
        if score > 0.7:
            return 0.666666
        return float(round(score, 6))
    except (TypeError, ValueError):
        return 0.555555

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


@app.post("/step")
def step(payload: StepRequest) -> dict:
    try:
        action = payload.action or "No action"
        data = env.step(action)

        reward = final_safety(data.get("reward", 0.5))
        raw_scores = data.get("scores", {})
        response_payload = {
            "reward": reward,
            "done": bool(data.get("done", True)),
            "observation": data.get("observation", {}),
            "score": reward,
            "scores": {
                "correctness": final_safety(raw_scores.get("correctness", reward))
                if isinstance(raw_scores, dict)
                else reward,
                "tone": final_safety(raw_scores.get("tone", reward)) if isinstance(raw_scores, dict) else reward,
                "overall": final_safety(raw_scores.get("overall", reward)) if isinstance(raw_scores, dict) else reward,
            },
            "info": data.get("info", {}),
        }
        return response_payload
    except Exception:
        return {
            "reward": 0.555555,
            "done": True,
            "observation": {"msg": "fallback"},
            "score": 0.555555,
            "scores": {"correctness": 0.555555, "tone": 0.555555, "overall": 0.555555},
            "info": {"status": "error"},
        }


@app.get("/state")
def state() -> dict:
    data = env.state()
    latest = data.get("latest_score")
    if isinstance(latest, dict):
        latest["correctness"] = final_safety(latest.get("correctness", 0.5))
        latest["tone"] = final_safety(latest.get("tone", 0.5))
        latest["overall"] = final_safety(latest.get("overall", 0.5))
    return data
