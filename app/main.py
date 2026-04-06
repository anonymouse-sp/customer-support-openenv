from fastapi import FastAPI, HTTPException

from app.environment import env
from app.models import ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse


app = FastAPI(title="Customer Support Chat Environment", version="0.1.0")


@app.get("/")
def root() -> dict:
    return {
        "service": "customer-support-openenv",
        "status": "ok",
        "tasks": env.available_tasks(),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": env.available_tasks()}


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    try:
        data = env.reset(payload.task_id)
        return ResetResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> StepResponse:
    try:
        data = env.step(payload.action)
        return StepResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**env.state())
