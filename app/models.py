from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1)


class Scenario(BaseModel):
    id: str
    title: str
    customer_message: str
    required_points: List[str]
    discouraged_points: List[str] = Field(default_factory=list)
    tone_requirements: List[str] = Field(default_factory=list)
    max_steps: int = 1


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    input: Optional[Any] = None


class ResetResponse(BaseModel):
    done: bool
    observation: Dict[str, str]
    info: Dict[str, str]


class StepRequest(BaseModel):
    action: Optional[str] = Field(default=None, min_length=1, description="Agent response text")
    input: Optional[Any] = None


class ScoreBreakdown(BaseModel):
    correctness: float
    tone: float
    overall: float


class StepResponse(BaseModel):
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    observation: Dict[str, str]
    score: ScoreBreakdown
    info: Dict[str, str]


class StateResponse(BaseModel):
    task_id: Optional[str]
    step_count: int
    done: bool
    history: List[Message]
    latest_score: Optional[ScoreBreakdown]
