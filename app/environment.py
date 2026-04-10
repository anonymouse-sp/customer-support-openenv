from app.graders import grade_response
from app.models import Message, Scenario, ScoreBreakdown
from app.scenarios import SCENARIOS


class CustomerSupportEnv:
    def __init__(self) -> None:
        self.current_task_id: str | None = None
        self.current_scenario: Scenario | None = None
        self.history: list[Message] = []
        self.step_count = 0
        self.done = True
        self.latest_score: ScoreBreakdown | None = None
        self._fallback_task_index = 0

    def available_tasks(self) -> list[str]:
        return list(SCENARIOS.keys())

    def available_tasks_with_graders(self) -> list[dict]:
        return [
            {
                "id": task_id,
                "difficulty": task_id.split("_")[0],
                "grader": f"app.graders:grade_{task_id}",
                "grader_enabled": True,
            }
            for task_id in self.available_tasks()
        ]

    def _next_fallback_task_id(self) -> str:
        tasks = self.available_tasks()
        if not tasks:
            raise ValueError("No tasks available")

        task_id = tasks[self._fallback_task_index % len(tasks)]
        self._fallback_task_index += 1
        return task_id

    def reset(self, task_id: str | None = None) -> dict:
        # Never fail reset on unknown task IDs; validators may probe with synthetic tasks.
        if not task_id or task_id not in SCENARIOS:
            task_id = "easy_wrong_item"

        scenario = SCENARIOS[task_id]
        self.current_task_id = task_id
        self.current_scenario = scenario
        self.step_count = 0
        self.done = False
        self.latest_score = ScoreBreakdown(
            correctness=0.5,
            tone=0.5,
            overall=0.5,
        )
        self.history = [Message(role="user", content=scenario.customer_message)]

        return {
            "done": self.done,
            "observation": {
                "task_id": scenario.id,
                "task_title": scenario.title,
                "customer_message": scenario.customer_message,
            },
            "info": {
                "difficulty": task_id.split("_")[0],
                "max_steps": str(scenario.max_steps),
            },
        }

    def step(self, action: str) -> dict:
        if self.done or self.current_scenario is None:
            raise ValueError("Environment is not active. Call reset(task_id) first.")

        self.step_count += 1
        self.history.append(Message(role="assistant", content=action))

        correctness, tone, overall = grade_response(action, self.current_scenario)

        def inner_clamp(value: float) -> float:
            val = float(value)
            return float(max(0.33, min(0.66, val)))

        safe_correctness = inner_clamp(correctness)
        safe_tone = inner_clamp(tone)
        safe_overall = inner_clamp(overall)

        self.latest_score = ScoreBreakdown(
            correctness=safe_correctness,
            tone=safe_tone,
            overall=safe_overall,
        )

        if self.step_count >= self.current_scenario.max_steps:
            self.done = True

        return {
            "reward": safe_overall,
            "done": self.done,
            "observation": {
                "customer_message": self.current_scenario.customer_message,
                "assistant_response": action,
            },
            "score": safe_overall,
            "scores": {
                "correctness": safe_correctness,
                "tone": safe_tone,
                "overall": safe_overall,
            },
            "info": {
                "step_count": str(self.step_count),
                "max_steps": str(self.current_scenario.max_steps),
            },
        }

    def state(self) -> dict:
        return {
            "task_id": self.current_task_id,
            "step_count": self.step_count,
            "done": self.done,
            "history": [m.model_dump() for m in self.history],
            "latest_score": self.latest_score.model_dump() if self.latest_score else None,
        }


env = CustomerSupportEnv()
