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

    def _next_fallback_task_id(self) -> str:
        tasks = self.available_tasks()
        if not tasks:
            raise ValueError("No tasks available")

        task_id = tasks[self._fallback_task_index % len(tasks)]
        self._fallback_task_index += 1
        return task_id

    def reset(self, task_id: str | None = None) -> dict:
        if not task_id:
            task_id = self._next_fallback_task_id()

        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task_id: {task_id}")

        scenario = SCENARIOS[task_id]
        self.current_task_id = task_id
        self.current_scenario = scenario
        self.step_count = 0
        self.done = False
        self.latest_score = None
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
        self.latest_score = ScoreBreakdown(
            correctness=correctness,
            tone=tone,
            overall=overall,
        )

        if self.step_count >= self.current_scenario.max_steps:
            self.done = True

        return {
            "reward": overall,
            "done": self.done,
            "observation": {
                "customer_message": self.current_scenario.customer_message,
                "assistant_response": action,
            },
            "score": overall,
            "scores": self.latest_score.model_dump(),
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
