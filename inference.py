import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

REQUIRED_ENV = {
    "HF_TOKEN": HF_TOKEN,
}


def print_log(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=True)}")


def validate_env() -> None:
    missing = [k for k, v in REQUIRED_ENV.items() if not v]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


def _required_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def create_client() -> OpenAI:
    # Required by problem statement: use OpenAI client with API_BASE_URL and HF token.
    return OpenAI(
        base_url=_required_env("API_BASE_URL", DEFAULT_API_BASE_URL),
        api_key=_required_env("HF_TOKEN"),
    )


def build_task_prompt(task_id: str) -> str:
    base = (
        "You are a customer support assistant. "
        "Write a calm, empathetic, concise response in 4-6 sentences. "
        "Always include: (1) apology, (2) acknowledgement of issue, "
        "(3) concrete next action, (4) request for required reference details."
    )

    task_specific = {
        "easy_wrong_item": (
            "Mention replacement or refund options and ask for the order ID/order number."
        ),
        "medium_billing_double_charge": (
            "Confirm billing investigation, include a realistic refund timeline in business days, "
            "and ask for transaction reference/payment ID/invoice number."
        ),
        "hard_refund_delayed_shipment": (
            "Acknowledge delivery delay and prior missed support, explain refund process, "
            "state compensation policy expectation, and offer escalation."
        ),
    }

    return f"{base} {task_specific.get(task_id, '')}".strip()


def generate_response(client: OpenAI, task_id: str, customer_message: str) -> str:
    prompt = build_task_prompt(task_id)

    completion = client.chat.completions.create(
        model=_required_env("MODEL_NAME", DEFAULT_MODEL_NAME),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.0,
        max_tokens=220,
    )
    return completion.choices[0].message.content or ""


def run_task(http: httpx.Client, client: OpenAI, task_id: str) -> dict[str, Any]:
    reset_resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()

    customer_message = reset_data["observation"]["customer_message"]
    assistant_action = generate_response(client, task_id, customer_message)

    step_resp = http.post(f"{ENV_BASE_URL}/step", json={"action": assistant_action}, timeout=30)
    step_resp.raise_for_status()
    step_data = step_resp.json()

    numeric_score = float(step_data.get("score", step_data.get("reward", 0.5)))
    raw_scores = step_data.get("scores")
    if isinstance(raw_scores, dict):
        score_payload = raw_scores
    else:
        score_payload = {
            "correctness": numeric_score,
            "tone": numeric_score,
            "overall": numeric_score,
        }

    return {
        "task_id": task_id,
        "reward": step_data["reward"],
        "score": numeric_score,
        "scores": score_payload,
        "done": step_data["done"],
    }


def main() -> None:
    validate_env()
    client = create_client()
    model_name = _required_env("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_base_url = _required_env("API_BASE_URL", DEFAULT_API_BASE_URL)

    with httpx.Client() as http:
        tasks_resp = http.get(f"{ENV_BASE_URL}/tasks", timeout=30)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json().get("tasks", [])

        start_t = time.time()
        print_log(
            "START",
            {
                "run_id": int(start_t),
                "model": model_name,
                "api_base_url": api_base_url,
                "task_count": len(tasks),
            },
        )

        results = []
        for index, task_id in enumerate(tasks, start=1):
            task_start = time.time()
            result = run_task(http, client, task_id)
            results.append(result)
            print_log(
                "STEP",
                {
                    "index": index,
                    "task_id": task_id,
                    "reward": result["reward"],
                    "score": result["score"],
                    "correctness": result["scores"]["correctness"],
                    "tone": result["scores"]["tone"],
                    "duration_sec": round(time.time() - task_start, 3),
                },
            )

        avg_reward = sum(r["reward"] for r in results) / max(1, len(results))

        print_log(
            "END",
            {
                "total_tasks": len(results),
                "average_reward": round(avg_reward, 4),
                "total_duration_sec": round(time.time() - start_t, 3),
                "results": results,
            },
        )


if __name__ == "__main__":
    main()
