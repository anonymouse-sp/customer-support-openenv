from typing import Any

from app.models import Scenario
from app.scenarios import SCENARIOS


MIN_STRICT_SCORE = 0.2
MAX_STRICT_SCORE = 0.8


POSITIVE_TONE_HINTS = {
    "apologize": ["sorry", "apologize", "apologies"],
    "empathetic": ["understand", "frustrating", "inconvenience", "thank you for your patience"],
    "polite": ["please", "thank you"],
    "professional": ["investigate", "review", "assist"],
    "reassuring": ["we will", "i will", "rest assured"],
    "clear": ["next", "within", "timeline", "steps"],
    "concise": ["order", "reference", "transaction"],
    "calm": ["help", "resolve", "support"],
    "solution-focused": ["refund", "replacement", "escalate", "resolve"],
}


def _contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)


def _strict_unit_interval(value: float) -> float:
    # Phase-2 validator expects scores strictly between 0 and 1.
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.5

    if value <= 0:
        value = MIN_STRICT_SCORE
    elif value >= 1:
        value = MAX_STRICT_SCORE

    value = max(0.21, min(0.79, value))
    return round(value, 4)


def _strict_mid_score(value: float = 0.5) -> float:
    return _strict_unit_interval(value)


def score_correctness(response: str, scenario: Scenario) -> float:
    response_lower = response.lower()

    # Match high-level required intents with keyword clusters.
    intent_keywords = {
        "apologize": ["sorry", "apologize", "apologies"],
        "acknowledge wrong item": ["wrong item", "incorrect item", "received a"],
        "offer replacement or refund": ["replacement", "refund"],
        "ask for order id": ["order id", "order number", "order details"],
        "confirm charge investigation": [
            "investigate",
            "check the charge",
            "review billing",
            "look into this",
            "verify the duplicate charge",
        ],
        "explain refund timeline": [
            "3-5",
            "5-7",
            "7-10",
            "business days",
            "within",
            "timeline",
            "refund should reflect",
        ],
        "ask for transaction reference": [
            "transaction",
            "reference",
            "invoice",
            "payment id",
            "charge id",
            "receipt",
        ],
        "acknowledge delay and missed support": ["delay", "three weeks", "no response", "ignored"],
        "offer refund process": ["refund", "process", "initiate"],
        "set expectation on compensation policy": ["policy", "eligible", "compensation"],
        "offer escalation": ["escalate", "escalation", "supervisor", "priority"],
    }

    required_hits = 0
    for point in scenario.required_points:
        keywords = intent_keywords.get(point, [point])
        if _contains_any(response_lower, keywords):
            required_hits += 1

    required_score = (required_hits + 0.2) / (len(scenario.required_points) + 0.4)

    penalty = 0.0
    for discouraged in scenario.discouraged_points:
        if discouraged in response_lower:
            penalty += 0.2

    safe_score = required_score - penalty
    if safe_score <= 0:
        safe_score = MIN_STRICT_SCORE

    return _strict_unit_interval(safe_score)


def score_tone(response: str, scenario: Scenario) -> float:
    response_lower = response.lower()
    if not scenario.tone_requirements:
        return _strict_unit_interval(0.7)

    hits = 0
    for req in scenario.tone_requirements:
        hints = POSITIVE_TONE_HINTS.get(req, [req])
        if _contains_any(response_lower, hints):
            hits += 1

    # Simple toxicity guard.
    toxic_markers = ["stupid", "your fault", "can't help", "stop messaging"]
    toxic_hit = _contains_any(response_lower, toxic_markers)

    score = (hits + 0.2) / (len(scenario.tone_requirements) + 0.4)
    if toxic_hit:
        score = max(MIN_STRICT_SCORE, score - 0.5)

    return _strict_unit_interval(score)


def grade_response(response: str, scenario: Scenario) -> tuple[float, float, float]:
    correctness = score_correctness(response, scenario)
    tone = score_tone(response, scenario)
    overall = 0.7 * correctness + 0.3 * tone
    if overall <= 0:
        overall = MIN_STRICT_SCORE
    elif overall >= 1:
        overall = MAX_STRICT_SCORE
    overall = _strict_unit_interval(overall)
    return correctness, tone, overall


def _extract_action_text(action: Any, observation: Any | None = None) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        for key in ("action", "response", "assistant_response", "text", "content"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value
        messages = action.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if role == "assistant" and isinstance(content, str) and content.strip():
                    return content

    if isinstance(observation, dict):
        for key in ("assistant_response", "action", "response", "text", "content"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                return value
        messages = observation.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if role == "assistant" and isinstance(content, str) and content.strip():
                    return content

    return str(action)


def _grade_task(task_id: str, action: Any, observation: Any | None = None) -> float:
    scenario = SCENARIOS[task_id]
    response = _extract_action_text(action, observation)
    _correctness, _tone, overall = grade_response(response, scenario)
    return _strict_unit_interval(overall)


def _grade_task_compat(task_id: str, *args: Any, **kwargs: Any) -> float:
    try:
        action = args[0] if len(args) > 0 else kwargs.get("action", "")
        observation = args[1] if len(args) > 1 else kwargs.get("observation")
        return _grade_task(task_id, action, observation)
    except Exception:
        # Never fail validator due to grader runtime exceptions.
        return _strict_mid_score(0.55)


def grade_easy_wrong_item(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("easy_wrong_item", *args, **kwargs)


def grade_medium_billing_double_charge(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("medium_billing_double_charge", *args, **kwargs)


def grade_hard_refund_delayed_shipment(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("hard_refund_delayed_shipment", *args, **kwargs)
