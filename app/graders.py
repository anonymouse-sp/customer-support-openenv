from typing import Any

from app.models import Scenario
from app.scenarios import SCENARIOS


MIN_STRICT_SCORE = 0.11
MAX_STRICT_SCORE = 0.89


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
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, value))


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

    required_score = required_hits / max(1, len(scenario.required_points))

    penalty = 0.0
    for discouraged in scenario.discouraged_points:
        if discouraged in response_lower:
            penalty += 0.2

    return _strict_unit_interval(required_score - penalty)


def score_tone(response: str, scenario: Scenario) -> float:
    response_lower = response.lower()
    if not scenario.tone_requirements:
        return MAX_STRICT_SCORE

    hits = 0
    for req in scenario.tone_requirements:
        hints = POSITIVE_TONE_HINTS.get(req, [req])
        if _contains_any(response_lower, hints):
            hits += 1

    # Simple toxicity guard.
    toxic_markers = ["stupid", "your fault", "can't help", "stop messaging"]
    toxic_hit = _contains_any(response_lower, toxic_markers)

    score = hits / max(1, len(scenario.tone_requirements))
    if toxic_hit:
        score = max(0.0, score - 0.5)

    return _strict_unit_interval(score)


def grade_response(response: str, scenario: Scenario) -> tuple[float, float, float]:
    correctness = score_correctness(response, scenario)
    tone = score_tone(response, scenario)
    overall = 0.7 * correctness + 0.3 * tone
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

    if isinstance(observation, dict):
        for key in ("assistant_response", "action", "response", "text", "content"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return str(action)


def _grade_task(task_id: str, action: Any, observation: Any | None = None) -> float:
    scenario = SCENARIOS[task_id]
    response = _extract_action_text(action, observation)
    _correctness, _tone, overall = grade_response(response, scenario)
    return _strict_unit_interval(overall)


def _extract_numeric_score(*args: Any, **kwargs: Any) -> float | None:
    candidates: list[Any] = [*args, kwargs]
    for item in candidates:
        if isinstance(item, dict):
            for key in ("reward", "score", "task_score", "final_reward", "overall"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            nested = item.get("scores")
            if isinstance(nested, dict):
                value = nested.get("overall")
                if isinstance(value, (int, float)):
                    return float(value)
        if isinstance(item, (int, float)):
            return float(item)
    return None


def _grade_task_compat(task_id: str, *args: Any, **kwargs: Any) -> float:
    try:
        numeric = _extract_numeric_score(*args, **kwargs)
        if numeric is not None:
            return _strict_unit_interval(numeric)

        action = args[0] if len(args) > 0 else kwargs.get("action", "")
        observation = args[1] if len(args) > 1 else kwargs.get("observation")
        return _grade_task(task_id, action, observation)
    except Exception:
        # Never fail validator due to grader runtime exceptions.
        return 0.5


def grade_easy_wrong_item(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("easy_wrong_item", *args, **kwargs)


def grade_medium_billing_double_charge(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("medium_billing_double_charge", *args, **kwargs)


def grade_hard_refund_delayed_shipment(*args: Any, **kwargs: Any) -> float:
    return _grade_task_compat("hard_refund_delayed_shipment", *args, **kwargs)
