from app.models import Scenario


EPSILON = 1e-6


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
    return max(EPSILON, min(1.0 - EPSILON, value))


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
        return 1.0

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
