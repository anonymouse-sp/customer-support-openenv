from app.models import Scenario


SCENARIOS = {
    "easy_wrong_item": Scenario(
        id="easy_wrong_item",
        title="Wrong Item Delivered",
        customer_message=(
            "I ordered a wireless mouse but received a keyboard. This is frustrating. "
            "I need the correct item quickly."
        ),
        required_points=[
            "apologize",
            "acknowledge wrong item",
            "offer replacement or refund",
            "ask for order id",
        ],
        discouraged_points=["blame customer", "deny issue"],
        tone_requirements=["polite", "empathetic", "clear"],
        max_steps=1,
    ),
    "medium_billing_double_charge": Scenario(
        id="medium_billing_double_charge",
        title="Double Charge Billing Complaint",
        customer_message=(
            "I was charged twice for the same subscription this month. "
            "Please fix this and confirm when I will get my money back."
        ),
        required_points=[
            "apologize",
            "confirm charge investigation",
            "explain refund timeline",
            "ask for transaction reference",
        ],
        discouraged_points=["promise impossible immediate bank settlement"],
        tone_requirements=["professional", "reassuring", "concise"],
        max_steps=1,
    ),
    "hard_refund_delayed_shipment": Scenario(
        id="hard_refund_delayed_shipment",
        title="Delayed Shipment and Escalated Refund",
        customer_message=(
            "My order is delayed by 3 weeks and support has ignored me. "
            "I want a full refund and compensation for the inconvenience."
        ),
        required_points=[
            "apologize",
            "acknowledge delay and missed support",
            "offer refund process",
            "set expectation on compensation policy",
            "offer escalation",
        ],
        discouraged_points=["hostile tone", "reject without explanation"],
        tone_requirements=["calm", "empathetic", "solution-focused"],
        max_steps=1,
    ),
}
