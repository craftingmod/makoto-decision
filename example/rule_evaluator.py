"""Simple rule based evaluator"""

from makoto_decision import Choices, Decision
from makoto_decision.evaluators import RuleEvaluator


def focus_rule(decision: Decision) -> dict[str, float]:
    in_focus = "in focus" in str(decision.context).lower()
    return {"keep": float(in_focus), "reject": float(not in_focus)}


decision = Decision(
    question="Should this frame be kept?",
    context="The subject is sharp and in focus.",
    choices=Choices.letters("keep", "reject"),
)
result = RuleEvaluator([focus_rule]).evaluate(decision)

print(result.selected)
print(dict(result.scores))
