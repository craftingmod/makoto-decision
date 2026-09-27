"""Simple hash based evaluator"""

from makoto_decision import Choice, Decision
from makoto_decision.evaluators import HashEvaluator


decision = Decision(
    question="Which option should this question choose?",
    context="HashEvaluator uses the question, not the context.",
    choices=(Choice("first", "First"), Choice("second", "Second")),
)
result = HashEvaluator().evaluate(decision)

assert result.selected == HashEvaluator().evaluate(decision).selected
assert result.selected is not None
assert result.scores[result.selected] == 1.0
print(result.selected)
print(dict(result.scores))
