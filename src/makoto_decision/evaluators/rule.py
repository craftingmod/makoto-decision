from collections.abc import Callable, Iterable, Mapping
from numbers import Real

from ..models import Decision, DecisionResult


class RuleEvaluator:
    """Simple callback-based evaluator intended for examples and lightweight rules."""

    def __init__(self, rules: Iterable[Callable[[Decision], Mapping[str, float]]]) -> None:
        self._rules: tuple[Callable[[Decision], Mapping[str, float]], ...] = tuple(rules)

    def evaluate(self, decision: Decision) -> DecisionResult:
        scores = {choice.value: 0.0 for choice in decision.choices}
        for rule in self._rules:
            for key, score in rule(decision).items():
                if key not in scores:
                    raise ValueError(f"rule returned unknown choice key {key!r}")
                if isinstance(score, bool) or not isinstance(score, Real):
                    raise TypeError(f"rule score for {key!r} must be numeric")
                scores[key] += float(score)

        selected = max(scores, key=scores.__getitem__)
        return DecisionResult(selected=selected, scores=scores)
