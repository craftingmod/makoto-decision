from typing import Any

from ..models import Decision, DecisionResult


def _nonnegative_hash(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


class HashEvaluator:
    def evaluate(self, decision: Decision) -> DecisionResult:
        query_hash = _nonnegative_hash(decision.context, "decision.context")
        scores: dict[str, float] = {}
        for choice in decision.choices:
            if choice.metadata is None or "hash" not in choice.metadata:
                raise ValueError(f"choice {choice.value!r} requires metadata['hash']")
            reference_hash = _nonnegative_hash(
                choice.metadata["hash"], f"choice {choice.value!r} metadata['hash']"
            )
            scores[choice.value] = -float((query_hash ^ reference_hash).bit_count())

        selected = max(scores, key=scores.__getitem__)
        return DecisionResult(selected=selected, scores=scores)
