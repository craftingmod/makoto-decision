import hashlib

from ..models import Decision, DecisionResult


class HashEvaluator:
    def evaluate(self, decision: Decision) -> DecisionResult:
        if decision.question is None:
            raise ValueError("HashEvaluator requires decision.question")

        digest = hashlib.sha256(decision.question.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(decision.choices)
        selected = decision.choices[index].value
        scores = {
            choice.value: float(choice_index == index)
            for choice_index, choice in enumerate(decision.choices)
        }
        return DecisionResult(selected=selected, scores=scores)
