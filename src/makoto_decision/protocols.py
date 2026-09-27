from typing import Protocol

from .models import Decision, DecisionResult


class Evaluator(Protocol):
    def evaluate(self, decision: Decision) -> DecisionResult: ...
