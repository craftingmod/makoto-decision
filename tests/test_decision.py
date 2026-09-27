from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest

from makoto_decision import Choice, Decision
from makoto_decision.evaluators import (
    HashEvaluator,
    LlamaCppEvaluator,
    MultiTokenChoiceError,
    RuleEvaluator,
    TokenBoundaryError,
)


@dataclass
class PrefillResultStub:
    logits: list[float]


class FakeLlama:
    def __init__(self, *, boundary: str = "valid") -> None:
        self.boundary: str = boundary
        self.tokenize_calls: list[tuple[bytes, bool, bool]] = []
        self.prefill_calls: list[list[int]] = []
        self.logits: list[float] = [0.0] * 12
        self.logits[10] = -3.5
        self.logits[11] = 2.0

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        self.tokenize_calls.append((text, add_bos, special))
        if text.endswith(b"Answer:"):
            return [1, 2]
        if text.endswith(b" A"):
            return [1, 3, 10] if self.boundary == "mismatch" else [1, 2, 10]
        if text.endswith(b" B"):
            return [1, 2, 11, 12] if self.boundary == "multi" else [1, 2, 11]
        raise AssertionError(f"unexpected tokenization input: {text!r}")

    def prefill(self, token_ids: list[int]) -> PrefillResultStub:
        self.prefill_calls.append(token_ids)
        return PrefillResultStub(self.logits)


class TokenizeOnly:
    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        return [len(text) + int(add_bos) + int(special)]


def _text_decision() -> Decision:
    return Decision(
        question="Choose one.",
        context="A short context.",
        choices=(Choice("a", "A"), Choice("b", "B")),
    )


def test_decision_requires_nonempty_unique_keys() -> None:
    with pytest.raises(ValueError, match="at least one choice"):
        Decision(choices=(), context="")
    with pytest.raises(ValueError, match="unique"):
        Decision(choices=(Choice("same", "A"), Choice("same", "B")), context="")


def test_llama_evaluator_prefills_once_and_selects_raw_logit() -> None:
    llama = FakeLlama()

    result = LlamaCppEvaluator(llama).evaluate(_text_decision())

    assert result.selected == "b"
    assert result.scores == {"a": -3.5, "b": 2.0}
    assert len(llama.prefill_calls) == 1
    assert llama.prefill_calls == [[1, 2]]
    assert all(add_bos and not special for _, add_bos, special in llama.tokenize_calls)


@pytest.mark.parametrize(
    ("boundary", "exception", "message"),
    [
        ("mismatch", TokenBoundaryError, "token boundary"),
        ("multi", MultiTokenChoiceError, "one token"),
    ],
)
def test_llama_evaluator_rejects_invalid_continuations(
    boundary: str, exception: type[ValueError], message: str
) -> None:
    llama = FakeLlama(boundary=boundary)

    with pytest.raises(exception, match=message):
        LlamaCppEvaluator(llama).evaluate(_text_decision())

    assert llama.prefill_calls == []


def test_llama_evaluator_requires_prefill() -> None:
    with pytest.raises(TypeError, match="prefill"):
        LlamaCppEvaluator(TokenizeOnly())


def test_hash_evaluator_scores_nonnegative_integer_hashes() -> None:
    decision = Decision(
        choices=(
            Choice("a", "A", {"hash": 0b1011}),
            Choice("b", "B", {"hash": 0b1111}),
        ),
        context=0b1010,
    )

    result = HashEvaluator().evaluate(decision)

    assert result.selected == "a"
    assert result.scores == {"a": -1.0, "b": -2.0}


@pytest.mark.parametrize("value", [-1, True])
def test_hash_evaluator_rejects_invalid_hashes(value: int | bool) -> None:
    decision = Decision(choices=(Choice("a", "A", {"hash": value}),), context=0)

    with pytest.raises((TypeError, ValueError)):
        HashEvaluator().evaluate(decision)


def test_rule_evaluator_accumulates_scores() -> None:
    decision = _text_decision()
    evaluator = RuleEvaluator(
        [
            lambda _: {"a": 1.0, "b": 0.5},
            lambda _: {"b": 1.0},
        ]
    )

    result = evaluator.evaluate(decision)

    assert result.selected == "b"
    assert result.scores == {"a": 1.0, "b": 1.5}


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        ({"other": 1}, ValueError, "unknown choice key"),
        ({"a": "high"}, TypeError, "must be numeric"),
    ],
)
def test_rule_evaluator_rejects_invalid_scores(
    updates: dict[str, object], exception: type[Exception], message: str
) -> None:
    evaluator = RuleEvaluator([lambda _: cast(Mapping[str, float], updates)])

    with pytest.raises(exception, match=message):
        evaluator.evaluate(_text_decision())
