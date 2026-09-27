import hashlib
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
)


@dataclass
class PrefillResultStub:
    logits: list[float]


class FakeLlama:
    def __init__(self, *, multi_token_target: str | None = None) -> None:
        self.multi_token_target: str | None = multi_token_target
        self.tokenize_calls: list[tuple[bytes, bool, bool]] = []
        self.chat_prefill_calls: list[list[dict[str, object]]] = []
        self.logits: list[float] = [0.0] * 12
        self.logits[10] = -3.5
        self.logits[11] = 2.0

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        self.tokenize_calls.append((text, add_bos, special))
        if text == b"A":
            return [10]
        if text == b"B":
            return [11, 12] if self.multi_token_target == "B" else [11]
        raise AssertionError(f"unexpected tokenization input: {text!r}")

    def create_chat_prefill(self, *, messages: list[dict[str, object]]) -> PrefillResultStub:
        self.chat_prefill_calls.append(messages)
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
    assert llama.tokenize_calls == [(b"A", False, False), (b"B", False, False)]
    assert llama.chat_prefill_calls == [
        [
            {
                "role": "user",
                "content": "A short context.\n\nChoose one.\n\nChoices:\nA: a\nB: b\n\n"
                "Respond with exactly one of: A, B",
            }
        ]
    ]


def test_llama_evaluator_rejects_multi_token_choice_targets() -> None:
    llama = FakeLlama(multi_token_target="B")

    with pytest.raises(MultiTokenChoiceError, match="one token"):
        LlamaCppEvaluator(llama).evaluate(_text_decision())

    assert llama.chat_prefill_calls == []


def test_llama_evaluator_rejects_duplicate_choice_targets() -> None:
    llama = FakeLlama()
    decision = Decision(
        question="Choose one.",
        context="A short context.",
        choices=(Choice("a", "A"), Choice("b", "A")),
    )

    with pytest.raises(ValueError, match="choice targets must be unique"):
        LlamaCppEvaluator(llama).evaluate(decision)

    assert llama.tokenize_calls == []
    assert llama.chat_prefill_calls == []


def test_llama_evaluator_requires_create_chat_prefill() -> None:
    with pytest.raises(TypeError, match="create_chat_prefill"):
        LlamaCppEvaluator(TokenizeOnly())


def test_hash_evaluator_selects_from_question_deterministically() -> None:
    question = "Which option should be selected?"
    decision = Decision(
        question=question,
        choices=(
            Choice("a", "A"),
            Choice("b", "B"),
            Choice("c", "C"),
        ),
        context="ignored by HashEvaluator",
    )

    result = HashEvaluator().evaluate(decision)
    index = int.from_bytes(hashlib.sha256(question.encode("utf-8")).digest()[:8], "big") % len(
        decision.choices
    )

    assert result.selected == decision.choices[index].value
    assert result.scores == {
        choice.value: float(choice_index == index)
        for choice_index, choice in enumerate(decision.choices)
    }
    same_question_different_context = Decision(
        question=question,
        choices=decision.choices,
        context=0,
    )
    assert HashEvaluator().evaluate(same_question_different_context).selected == result.selected


def test_hash_evaluator_requires_question() -> None:
    decision = Decision(choices=(Choice("a", "A"),), context="")
    with pytest.raises(ValueError, match="requires decision.question"):
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
    assert result.probabilities == pytest.approx({"a": 0.3775406687981454, "b": 0.6224593312018546})


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
