import runpy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from makoto_decision import Choice, Decision, DecisionResult


class FakeLlama:
    def __init__(self) -> None:
        self.tokenize_calls: list[bytes] = []
        self.messages: list[dict[str, str]] = []

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        assert not add_bos
        assert not special
        self.tokenize_calls.append(text)
        return {b"Y": [1], b"N": [2]}[text]

    def create_chat_completion(self, **kwargs: Any) -> None:
        self.messages = cast(list[dict[str, str]], kwargs["messages"])
        processor = cast(
            Callable[[list[int], list[float]], list[float]],
            kwargs["logits_processor"][0],
        )
        processor([], [0.0, 2.5, -1.0])


def test_upstream_evaluator_uses_choice_targets_and_value_keys(monkeypatch: MonkeyPatch) -> None:
    numpy = ModuleType("numpy")

    def copy_array(scores: list[float], *, copy: bool) -> list[float]:
        assert copy
        return scores.copy()

    numpy.__dict__["array"] = copy_array
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.__dict__["Llama"] = object
    llama_cpp.__dict__["LogitsProcessorList"] = list
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp)

    example_path = Path(__file__).parents[1] / "example" / "upstream_llama_backend.py"
    evaluator_type = runpy.run_path(str(example_path))["UpstreamLlamaCppEvaluator"]
    llama = FakeLlama()
    decision = Decision(
        question="Pick one.",
        context="A short context.",
        choices=(Choice("yes", "Y"), Choice("no", "N")),
    )

    result: DecisionResult = evaluator_type(llama).evaluate(decision)

    assert llama.tokenize_calls == [b"Y", b"N"]
    assert "Choices:\nY: yes\nN: no" in llama.messages[0]["content"]
    assert result.selected == "yes"
    assert isinstance(result.scores, Mapping)
    assert result.scores == {"yes": 2.5, "no": -1.0}

    duplicate_llama = FakeLlama()
    duplicate_target_decision = Decision(
        question="Pick one.",
        context="A short context.",
        choices=(Choice("yes", "Y"), Choice("no", "Y")),
    )
    with pytest.raises(ValueError, match="choice targets must be unique"):
        evaluator_type(duplicate_llama).evaluate(duplicate_target_decision)
    assert duplicate_llama.tokenize_calls == []
    assert duplicate_llama.messages == []
