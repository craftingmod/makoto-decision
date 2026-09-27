from collections.abc import Mapping
from dataclasses import dataclass
from math import exp
from pathlib import Path
from string import ascii_uppercase
from typing import Any


@dataclass(frozen=True)
class Choice:
    value: str
    target: str
    metadata: Mapping[str, Any] | None = None


class Choices:
    @staticmethod
    def yes_or_no() -> tuple[Choice, ...]:
        return (Choice("yes", "Y"), Choice("no", "N"))

    @staticmethod
    def letters(*labels: str) -> tuple[Choice, ...]:
        if len(labels) > len(ascii_uppercase):
            raise ValueError("letter choices support at most 26 choices")
        return tuple(Choice(label, ascii_uppercase[index]) for index, label in enumerate(labels))


@dataclass(frozen=True)
class Image:
    source: str | Path | bytes


@dataclass(frozen=True)
class Audio:
    source: str | Path | bytes


@dataclass(frozen=True)
class Video:
    source: str | Path | bytes


ContextPart = str | Image | Audio | Video
Context = str | ContextPart | tuple[ContextPart, ...]


@dataclass(frozen=True)
class Decision:
    choices: tuple[Choice, ...]
    context: Context | int
    question: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        choices = tuple(self.choices)
        object.__setattr__(self, "choices", choices)
        if not choices:
            raise ValueError("decision must have at least one choice")
        for choice in choices:
            if not choice.value.strip():
                raise ValueError("choice values must be non-empty strings")
            if not choice.target.strip():
                raise ValueError("choice targets must be non-empty strings")
        values = [choice.value for choice in choices]
        if len(set(values)) != len(values):
            raise ValueError("choice values must be unique")


@dataclass(frozen=True)
class DecisionResult:
    selected: str | None
    scores: Mapping[str, float]
    metadata: Mapping[str, Any] | None = None

    @property
    def probabilities(self) -> Mapping[str, float]:
        if not self.scores:
            return {}

        max_score = max(self.scores.values())
        weights = {choice: exp(score - max_score) for choice, score in self.scores.items()}
        total = sum(weights.values())
        return {choice: weight / total for choice, weight in weights.items()}
