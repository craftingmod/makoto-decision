from base64 import b64encode
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models import Audio, Context, ContextPart, Decision, DecisionResult, Image, Video


class TokenBoundaryError(ValueError):
    """Raised when a continuation changes the tokenization of its prompt."""


class MultiTokenChoiceError(ValueError):
    """Raised when a choice target is not exactly one token."""


class LlamaCppEvaluator:
    def __init__(self, llama: Any) -> None:
        tokenize = getattr(llama, "tokenize", None)
        create_chat_prefill = getattr(llama, "create_chat_prefill", None)
        if not callable(tokenize):
            raise TypeError(
                "llama must provide callable tokenize(bytes, add_bos=..., special=False)"
            )
        if not callable(create_chat_prefill):
            raise TypeError("llama must provide callable create_chat_prefill(messages=...)")
        self._tokenize: Callable[..., Any] = tokenize
        self._create_chat_prefill: Callable[..., Any] = create_chat_prefill

    def evaluate(self, decision: Decision) -> DecisionResult:
        messages = self._build_messages(decision)
        targets = [choice.target for choice in decision.choices]
        if len(targets) != len(set(targets)):
            raise ValueError("choice targets must be unique")

        choice_tokens: dict[str, int] = {}
        for choice, target in zip(decision.choices, targets, strict=True):
            tokens = self._tokenize(target.encode("utf-8"), add_bos=False, special=False)
            if len(tokens) != 1:
                raise MultiTokenChoiceError(
                    f"choice {choice.value!r} target must tokenize to one token; got {len(tokens)}"
                )
            choice_tokens[choice.value] = tokens[0]

        has_media = any(not isinstance(part, str) for part in self._context_parts(decision.context))
        try:
            prefill_result = self._create_chat_prefill(messages=messages)
        except Exception as error:
            if not has_media:
                raise
            raise RuntimeError(
                "Failed to prefill multimodal decision using the provided Llama instance. "
                "Ensure its model, mmproj, and chat handler support this media type."
            ) from error

        try:
            logits = prefill_result.logits
        except AttributeError as error:
            raise TypeError(
                "llama.create_chat_prefill() must return an object with a logits attribute"
            ) from error

        scores = {
            choice.value: float(logits[choice_tokens[choice.value]]) for choice in decision.choices
        }
        selected = max(scores, key=scores.__getitem__)
        return DecisionResult(selected=selected, scores=scores)

    @classmethod
    def _build_messages(cls, decision: Decision) -> list[dict[str, Any]]:
        parts = cls._context_parts(decision.context)
        prompt = cls._render_decision_prompt(decision)
        if all(isinstance(part, str) for part in parts):
            text = "\n\n".join([*(part for part in parts if isinstance(part, str)), prompt])
            return [{"role": "user", "content": text}]

        content = [cls._to_content_part(part) for part in parts]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _context_parts(context: Context | int) -> tuple[ContextPart, ...]:
        if isinstance(context, (str, Image, Audio, Video)):
            return (context,)
        if isinstance(context, tuple):
            return context
        raise TypeError("llama decisions require text or image, audio, and video context")

    @staticmethod
    def _render_decision_prompt(decision: Decision) -> str:
        sections: list[str] = []
        if decision.question is not None:
            sections.append(decision.question)
        sections.append(
            "Choices:\n"
            + "\n".join(f"{choice.target}: {choice.value}" for choice in decision.choices)
        )
        sections.append(
            "Respond with exactly one of: "
            + ", ".join(choice.target for choice in decision.choices)
        )
        return "\n\n".join(sections)

    @classmethod
    def _to_content_part(cls, part: object) -> dict[str, Any]:
        if isinstance(part, str):
            return {"type": "text", "text": part}
        if isinstance(part, Image):
            return {
                "type": "image_url",
                "image_url": {"url": cls._media_url(part.source, "image")},
            }
        if isinstance(part, Audio):
            return {
                "type": "audio_url",
                "audio_url": {"url": cls._media_url(part.source, "audio")},
            }
        if isinstance(part, Video):
            return {
                "type": "video_url",
                "video_url": {"url": cls._media_url(part.source, "video")},
            }
        raise TypeError(f"unsupported decision context part: {type(part)!r}")

    @staticmethod
    def _media_url(source: str | Path | bytes, media_type: str) -> str:
        if isinstance(source, bytes):
            encoded = b64encode(source).decode("ascii")
            return f"data:{media_type}/octet-stream;base64,{encoded}"
        return str(source)
