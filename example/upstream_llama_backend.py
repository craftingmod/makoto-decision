"""Custom evaluator example with upstream llama_cpp_python."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from llama_cpp import Llama, LogitsProcessorList

from makoto_decision import Decision, DecisionResult

if TYPE_CHECKING:
    import numpy.typing as npt

class UpstreamLlamaCppEvaluator:
    """Minimal text-only evaluator for upstream llama-cpp-python.

    This is an example custom backend, not an integrated backend.

    Limitations:
    - text-only context
    - standalone single-token choice targets
    - uses a one-token completion to capture the first-token logits
    """

    def __init__(self, llama: Llama) -> None:
        self._llama = llama

    def evaluate(self, decision: Decision) -> DecisionResult:
        if not isinstance(decision.context, str):
            raise TypeError(
                "UpstreamLlamaCppEvaluator example only supports text context"
            )

        targets = [choice.target for choice in decision.choices]
        if len(targets) != len(set(targets)):
            raise ValueError("choice targets must be unique")
        token_ids = [self._choice_token_id(target) for target in targets]

        prompt = self._render_prompt(decision)

        captured_logits: npt.NDArray[np.single] | None = None

        def capture_logits(
            input_ids: npt.NDArray[np.intc],
            scores: npt.NDArray[np.single],
        ) -> npt.NDArray[np.single]:
            nonlocal captured_logits

            if captured_logits is None:
                captured_logits = np.array(scores, copy=True)

            # Do not modify sampling.
            return scores

        self._llama.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=1,
            temperature=0.0,
            logits_processor=LogitsProcessorList([capture_logits]),
        )

        if captured_logits is None:
            raise RuntimeError("Failed to capture next-token logits")

        scores = {
            choice.value: float(captured_logits[token_id])
            for choice, token_id in zip(decision.choices, token_ids, strict=True)
        }
        selected = max(scores, key=scores.__getitem__)

        return DecisionResult(
            selected=selected,
            scores=scores,
        )

    def _choice_token_id(self, target: str) -> int:
        tokens = self._llama.tokenize(
            target.encode("utf-8"),
            add_bos=False,
            special=False,
        )

        if len(tokens) != 1:
            raise ValueError(
                f"Choice target must encode to exactly one token: "
                f"{target!r} encoded to {len(tokens)} tokens"
            )

        return tokens[0]

    @staticmethod
    def _render_prompt(decision: Decision) -> str:
        choices = "\n".join(
            f"{choice.target}: {choice.value}"
            for choice in decision.choices
        )

        targets = ", ".join(
            choice.target
            for choice in decision.choices
        )

        return (
            f"{decision.context}\n\n"
            f"{decision.question}\n\n"
            f"Choices:\n"
            f"{choices}\n\n"
            f"Respond with exactly one of: {targets}"
        )
