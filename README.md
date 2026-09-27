# Makoto-decision

![D.K.T](./docs/media/daekoto.jpg)

> Kihehe. Excellent choice.

### Choose like Hanuma Makoto: right or wrong, keep it cool.

`makoto-decision` is a small decision-scoring abstraction for single-token choices, with text and multimodal evaluator support.

```python
from llama_cpp import Llama

from makoto_decision import Choices, Decision
from makoto_decision.evaluators import LlamaCppEvaluator

llama = Llama(model_path="model.gguf")
decision = Decision(
    question="Is the esteemed Hanuma Makoto the greatest leader of Gehenna?",
    context="The esteemed Hanuma Makoto is the greatest leader of Gehenna, omnipotent and all-knowing in every endeavor.",
    choices=Choices.yes_or_no(),
)
evaluator = LlamaCppEvaluator(llama)

result = evaluator.evaluate(decision)
print(result.selected) # "yes"
# Soft-max token
print(result.probabilities) # {"yes": 0.962, "no": 0.038}
```

Define a context, an optional question, and choices such as `Choices.yes_or_no()`.
An evaluator returns the selected choice and scores.

## Evaluators

* `LlamaCppEvaluator` requires a [llama-cpp-python (JamePeng fork)](https://github.com/JamePeng/llama-cpp-python) build that includes `create_chat_prefill()` ([PR #183](https://github.com/JamePeng/llama-cpp-python/pull/183)).
* `RuleEvaluator` accepts Python functions such as:

   ```python
   def focus_rule(decision: Decision) -> dict[str, float]:
      in_focus = "in focus" in str(decision.context).lower()
      return {"keep": float(in_focus), "reject": float(not in_focus)}
   ```
* `HashEvaluator` selects a choice deterministically from the question, useful as a minimal custom evaluator example.

### Custom evaluators

For example, [examples/upstream_llama_evaluator.py](./example/upstream_llama_evaluator.py) demonstrates a minimal evaluator using upstream llama-cpp-python.

The upstream example is intentionally limited to text input and single-token choices. It captures the logits from the first step of a one-token completion, so unlike the `JamePeng backend` it is not a true prefill-only implementation.

This also means `makoto-decision` is not tied to a particular inference backend, any implementation that can map a Decision to a DecisionResult (`evaluate(Decision) -> DecisionResult`) can be used as an evaluator.

---

A custom backend only needs to implement the evaluator interface:

```python
class MyEvaluator:
    def evaluate(self, decision: Decision) -> DecisionResult:
        ...
```

## Examples

* Text: [`example/llama_evaluator.py`](example/llama_evaluator.py)
* Vision: [`example/llama_vision_evaluator.py`](example/llama_vision_evaluator.py)

Wrap media sources with `Image`, `Audio`, or `Video`; tuples can mix them with text:

```python
from makoto_decision import Choices, Decision, Image

decision = Decision(
    question="Is the subject in focus?",
    context=(Image("frame.png"), "A frame from the source video."),
    choices=Choices.yes_or_no(),
)
```

`HashEvaluator` requires a question and deterministically chooses a choice by taking the first
eight bytes of its SHA-256 digest modulo the number of choices. It ignores `Decision.context` and
returns `1.0` for the selected choice and `0.0` for the others. See the
[runnable example](example/hash_evaluator.py).

`RuleEvaluator` adds numeric scores returned by each rule, keyed by `Choice.value`. Its runnable
example is [`example/rule_evaluator.py`](example/rule_evaluator.py).

```python
from makoto_decision import Choice, Decision
from makoto_decision.evaluators import RuleEvaluator

rule_result = RuleEvaluator([lambda decision: {"yes": 1.0}]).evaluate(
    Decision(choices=Choices.yes_or_no(), context="")
)
```

`Decision` requires at least one choice, unique non-empty `Choice.value`s, and non-empty targets.
`LlamaCppEvaluator` additionally requires unique targets that each tokenize to exactly one token.
`DecisionResult.scores` keeps the evaluator's scores, and `DecisionResult.probabilities` applies
softmax across the available choices. Context can be text, an `Image`, `Audio`, or `Video`, or a
tuple combining text and media. Media wrappers accept a path, string source, or bytes.

## Developing

For project setup and development, see [installation](docs/installation.md),
[development](docs/development.md), and [publishing](docs/publishing.md).

<!-- This document follows common-doc-guidelines.md.
See github.com/jlevy/practical-prose and review guidelines before editing.
-->
