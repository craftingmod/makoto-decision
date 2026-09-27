# Minimal Decision Framework Architecture

## 1. 목표

이 프로젝트의 첫 버전은 **결정 문제와 결정 방식을 분리하는 최소한의 API**만 제공한다.

핵심 구조는 다음 세 가지뿐이다.

```text
Decision
   ↓
Evaluator
   ↓
DecisionResult
```

처음부터 inference backend 전체를 추상화하거나, tokenizer/runtime/capability 계층을 별도로 만들지 않는다.

필요한 기능은 각 `Evaluator` 내부에 구현하고, 실제 중복이 생겼을 때만 별도 abstraction으로 추출한다.

---

# 2. 설계 원칙

## 2.1 최소 public API

초기 public API는 가능한 한 작게 유지한다.

```text
Choice
Decision
DecisionResult
Evaluator
```

이 네 가지 정도면 충분하다.

그 외의 구조는 우선 implementation detail로 둔다.

예:

- prompt compiler
- tokenizer wrapper
- next-token runtime
- sequence scorer
- KV lifecycle
- multimodal runtime
- backend adapter

이들은 첫 버전에서 공통 abstraction으로 만들지 않는다.

---

## 2.2 추상화를 미리 만들지 않는다

다음과 같은 구조는 처음부터 만들지 않는다.

```text
Decision
   ↓
PromptCompiler
   ↓
CompiledDecision
   ↓
Runtime
   ↓
Tokenizer
   ↓
LogitsProvider
   ↓
Evaluator
```

확장성은 좋아 보이지만, 실제 요구가 없는 상태에서 구현하면 구조만 복잡해질 가능성이 높다.

대신 처음에는:

```text
Decision
   ↓
LlamaCppEvaluator
   ↓
DecisionResult
```

처럼 직접 구현한다.

나중에 다른 backend를 실제로 추가했을 때 중복되는 코드가 확인되면 그때 abstraction을 추출한다.

---

# 3. 기본 데이터 모델

## 3.1 Choice

```python
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Choice:
    key: str
    value: Any
    metadata: Mapping[str, Any] | None = None
```

예:

```python
Choice(
    key="yes",
    value="Y",
)
```

또는 Hash evaluator에서:

```python
Choice(
    key="reference_a",
    value="Reference A",
    metadata={
        "hash": reference_hash,
    },
)
```

`value`는 evaluator마다 다르게 해석할 수 있도록 너무 엄격하게 제한하지 않는다.

---

# 4. Decision

```python
@dataclass(frozen=True)
class Decision:
    choices: tuple[Choice, ...]
    context: Any
    question: str | None = None
    metadata: Mapping[str, Any] | None = None
```

예:

```python
decision = Decision(
    question="Is this frame suitable?",
    choices=(
        Choice("yes", "Y"),
        Choice("no", "N"),
    ),
    context=frame,
)
```

`Decision`은 어떤 backend에서 실행될지 모른다.

다음 개념을 알지 않는다.

- llama.cpp
- tokenizer
- logits
- KV cache
- MTMD
- transformers
- MLX
- vLLM

즉 `Decision`은 오직 "무엇을 결정할 것인가"만 표현한다.

---

# 5. DecisionResult

```python
@dataclass(frozen=True)
class DecisionResult:
    selected: str | None
    scores: Mapping[str, float]
    metadata: Mapping[str, Any] | None = None
```

예:

```python
DecisionResult(
    selected="yes",
    scores={
        "yes": 4.82,
        "no": 1.11,
    },
)
```

`scores`는 반드시 확률일 필요가 없다.

Evaluator에 따라 다음이 될 수 있다.

- raw logits
- log probability
- cosine similarity
- hash similarity
- heuristic score
- rule score

core에서는 score 의미를 강제하지 않는다.

---

# 6. Evaluator Protocol

공통 인터페이스는 매우 작게 유지한다.

```python
from typing import Protocol


class Evaluator(Protocol):
    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        ...
```

이 인터페이스만 만족하면 어떤 evaluator도 사용할 수 있다.

```text
Evaluator
   ├─ LlamaCppEvaluator
   ├─ HashEvaluator
   ├─ RuleEvaluator
   ├─ EmbeddingEvaluator
   └─ 기타 custom evaluator
```

---

# 7. LlamaCppEvaluator

첫 LLM 구현은 별도의 Runtime abstraction 없이 직접 만든다.

예:

```python
class LlamaCppEvaluator:
    def __init__(self, llama):
        self._llama = llama

    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        prompt = self._build_prompt(decision)

        logits = self._next_token_logits(prompt)

        scores: dict[str, float] = {}

        for choice in decision.choices:
            token_ids = self._tokenize_continuation(
                prompt,
                str(choice.value),
            )

            if len(token_ids) != 1:
                raise MultiTokenChoiceError(
                    f"{choice.value!r} tokenized to "
                    f"{token_ids}"
                )

            token_id = token_ids[0]

            scores[choice.key] = float(
                logits[token_id]
            )

        selected = max(
            scores,
            key=scores.get,
        )

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

중요한 점은 다음 helper들을 처음에는 public abstraction으로 만들지 않는 것이다.

```python
_build_prompt(...)
_next_token_logits(...)
_tokenize_continuation(...)
```

모두 `LlamaCppEvaluator`의 private implementation detail로 둔다.

---

# 8. Prompt 생성

초기 버전에서는 `PromptCompiler` 인터페이스를 만들 필요가 없다.

단순 private method로 시작한다.

```python
def _build_prompt(
    self,
    decision: Decision,
) -> str:
    ...
```

예:

```text
Determine the best answer.

Question:
Is this frame suitable?

Choices:
Y = yes
N = no

Answer:
```

나중에 prompt 종류가 여러 개로 늘어나고 실제 교체 요구가 생기면 그때 compiler abstraction을 추출한다.

예:

```python
class PromptCompiler(Protocol):
    ...
```

하지만 v0/v1에서는 필요하지 않다.

---

# 9. Tokenization

LLM decision에서 중요한 부분은 choice tokenization이다.

단순히:

```python
tokenize("Y")
```

만 하면 실제 continuation tokenization과 다를 수 있다.

예:

```text
"Y"
" Y"
```

또는:

```text
Answer:
```

뒤의 `Y`와:

```text
Answer: 
```

뒤의 `Y`가 다르게 tokenized될 수 있다.

따라서 evaluator 내부에서는 continuation 기준으로 검사하는 것이 좋다.

개념적으로:

```python
def _tokenize_continuation(
    self,
    prefix: str,
    continuation: str,
) -> list[int]:
    prefix_tokens = tokenize(prefix)
    full_tokens = tokenize(
        prefix + continuation
    )

    if full_tokens[:len(prefix_tokens)] != prefix_tokens:
        raise TokenBoundaryError(...)

    return full_tokens[len(prefix_tokens):]
```

이 로직도 첫 버전에서는 별도 tokenizer interface로 빼지 않는다.

---

# 10. Prefill / Logits

`LlamaCppEvaluator`는 llama.cpp 또는 llama-cpp-python fork가 제공하는 prefill API를 직접 사용한다.

예:

```python
def _next_token_logits(
    self,
    prompt: str,
):
    result = self._llama.create_chat_prefill(
        ...
    )

    return result.logits
```

즉 현재 필요한 backend가 llama.cpp 하나라면 그 backend를 그냥 직접 사용한다.

`NextTokenRuntime` 같은 공통 interface는 만들지 않는다.

---

# 11. HashEvaluator

LLM과 전혀 상관없는 evaluator도 동일한 API 아래 들어갈 수 있다.

```python
class HashEvaluator:
    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        query_hash = compute_hash(
            decision.context
        )

        scores: dict[str, float] = {}

        for choice in decision.choices:
            reference_hash = (
                choice.metadata["hash"]
            )

            scores[choice.key] = (
                hash_similarity(
                    query_hash,
                    reference_hash,
                )
            )

        selected = max(
            scores,
            key=scores.get,
        )

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

이 evaluator는 다음을 전혀 알 필요가 없다.

- prompt
- tokenizer
- logits
- llama.cpp

---

# 12. RuleEvaluator

간단한 heuristic이나 rule 기반 판정도 그대로 넣을 수 있다.

```python
class RuleEvaluator:
    def __init__(self, rules):
        self._rules = rules

    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        scores = {
            choice.key: 0.0
            for choice in decision.choices
        }

        for rule in self._rules:
            updates = rule(decision)

            for key, score in updates.items():
                scores[key] += score

        selected = max(
            scores,
            key=scores.get,
        )

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

---

# 13. EmbeddingEvaluator

필요하다면 나중에 같은 방식으로 추가한다.

```python
class EmbeddingEvaluator:
    def __init__(self, encoder):
        self._encoder = encoder

    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        ...
```

core API를 바꿀 필요는 없다.

---

# 14. 패키지 구조

초기에는 다음 정도로 충분하다.

```text
decision/
├── __init__.py
├── models.py
├── protocols.py
├── errors.py
└── evaluators/
    ├── __init__.py
    ├── llama_cpp.py
    ├── hash.py
    └── rule.py
```

굳이 다음 구조를 미리 만들 필요는 없다.

```text
runtime/
compiler/
backend/
capabilities/
tokenizer/
scoring/
```

실제 요구가 생겼을 때 추가한다.

---

# 15. 권장 public API

예:

```python
from decision import (
    Choice,
    Decision,
    DecisionResult,
    Evaluator,
)
```

backend-specific 구현은:

```python
from decision.evaluators import (
    LlamaCppEvaluator,
    HashEvaluator,
    RuleEvaluator,
)
```

정도로 둔다.

---

# 16. 사용 예시

## 16.1 Llama.cpp

```python
decision = Decision(
    question="Is this frame suitable?",
    choices=(
        Choice(
            key="yes",
            value="Y",
        ),
        Choice(
            key="no",
            value="N",
        ),
    ),
    context=frame,
)

evaluator = LlamaCppEvaluator(
    llama,
)

result = evaluator.evaluate(
    decision
)

print(result.selected)
print(result.scores)
```

---

## 16.2 Hash

```python
decision = Decision(
    question="Which reference is closer?",
    choices=(
        Choice(
            key="a",
            value="Reference A",
            metadata={
                "hash": hash_a,
            },
        ),
        Choice(
            key="b",
            value="Reference B",
            metadata={
                "hash": hash_b,
            },
        ),
    ),
    context=image,
)

result = HashEvaluator().evaluate(
    decision
)
```

---

# 17. Multi-token Choice

초기 `LlamaCppEvaluator`는 single-token choice만 지원해도 된다.

예:

```text
Y
N
```

또는 tokenizer상 실제로 하나의 token이 되는 값.

다음처럼 multi-token이 될 가능성이 있는 선택지는 v1에서 지원하지 않아도 된다.

```text
YES
NO
UNKNOWN
```

혹은:

```text
walking
standing
running
```

지원되지 않는 경우 명시적으로 에러를 낸다.

```python
raise MultiTokenChoiceError(...)
```

---

# 18. Multi-token 지원은 나중에

실제로 필요해질 때 별도의 evaluator를 추가한다.

예:

```python
class LlamaCppSequenceEvaluator:
    ...
```

여기에서는:

- token-by-token eval
- KV cache
- continuation append
- log probability accumulation

등을 처리할 수 있다.

하지만 이 기능을 현재 `LlamaCppEvaluator`에 억지로 넣지 않는다.

---

# 19. MTMD 지원

MTMD 역시 처음부터 공통 abstraction으로 만들지 않는다.

필요하다면 별도 evaluator를 만들 수 있다.

```python
class LlamaCppMultimodalEvaluator:
    ...
```

또는 기존 evaluator가 충분히 작다면:

```python
LlamaCppEvaluator(
    llama,
    multimodal_handler=...
)
```

정도로 확장할 수도 있다.

어느 쪽이 적절한지는 실제 구현 중복을 본 뒤 결정한다.

---

# 20. 언제 abstraction을 추출할 것인가

다음 상황이 실제로 나타났을 때만 추출한다.

예를 들어:

```text
LlamaCppEvaluator
MLXEvaluator
```

두 구현이 모두 아래 코드를 중복한다면:

```text
tokenize continuation
next-token logits
choice scoring
```

그때:

```python
class NextTokenRuntime(Protocol):
    ...
```

을 만드는 것을 고려한다.

즉:

```text
첫 번째 구현
→ 그냥 작성

두 번째 구현
→ 중복 관찰

세 번째 구현 또는 명확한 공통 경계
→ abstraction 추출
```

방식으로 진행한다.

---

# 21. 나중에 추출 가능한 구조

필요해지면 현재 private method를 그대로 분리할 수 있다.

현재:

```python
class LlamaCppEvaluator:
    def _build_prompt(...): ...
    def _tokenize_continuation(...): ...
    def _next_token_logits(...): ...
```

나중에:

```text
LlamaCppEvaluator
   ├─ PromptCompiler
   └─ NextTokenRuntime
```

로 옮길 수 있다.

즉 처음부터 abstraction을 만들지는 않지만,
**나중에 분리하기 쉬운 경계는 유지한다.**

---

# 22. v0 / v1 구현 범위

## 반드시 구현

- `Choice`
- `Decision`
- `DecisionResult`
- `Evaluator` protocol
- `LlamaCppEvaluator`

## 작은 예제 구현

- `HashEvaluator`
- `RuleEvaluator`

## 선택

- basic validation
- custom exception
- simple score normalization utility

---

# 23. v0 / v1에서 하지 않을 것

다음은 의도적으로 제외한다.

- universal backend abstraction
- backend registry
- automatic backend discovery
- PromptCompiler protocol
- NextTokenRuntime protocol
- tokenizer interface
- KV cache abstraction
- multi-token common runtime
- universal multimodal interface
- transformers adapter
- MLX adapter
- vLLM adapter
- generation abstraction
- streaming abstraction
- batching abstraction
- distributed inference

필요하기 전에는 구현하지 않는다.

---

# 24. 핵심 구조

최종적으로 첫 버전은 아래 구조만 명확하면 된다.

```text
                 ┌────────────────────┐
                 │      Decision      │
                 │                    │
                 │ context            │
                 │ choices            │
                 │ question(optional) │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │     Evaluator      │
                 └─────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    LlamaCppEvaluator  HashEvaluator   RuleEvaluator
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                 ┌────────────────────┐
                 │   DecisionResult   │
                 │                    │
                 │ selected           │
                 │ scores             │
                 │ metadata           │
                 └────────────────────┘
```

---

# 25. 설계 철학

이 프로젝트의 첫 번째 목표는 완벽한 abstraction이 아니다.

목표는:

> 다양한 방식의 판단기를 동일한 Decision API 아래 놓을 수 있는 가장 작은 구조를 만드는 것.

따라서 처음에는:

```text
Decision
Evaluator
DecisionResult
```

만 안정적으로 만든다.

LLM evaluator에 필요한:

```text
prompt
tokenization
prefill
logits
```

등은 우선 evaluator 내부에 둔다.

그리고 실제 backend 추가나 multi-token 지원 과정에서 중복이 명확해졌을 때:

```text
PromptCompiler
NextTokenRuntime
SequenceScoringRuntime
```

같은 abstraction을 하나씩 추출한다.

즉 이 설계의 핵심은:

> 추상화를 많이 만드는 것이 아니라, 나중에 추상화를 쉽게 뽑을 수 있도록 단순한 경계를 유지하는 것이다.

---

# 26. v0 실행 계획 (2026-09-27)

이 절은 위 예시 중 아직 정해지지 않은 입력 계약을 고정한다. 실제 패키지 이름은
`decision`이 아니라 이 저장소의 `makoto_decision`이다.

## 26.1 구현 순서와 담당

1. **Core:** `src/makoto_decision/`에 불변 `Choice`, `Decision`,
   `DecisionResult`와 `Evaluator` protocol을 둔다. 패키지 루트에서 네 이름을
   내보낸다. `Decision`은 비어 있거나 중복된 choice key를 거부한다. 점수는
   확률로 정규화하지 않는다.
2. **텍스트 Llama evaluator:** `makoto_decision.evaluators`에서
   `LlamaCppEvaluator`를 내보낸다. 주입된 llama 객체의 `tokenize(bytes,
   add_bos=..., special=False)`와 `prefill(token_ids)`만 사용한다. 별도
   llama-cpp-python 의존성을 추가하지 않는다. `context`와 choice `value`는
   문자열이어야 한다. 질문과 문자열 context, 선택지의 key/value로 한 개의
   고정된 텍스트 prompt를 만들고 `Answer:` 뒤의 `" " + value` continuation을
   비교한다. prefix와 `prefix + " " + value`를 같은 BOS 설정으로 토큰화해
   prefix 토큰열이 그대로 유지되는지 검사한다. 경계가 다르면
   `TokenBoundaryError`, continuation이 한 토큰이 아니면
   `MultiTokenChoiceError`를 낸다. 모든 선택지를 검사한 뒤 한 번 prefill하고
   해당 token ID의 raw logit으로 선택한다. 동점이면 입력의 첫 choice가 이긴다.
   llama 객체는 evaluator가 소유하지 않으며 종료하지 않는다.
3. **작은 예제:** `HashEvaluator`는 음수가 아닌 정수 `context`와 각 choice
   metadata의 음수가 아닌 정수 `hash`를 받아 XOR의 `bit_count()`에 음수를
   붙여 점수화한다.
   이미지 해시 생성은 호출자 책임이다. `RuleEvaluator`는 순서대로 실행할
   rule 호출 가능 객체들을 받고, choice key별 초기 점수 0에 반환된 점수를
   더한다. 알려지지 않은 key 또는 숫자가 아닌 점수는 거부한다.
4. **검증과 문서:** 모델 없는 fake llama로 token 경계, 다중 토큰 거부,
   prefill 한 번 호출, raw logit 선택을 확인한다. Hash/Rule 예제와 core
   validation에 작은 테스트를 둔다. README에 실제 import와 텍스트 사용
   예제를 추가하고 placeholder 테스트는 제거한다. `make lint-check`,
   `make test`, `make build`를 실행한다.

## 26.2 완료 기준과 중단 규칙

- 위의 public API, 예외, 예제 evaluator가 설치한 패키지에서 import된다.
- llama 모델이나 GPU가 없어도 단위 테스트가 실행된다. 실모델 추론 성공은
  단위 테스트의 결과로 주장하지 않는다.
- `prefill()`이 없는 llama 객체에는 추측성 호환 경로를 추가하지 않는다.
  그 경우 명확한 오류를 내고 실제 지원 대상이 확인될 때 확장한다.
- 이번 버전에는 chat template, MTMD, 이미지 입력, multi-token scoring,
  backend registry, score normalization 및 새 런타임 추상화를 추가하지 않는다.
