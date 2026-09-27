# Decision Framework Architecture Draft

## 1. 목표

이 프로젝트의 핵심 목표는 **결정 문제(Decision)를 특정 inference backend와 분리**하는 것이다.

라이브러리는 다음을 직접 책임지지 않는다.

- llama.cpp 전체 inference lifecycle
- transformers / MLX / vLLM 등 모든 backend 지원
- generation API
- streaming
- sampling
- backend별 KV cache 관리
- backend별 multimodal lifecycle

대신 다음만 공통 추상화한다.

1. 무엇을 결정해야 하는가
2. 가능한 선택지는 무엇인가
3. 어떤 evaluator가 이를 평가하는가
4. evaluator 결과를 어떤 공통 형식으로 반환하는가

즉:

```text
Decision
   ↓
Evaluator
   ↓
DecisionResult
```

를 라이브러리의 중심 구조로 둔다.

---

## 2. 핵심 원칙

### 2.1 Decision은 backend를 모른다

`Decision`은 llama.cpp, tokenizer, logits, KV cache를 알지 않는다.

```python
@dataclass(frozen=True)
class Decision:
    question: str
    choices: tuple["Choice", ...]
    context: object | None = None
```

예:

```python
decision = Decision(
    question="Is this frame suitable as a continuation frame?",
    choices=(
        Choice(key="yes", text="Y"),
        Choice(key="no", text="N"),
    ),
    context=frame,
)
```

---

### 2.2 Prompt는 Decision의 본질이 아니다

LLM evaluator에서는 prompt가 필요하지만, Hash evaluator나 Rule evaluator에는 필요하지 않다.

따라서:

```text
Decision
   ├─ LLM Evaluator
   │    └─ PromptCompiler
   ├─ Hash Evaluator
   ├─ Embedding Evaluator
   └─ Rule Evaluator
```

와 같이 구성한다.

`Decision.compile_prompt()`처럼 Decision 자체에 LLM 특화 책임을 넣는 것은 가능하면 피한다.

---

## 3. 기본 데이터 모델

### 3.1 Choice

```python
@dataclass(frozen=True)
class Choice:
    key: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)
```

예:

```python
Choice(
    key="yes",
    text="Y",
)
```

또는 Hash 기반 evaluator에서:

```python
Choice(
    key="reference_a",
    text="Reference A",
    metadata={
        "hash": "...",
    },
)
```

---

### 3.2 Decision

```python
@dataclass(frozen=True)
class Decision:
    question: str
    choices: tuple[Choice, ...]
    context: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

`context`는 evaluator가 판정할 실제 입력을 담는다.

예:

- text
- image
- frame
- embedding
- structured metadata
- arbitrary Python object

초기 버전에서는 context 타입을 과도하게 제한하지 않는다.

---

### 3.3 DecisionResult

모든 evaluator가 동일한 결과 타입을 반환하도록 한다.

```python
@dataclass(frozen=True)
class DecisionResult:
    selected: str | None
    scores: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)
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

확률 정규화는 evaluator마다 의미가 다를 수 있으므로 core에서는 강제하지 않는다.

즉 `scores`는 반드시 probability일 필요가 없다.

예:

- logits
- log-probability
- cosine similarity
- hash similarity
- rule weight
- custom score

---

## 4. Evaluator 인터페이스

가장 기본적인 인터페이스는 다음 정도로 제한한다.

```python
class DecisionEvaluator(Protocol):
    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        ...
```

이 인터페이스만 맞으면 어떤 방식의 decision engine도 붙일 수 있다.

```text
DecisionEvaluator
   ├─ NextTokenEvaluator
   ├─ SequenceEvaluator
   ├─ HashSimilarityEvaluator
   ├─ EmbeddingSimilarityEvaluator
   ├─ RuleEvaluator
   └─ CompositeEvaluator
```

---

# 5. LLM Evaluator 구조

LLM evaluator는 다시 다음 책임으로 나눈다.

```text
Decision
   ↓
PromptCompiler
   ↓
CompiledPrompt
   ↓
Runtime
   ↓
logits
   ↓
Choice Scorer
   ↓
DecisionResult
```

---

## 5.1 PromptCompiler

Prompt 생성은 LLM evaluator 내부의 별도 객체로 둔다.

```python
class PromptCompiler(Protocol):
    def compile(
        self,
        decision: Decision,
    ) -> "CompiledPrompt":
        ...
```

예:

```python
@dataclass(frozen=True)
class CompiledPrompt:
    prompt: str
    choices: tuple[Choice, ...]
```

예시 결과:

```text
You are evaluating whether the provided frame can be used
as a continuation frame.

Question:
Is this frame suitable as a continuation frame?

Choices:
Y = suitable
N = unsuitable

Answer:
```

---

# 6. Runtime 추상화

여기서 가장 중요한 부분은 **Backend 전체를 추상화하지 않는 것**이다.

다음 같은 거대한 인터페이스는 만들지 않는다.

```python
class Backend:
    def generate(...): ...
    def chat(...): ...
    def stream(...): ...
    def embed(...): ...
    def tokenize(...): ...
    def detokenize(...): ...
    def prefill(...): ...
    def kv_cache(...): ...
```

이렇게 만들면 Decision library가 inference framework가 된다.

대신 capability 단위로 나눈다.

---

## 6.1 NextTokenRuntime

첫 구현에서는 이것만 지원해도 충분하다.

```python
class NextTokenRuntime(Protocol):
    def tokenize_continuation(
        self,
        prefix: str,
        continuation: str,
    ) -> Sequence[int]:
        ...

    def next_token_logits(
        self,
        prompt: str,
    ) -> NDArray[np.floating]:
        ...
```

즉 runtime이 제공해야 하는 기능은 단 두 개다.

```text
prompt → next-token logits

(prefix, continuation) → continuation token ids
```

---

## 6.2 왜 단순 tokenize(text)가 아닌가

다음은 tokenizer에 따라 다를 수 있다.

```text
"Y"
" Y"
```

그리고:

```text
Answer:
```

뒤에 붙는 `"Y"`와:

```text
Answer: 
```

뒤에 붙는 `"Y"`의 tokenization이 달라질 수도 있다.

따라서 evaluator가 필요한 것은:

```python
tokenize("Y")
```

가 아니라:

```python
tokenize_continuation(
    prefix=prompt,
    continuation="Y",
)
```

이다.

---

## 6.3 일반적인 구현

예:

```python
def tokenize_continuation(
    prefix: str,
    continuation: str,
) -> Sequence[int]:
    prefix_tokens = tokenize(prefix)
    full_tokens = tokenize(prefix + continuation)

    if full_tokens[:len(prefix_tokens)] != prefix_tokens:
        raise TokenBoundaryError(
            "Continuation changes prefix tokenization"
        )

    return full_tokens[len(prefix_tokens):]
```

실제 backend에 따라 더 정확한 tokenizer-specific 구현이 필요할 수 있다.

---

# 7. NextTokenEvaluator

single-token choice용 가장 단순한 evaluator다.

```python
class NextTokenEvaluator:
    def __init__(
        self,
        runtime: NextTokenRuntime,
        compiler: PromptCompiler,
    ):
        self.runtime = runtime
        self.compiler = compiler

    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        compiled = self.compiler.compile(decision)

        logits = self.runtime.next_token_logits(
            compiled.prompt
        )

        scores: dict[str, float] = {}

        for choice in compiled.choices:
            token_ids = self.runtime.tokenize_continuation(
                compiled.prompt,
                choice.text,
            )

            if len(token_ids) != 1:
                raise MultiTokenChoiceError(
                    f"{choice.text!r} tokenized to {token_ids}"
                )

            scores[choice.key] = float(
                logits[token_ids[0]]
            )

        selected = max(scores, key=scores.get)

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

---

# 8. llama.cpp Adapter

llama.cpp 지원은 core에 깊게 박지 않는다.

얇은 adapter만 작성한다.

```python
class LlamaCppRuntime:
    def __init__(self, llama):
        self._llama = llama

    def tokenize_continuation(
        self,
        prefix: str,
        continuation: str,
    ) -> Sequence[int]:
        ...

    def next_token_logits(
        self,
        prompt: str,
    ) -> NDArray[np.floating]:
        result = self._llama.create_chat_prefill(
            ...
        )

        return result.logits
```

JamePeng fork의 `create_chat_prefill()`이 존재한다면 이 계층은 매우 얇아진다.

---

# 9. Multi-token Choice는 별도 capability로 분리

초기 API에서 multi-token continuation까지 억지로 일반화하지 않는다.

예:

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

이 token 하나가 아닐 수 있다.

이 경우 단순 next-token logits로 전체 sequence의 score를 계산할 수 없다.

따라서 별도 capability를 둔다.

```python
class SequenceScoringRuntime(Protocol):
    def score_continuation(
        self,
        prefix: str,
        continuation: str,
    ) -> float:
        ...
```

이 구현에는 backend에 따라 다음이 필요할 수 있다.

- repeated eval
- KV cache lifecycle
- continuation token append
- log-prob accumulation
- sequence length normalization

따라서 v1 core에서는 optional extension으로 취급한다.

---

# 10. Hash Similarity Evaluator

LLM과 전혀 관계없는 evaluator도 동일한 Decision API 위에서 동작한다.

예:

```python
class HashSimilarityEvaluator:
    def evaluate(
        self,
        decision: Decision,
    ) -> DecisionResult:
        query_hash = compute_hash(
            decision.context
        )

        scores = {}

        for choice in decision.choices:
            reference_hash = choice.metadata["hash"]

            scores[choice.key] = hash_similarity(
                query_hash,
                reference_hash,
            )

        selected = max(scores, key=scores.get)

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

이 구현은:

- tokenizer
- prompt
- logits
- llama.cpp

를 전혀 사용하지 않는다.

---

# 11. Embedding Similarity Evaluator

동일한 방식으로 embedding 기반 evaluator도 가능하다.

```python
class EmbeddingSimilarityEvaluator:
    def __init__(self, encoder):
        self.encoder = encoder

    def evaluate(self, decision):
        query = self.encoder.encode(
            decision.context
        )

        scores = {}

        for choice in decision.choices:
            reference = choice.metadata["embedding"]

            scores[choice.key] = cosine_similarity(
                query,
                reference,
            )

        selected = max(scores, key=scores.get)

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

---

# 12. Rule Evaluator

가장 단순한 rule 기반 decision도 지원할 수 있다.

```python
class RuleEvaluator:
    def __init__(self, rules):
        self.rules = rules

    def evaluate(self, decision):
        scores = {
            choice.key: 0.0
            for choice in decision.choices
        }

        for rule in self.rules:
            updates = rule(decision)

            for key, score in updates.items():
                scores[key] += score

        selected = max(scores, key=scores.get)

        return DecisionResult(
            selected=selected,
            scores=scores,
        )
```

---

# 13. Composite / Ensemble Evaluator

여러 evaluator를 합치는 것도 가능하다.

```python
ensemble = WeightedEvaluator([
    (llm_evaluator, 0.7),
    (hash_evaluator, 0.2),
    (rule_evaluator, 0.1),
])
```

개념적으로:

```text
                ┌─ LLM ───────┐
Decision ───────┼─ Hash ──────┼─→ Weighted Scores → Result
                └─ Rules ─────┘
```

단, 서로 다른 evaluator의 raw score scale이 다를 수 있다.

따라서 ensemble은 다음 중 하나를 명시적으로 선택해야 한다.

- raw score 사용
- min-max normalization
- softmax normalization
- z-score
- rank aggregation
- evaluator-specific normalization

core에서는 특정 normalization을 강제하지 않는다.

---

# 14. Capability 계층

초기 설계는 capability를 다음과 같이 단계적으로 둔다.

```text
Level 0
DecisionEvaluator
    └─ arbitrary evaluator

Level 1
NextTokenRuntime
    └─ single-token LLM decision

Level 2
SequenceScoringRuntime
    └─ multi-token choice scoring

Level 3
Multimodal Runtime
    └─ image / audio / video context

Level 4
Stateful Runtime
    └─ explicit KV lifecycle
```

처음부터 Level 4까지 통합하지 않는다.

---

# 15. Multimodal 확장

MTMD를 지원한다고 해서 core Decision이 multimodal backend를 알아야 하는 것은 아니다.

예:

```python
decision = Decision(
    question="Does this image match the target subject?",
    choices=(...),
    context=image,
)
```

LLM evaluator가 multimodal이면:

```text
Decision.context
    ↓
Multimodal Prompt Compiler
    ↓
MTMD Runtime
    ↓
logits
```

Hash evaluator이면:

```text
Decision.context
    ↓
pHash
    ↓
similarity
```

즉 같은 `Decision`을 서로 다른 evaluator로 평가할 수 있다.

---

# 16. 패키지 구조 제안

초기에는 다음 정도면 충분하다.

```text
decision/
├── __init__.py
├── models.py
├── protocols.py
├── errors.py
│
├── evaluators/
│   ├── __init__.py
│   ├── next_token.py
│   ├── hash.py
│   ├── rule.py
│   └── composite.py
│
├── llm/
│   ├── __init__.py
│   ├── compiler.py
│   └── runtime.py
│
└── adapters/
    └── llama_cpp.py
```

multi-token 구현이 실제로 필요해질 때:

```text
decision/
└── llm/
    └── sequence.py
```

정도를 추가한다.

---

# 17. Dependency 방향

중요한 dependency 방향은 다음과 같다.

```text
models
 ↑
protocols
 ↑
evaluators
 ↑
adapters
```

즉 core model은 backend adapter를 import하지 않는다.

금지:

```text
Decision
 ↓
llama.cpp
```

허용:

```text
LlamaCppRuntime
 ↓
Decision Protocols
```

---

# 18. v1 범위

첫 버전에서는 다음까지만 구현하는 것을 권장한다.

## Core

- `Choice`
- `Decision`
- `DecisionResult`
- `DecisionEvaluator`

## LLM

- `PromptCompiler`
- `CompiledPrompt`
- `NextTokenRuntime`
- `NextTokenEvaluator`

## Example evaluator

- `HashSimilarityEvaluator`
- `RuleEvaluator`

## Adapter

- `LlamaCppRuntime`

---

# 19. v1에서 하지 않을 것

다음은 의도적으로 제외한다.

- universal generation API
- streaming abstraction
- full chat API abstraction
- backend registry
- automatic backend discovery
- transformers adapter 전체 구현
- MLX adapter 전체 구현
- vLLM adapter 전체 구현
- automatic KV cache abstraction
- universal multimodal abstraction
- arbitrary multi-token scoring
- distributed inference
- batching abstraction

필요할 때만 추가한다.

---

# 20. 사용 예시

## 20.1 LLM Decision

```python
decision = Decision(
    question="Is this frame suitable?",
    choices=(
        Choice("yes", "Y"),
        Choice("no", "N"),
    ),
    context=frame,
)

runtime = LlamaCppRuntime(llama)

evaluator = NextTokenEvaluator(
    runtime=runtime,
    compiler=SimpleDecisionPromptCompiler(),
)

result = evaluator.evaluate(decision)

print(result.selected)
print(result.scores)
```

---

## 20.2 Hash Decision

```python
decision = Decision(
    question="Which reference is most similar?",
    choices=(
        Choice(
            "a",
            "Reference A",
            metadata={"hash": hash_a},
        ),
        Choice(
            "b",
            "Reference B",
            metadata={"hash": hash_b},
        ),
    ),
    context=image,
)

result = HashSimilarityEvaluator().evaluate(
    decision
)
```

---

## 20.3 동일한 Decision, 다른 Evaluator

가능하다면 이 형태가 이 라이브러리의 핵심 장점이다.

```python
decision = build_decision(...)

llm_result = llm_evaluator.evaluate(decision)
hash_result = hash_evaluator.evaluate(decision)
rule_result = rule_evaluator.evaluate(decision)
```

Decision을 정의하는 코드와 판단 엔진을 완전히 분리한다.

---

# 21. 설계 철학 요약

이 프로젝트는 "모든 inference backend를 추상화하는 프레임워크"가 아니다.

목표는 다음과 같다.

```text
Decision = 문제 정의

Evaluator = 판단 전략

Runtime = evaluator가 필요로 하는 최소 backend capability

Adapter = 특정 backend를 capability에 연결하는 얇은 접착층
```

특히 LLM 지원에서 가장 중요한 원칙은:

> backend 전체를 wrapper하지 말고, decision evaluator가 실제로 사용하는 capability만 wrapper한다.

초기에는:

```text
tokenize_continuation
next_token_logits
```

두 기능이면 충분하다.

multi-token scoring, KV lifecycle, MTMD 등은 실제 요구가 생겼을 때 독립 capability로 추가한다.

이 구조를 유지하면 LLM 기반 Jev-style evaluator뿐 아니라 hash, embedding, rules, heuristic, ensemble 같은 전혀 다른 decision mechanism도 동일한 API 아래 자연스럽게 공존할 수 있다.
