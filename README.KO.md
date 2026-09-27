# Makoto-decision

![대코토](./docs/media/daekoto.jpg)

> 키히히힛. 좋은 선택이야.

### [하누마 마코토](https://namu.wiki/w/%ED%95%98%EB%88%84%EB%A7%88%20%EB%A7%88%EC%BD%94%ED%86%A0)처럼 선택하기: 맞든 틀리든, 쿨하게.

`makoto-decision`은 문장 및 미디어를 지원하는 하나의 토큰으로 선택하는 작은 의사 스코어링 구현 python 라이브러리입니다.

```python
from llama_cpp import Llama

from makoto_decision import Choices, Decision
from makoto_decision.evaluators import LlamaCppEvaluator

llama = Llama(model_path="model.gguf")
decision = Decision(
    question="하누마 마코토님은 게헨나의 가장 위대한 지도자입니까?",
    context="하누마 마코토님께서는 게헨나의 가장 위대한 지도자이며 모든 일에 전지전능하다.",
    choices=Choices.yes_or_no(),
)
evaluator = LlamaCppEvaluator(llama)

result = evaluator.evaluate(decision)
print(result.selected) # "yes"
 # 다음 토큰 logic에서 softmax한 조건부 확률 비스무리한 거
print(result.probabilities) # {"yes": 0.962, "no": 0.038}

```

문맥, 선택적인 질문, `Choices.yes_or_no()` 같은 선택지를 정의합니다.
평가기(evaluator)는 선택된 항목과 점수를 반환합니다.

<sub>한국어 모델은 [Mica 4B](https://huggingface.co/sky7350/Mica-v0.1-4B)를 추천해요 (단, 문장만 지원.)</sub>

## 평가기

* `LlamaCppEvaluator`에는 `create_chat_prefill()`이 포함된 [llama-cpp-python(JamePeng 포크)](https://github.com/JamePeng/llama-cpp-python) 빌드가 필요합니다([PR #183](https://github.com/JamePeng/llama-cpp-python/pull/183)).
* `RuleEvaluator`는 다음과 같은 Python 함수를 받습니다.

   ```python
   def focus_rule(decision: Decision) -> dict[str, float]:
      in_focus = "in focus" in str(decision.context).lower()
      return {"keep": float(in_focus), "reject": float(not in_focus)}
   ```
* `HashEvaluator`는 질문을 바탕으로 항상 같은 선택지를 고릅니다, 커스텀 Evaluator 구현 예시로 쓰기 적당합니다.

### 커스텀 평가기

예를 들어, [examples/upstream_llama_evaluator.py](./example/upstream_llama_evaluator.py)는 순정 `llama-cpp-python`을 사용한 최소한의 evaluator 구현 예시를 보여줍니다.

순정 `llama-cpp-python` 예시는 의도적으로 텍스트 입력과 single-token choice만 지원하도록 제한되어 있습니다. 또한 1-token completion의 첫 단계에서 logits를 캡처하는 방식이므로, `JamePeng backend`와 달리 진정한 prefill-only 구현은 아닙니다.

이는 동시에 `makoto-decision`이 특정 inference backend에 종속되지 않는다는 예제가 됩니다. `Decision`을 `DecisionResult`로 변환할 수 있는 구현, 즉 `evaluate(Decision) -> DecisionResult` 인터페이스를 만족하는 구현이라면 evaluator로 사용할 수 있습니다.

---

커스텀 백앤드는 이 evaluator interface만 구현하면 됩니다.

```python
class MyEvaluator:
    def evaluate(self, decision: Decision) -> DecisionResult:
        ...
```

## 예제

* 텍스트: [`example/llama_evaluator.py`](example/llama_evaluator.py)
* 이미지: [`example/llama_vision_evaluator.py`](example/llama_vision_evaluator.py)

미디어 소스는 `Image`, `Audio`, `Video`로 감쌉니다. 튜플에는 텍스트와 미디어를 함께 넣을 수 있습니다.

```python
from makoto_decision import Choices, Decision, Image

decision = Decision(
    question="피사체에 초점이 맞았나요?",
    context=(Image("frame.png"), "원본 영상의 한 프레임입니다."),
    choices=Choices.yes_or_no(),
)
```

`HashEvaluator`에는 질문이 필요합니다. 질문의 SHA-256 해시에서 앞 8바이트를 정수로 읽고
선택지 수로 나눈 나머지를 사용해 선택지를 고릅니다. `Decision.context`는 사용하지 않으며,
선택된 항목에는 `1.0`, 나머지에는 `0.0`을 반환합니다. [실행 예제](example/hash_evaluator.py)를 참고하세요.

`RuleEvaluator`는 각 규칙이 `Choice.value`를 키로 반환한 숫자 점수를 합산합니다.
[실행 예제](example/rule_evaluator.py)도 참고하세요.

```python
from makoto_decision import Choice, Decision
from makoto_decision.evaluators import RuleEvaluator

rule_result = RuleEvaluator([lambda decision: {"yes": 1.0}]).evaluate(
    Decision(choices=Choices.yes_or_no(), context="")
)
```

`Decision`에는 하나 이상의 선택지가 필요합니다. `Choice.value`는 비어 있지 않고 서로 달라야 하며,
각 선택지의 대상 문자열(`target`)도 비어 있지 않아야 합니다.
`LlamaCppEvaluator`에서는 대상 문자열이 서로 다르고 각각 정확히 하나의 토큰으로 분리되어야 합니다.
`DecisionResult.scores`에는 평가기 점수를 그대로 보존하고, `DecisionResult.probabilities`에는
사용 가능한 선택지에 대한 softmax 확률을 제공합니다.
문맥에는 텍스트, `Image`, `Audio`, `Video` 또는 텍스트와 미디어를 섞은 튜플을 사용할 수 있습니다.
미디어 래퍼는 경로, 문자열 소스, `byte[]`를 받습니다.

## 개발

프로젝트 설정과 개발 방법은 [설치](docs/installation.md), [개발](docs/development.md),
[배포](docs/publishing.md) 문서를 참고하세요.

<!-- This document follows common-doc-guidelines.md.
See github.com/jlevy/practical-prose and review guidelines before editing.
-->
