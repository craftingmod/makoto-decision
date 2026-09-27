# Makoto-decision

![대코토](./docs/media/daekoto.jpg)

> 키히히힛. 좋은 선택이야.

### 하누마 마코토처럼 선택하기: 맞든 틀리든, 쿨하게.

`makoto-decision`은 미리 정한 선택지 중 하나를 고르는 가벼운 Python 라이브러리입니다.

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
 # 확률이 아닌 다음 토큰의 logic이다요
print(result.scores) # {"yes": 1.0, "no": 0.0}

```

문맥, 선택적인 질문, `Choices.yes_or_no()` 같은 선택지를 정의합니다.
평가기(evaluator)는 선택된 항목과 점수를 반환합니다.

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

```python
class MyEvaluator:
    def evaluate(self, decision: Decision) -> DecisionResult:
        ...
```

`evaluate(Decision) -> DecisionResult`를 구현한 아무 객체나 평가기로 사용할 수 있습니다.

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
평가기들은 점수를 확률로 정규화하지 않고 `DecisionResult(selected, scores)`를 반환합니다.
문맥에는 텍스트, `Image`, `Audio`, `Video` 또는 텍스트와 미디어를 섞은 튜플을 사용할 수 있습니다.
미디어 래퍼는 경로, 문자열 소스, 바이트를 받습니다.

## 개발

프로젝트 설정과 개발 방법은 [설치](docs/installation.md), [개발](docs/development.md),
[배포](docs/publishing.md) 문서를 참고하세요.

<!-- This document follows common-doc-guidelines.md.
See github.com/jlevy/practical-prose and review guidelines before editing.
-->
