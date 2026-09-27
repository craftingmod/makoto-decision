from llama_cpp import Llama
import argparse

from makoto_decision import Choices, Decision
from makoto_decision.evaluators import LlamaCppEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()

# Use a llama-cpp-python build that provides create_chat_prefill().
llama = Llama(
    model_path=args.model,
    n_ctx=4096,
)
decision = Decision(
    question="Is the following statement true?",
    context="2 + 2 = 4",
    choices=Choices.yes_or_no(),
)
result = LlamaCppEvaluator(llama).evaluate(decision)

print(result.selected)
print(dict(result.scores))  # raw next-token logits

llama.close()
