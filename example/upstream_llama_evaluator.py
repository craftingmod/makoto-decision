"""Using custom evaluator with decision example."""
from llama_cpp import Llama
import argparse

from makoto_decision import Choices, Decision
from upstream_llama_backend import UpstreamLlamaCppEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()

llama = Llama(model_path=args.model)

evaluator = UpstreamLlamaCppEvaluator(llama)

decision = Decision(
    question="Is the following statement true?",
    context="2 + 2 = 4",
    choices=Choices.yes_or_no(),
)

result = evaluator.evaluate(decision)

print(result.selected)
print(result.scores)
llama.close()