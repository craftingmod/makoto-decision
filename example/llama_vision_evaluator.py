"""Classify example/media/cat.jpg with a Gemma4 vision model.

Run with a llama-cpp-python build that provides create_chat_prefill():
python example/llama_vision_evaluator.py --model model.gguf --mmproj mmproj.gguf
"""

import argparse
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_multimodal import Gemma4ChatHandler
from makoto_decision import Choices, Decision, Image
from makoto_decision.evaluators import LlamaCppEvaluator

EXAMPLE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Check whether an image contains a cat.")
parser.add_argument("-m", "--model", type=Path, required=True, help="LLaVA 1.5 model GGUF")
parser.add_argument("-mm", "--mmproj", type=Path, required=True, help="Matching mmproj GGUF")
parser.add_argument(
    "-i", "--image", type=Path, default=EXAMPLE_DIR / "media" / "cat.jpg", help="Image to classify"
)
args = parser.parse_args()

llama = Llama(
    model_path=str(args.model),
    chat_handler=Gemma4ChatHandler(
        mmproj_path=args.mmproj,
        enable_thinking=False,
    ),
    n_ctx=4096,
)
try:
    decision = Decision(
        question="Does this image show a cat?",
        context=Image(args.image),
        choices=Choices.letters("cat", "not a cat"),
    )
    result = LlamaCppEvaluator(llama).evaluate(decision)
    print(result.selected)
    print(dict(result.probabilities))  # softmax
finally:
    llama.close()
