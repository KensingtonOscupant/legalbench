# This script exemplifies how to call an LLM on a single LegalBench task.

import weave
from weave import Model
import asyncio

TASK               = "abercrombie"
WEAVE_TEAM         = "fuels"
WEAVE_PROJECT      = "legalbench-leaderboard"

client = weave.init(f"{WEAVE_TEAM}/{WEAVE_PROJECT}")

dataset = weave.ref(f"{TASK}_test").get()

"""This script det"""

class MyModel5(Model):
    prompt: str

    @weave.op()
    def predict(self, text: str):
        prompt = self.prompt

        prompt_for_model = prompt.replace("{{text}}", text)
        
        return {'generation': prompt_for_model}

# Load base prompt
with open(f"tasks/abercrombie/base_prompt.txt") as in_file:
    prompt_template = in_file.read()

model = MyModel5(prompt=prompt_template)

eval = weave.ref(f"{TASK}_evaluation").get()

asyncio.run(eval.evaluate(model))
