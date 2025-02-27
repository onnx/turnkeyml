"""
This example demonstrates how to use the LEAP API to load a model for
inference on CPU via OnnxRuntime-GenAI using the oga-cpu recipe, and then
use a thread to generate a streaming the response to a prompt.

Note: this approach only works with recipes that support lemonade's OrtGenaiStreamer,
i.e., OGA-based recipes such as oga-cpu, oga-igpu, oga-npu, and oga-hybrid.

Make sure you have set up your OGA device in your Python environment.
See for details:
https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/getting_started.md#install-onnxruntime-genai
"""

from threading import Thread
from lemonade import leap
from lemonade.tools.ort_genai.oga import OrtGenaiStreamer

model, tokenizer = leap.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", recipe="oga-cpu")

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids

streamer = OrtGenaiStreamer(tokenizer)
generation_kwargs = {
    "input_ids": input_ids,
    "streamer": streamer,
    "max_new_tokens": 30,
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Generate the response using streaming
for new_text in streamer:
    print(new_text)

thread.join()
