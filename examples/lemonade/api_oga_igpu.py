"""
This example demonstrates how to use the LEAP API to load a model for
inference on integrated GPUs (iGPUs) via OnnxRuntime-Genai (OGA)
using the oga-igpu recipe, and then use it to generate the response to a prompt.

Make sure you have set up your OGA device in your Python environment.
See for details:
https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/getting_started.md#install-onnxruntime-genai
"""

from lemonade import leap

model, tokenizer = leap.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", recipe="oga-igpu")

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)

print(tokenizer.decode(response[0]))
