
# 🍋 Lemonade Server Models
 
This document provides the models we recommend for use with Lemonade Server. Click on any model to learn more details about it, such as the [Lemonade Recipe](https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/lemonade_api.md) used to load the model.

## Naming Convention
The format of each Lemonade name is a combination of the name in the base checkpoint and the backend where the model will run. So, if the base checkpoint is `meta-llama/Llama-3.2-1B-Instruct`, and it has been optimized to run on Hybrid, the resulting name is Llama-3.2-3B-Instruct-Hybrid.

## Installing Additional Models

Once you've installed Lemonade Server, you can install any model on this list using the `lemonade-server pull` command. 

Example:

```bash
lemonade-server pull Qwen2.5-0.5B-Instruct-CPU
```

> Note: `lemonade-server` is a utility that is added to your PATH when you install Lemonade Server with the GUI installer.
> If you are using Lemonade Server from a Python environment, use the `lemonade-server-dev pull` command instead.

## Supported Models

### Hybrid

<details>
<summary>Llama-3.2-1B-Instruct-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid](https://huggingface.co/amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>

<details>
<summary>Llama-3.2-3B-Instruct-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid](https://huggingface.co/amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>

<details>
<summary>Phi-3-Mini-Instruct-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-fp16-onnx-hybrid](https://huggingface.co/amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-fp16-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>

<details>
<summary>Qwen-1.5-7B-Chat-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-fp16-onnx-hybrid](https://huggingface.co/amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-fp16-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>

<details>
<summary>DeepSeek-R1-Distill-Llama-8B-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-hybrid](https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | True |

</details>

<details>
<summary>DeepSeek-R1-Distill-Qwen-7B-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/DeepSeek-R1-Distill-Qwen-7B-awq-asym-uint4-g128-lmhead-onnx-hybrid](https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-7B-awq-asym-uint4-g128-lmhead-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | True |

</details>

<details>
<summary>Mistral-7B-v0.3-Instruct-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-fp16-onnx-hybrid](https://huggingface.co/amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-fp16-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>

<details>
<summary>Llama-3.1-8B-Instruct-Hybrid</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Llama-3.1-8B-Instruct-awq-asym-uint4-g128-lmhead-onnx-hybrid](https://huggingface.co/amd/Llama-3.1-8B-Instruct-awq-asym-uint4-g128-lmhead-onnx-hybrid) |
| Recipe | oga-hybrid |
| Reasoning | False |

</details>


### CPU

<details>
<summary>Qwen2.5-0.5B-Instruct-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx](https://huggingface.co/amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx) |
| Recipe | oga-cpu |
| Reasoning | False |

</details>

<details>
<summary>Llama-3.2-1B-Instruct-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Llama-3.2-1B-Instruct-awq-uint4-float16-cpu-onnx](https://huggingface.co/amd/Llama-3.2-1B-Instruct-awq-uint4-float16-cpu-onnx) |
| Recipe | oga-cpu |
| Reasoning | False |

</details>

<details>
<summary>Llama-3.2-3B-Instruct-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Llama-3.2-3B-Instruct-awq-uint4-float16-cpu-onnx](https://huggingface.co/amd/Llama-3.2-3B-Instruct-awq-uint4-float16-cpu-onnx) |
| Recipe | oga-cpu |
| Reasoning | False |

</details>

<details>
<summary>Phi-3-Mini-Instruct-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Phi-3-mini-4k-instruct_int4_float16_onnx_cpu](https://huggingface.co/amd/Phi-3-mini-4k-instruct_int4_float16_onnx_cpu) |
| Recipe | oga-cpu |
| Reasoning | False |

</details>

<details>
<summary>Qwen-1.5-7B-Chat-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/Qwen1.5-7B-Chat_uint4_asym_g128_float16_onnx_cpu](https://huggingface.co/amd/Qwen1.5-7B-Chat_uint4_asym_g128_float16_onnx_cpu) |
| Recipe | oga-cpu |
| Reasoning | False |

</details>

<details>
<summary>DeepSeek-R1-Distill-Llama-8B-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-cpu](https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-cpu) |
| Recipe | oga-cpu |
| Reasoning | True |

</details>

<details>
<summary>DeepSeek-R1-Distill-Qwen-7B-CPU</summary>

| Key | Value |
| --- | ----- |
| Checkpoint | [amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-cpu](https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-cpu) |
| Recipe | oga-cpu |
| Reasoning | True |

</details>

