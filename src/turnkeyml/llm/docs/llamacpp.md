# LLAMA.CPP

Run transformer models using a Llama.cpp binary and checkpoint. This model can then be used with chatting or benchmarks such as MMLU.

## Prerequisites

This flow has been verified with a generic Llama.cpp model.

## Installation

### 1. Set up Environment

Build or obtain the Llama.cpp model and desired checkpoint

### 2. Install genai

Following the genai [README](../README.md), install lemonade by running `pip install -e genai`

## Usage

The Llama.cpp tool currently supports the following parameters

| Parameter    | Definition                                                                  | Default |
| ---------    | ----------------------------------------------------                        | ------- |
| executable   | Path to the Llama.cpp-generated application binary                          | None    |
| model-binary | Model checkpoint (do not use if --input is passed to lemonade)              | None    |
| threads      | Number of threads to use for computation                                    | 1       |
| context-size | Maximum context length                                                      | 512     |
| temp         | Temperature to use for inference (leave out to use the application default) | None    |

### Example

```bash
lemonade --input checkpoint.gguf load-llama-cpp --executable /app/main accuracy-mmlu --ntrain 5
```
