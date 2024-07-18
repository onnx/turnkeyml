# Welcome to ONNX TurnkeyML

[![Turnkey tests](https://github.com/onnx/turnkeyml/actions/workflows/test_turnkey.yml/badge.svg)](https://github.com/onnx/turnkeyml/tree/main/test "Check out our tests")
[![OS - Windows | Linux](https://img.shields.io/badge/OS-windows%20%7C%20linux-blue)](https://github.com/onnx/turnkeyml/blob/main/docs/install.md "Check out our instructions")
[![Made with Python](https://img.shields.io/badge/Python-3.8,3.10-blue?logo=python&logoColor=white)](https://github.com/onnx/turnkeyml/blob/main/docs/install.md "Check out our instructions")

> **Notice**: TurnkeyML has just gone through a major refresh of its interfaces and source code.

We are on a mission to make it easy to use the most important tools in the ONNX ecosystem. TurnkeyML accomplishes this by providing a no-code CLI, `turnkey`, as well as a low-code API, that provide seamless integration of these tools.

## Getting Started

Getting started is as simple as `pip install turnkeyml` for most users.

## Demo

Here's `turnkey` in action: BERT-Base is exported from PyTorch to ONNX using `torch.onnx.export`, optimized for inference with `onnxruntime`, and converted to float16 with `onnxmltools`:

![Basic Demo Video](img/basic_demo.gif)

Breaking it down:

1. `turnkey -i bert.py` feeds [`bert.py`](https://github.com/onnx/turnkeyml/blob/main/models/transformers/bert.py), a basic PyTorch script that instantiates BERT, into the tool sequence, starting with...
1. `discover` is a tool that finds the PyTorch model in a script and passes it to the next tool, which is...
1. `export-pytorch`, which takes a PyTorch model and converts it to an ONNX model, then passes it to...
1. `optimize-ort`, which uses `onnxruntime` to optimize the model's compute graph, then passes it to...
1. `convert-fp16`, which uses `onnxmltools` to convert the ONNX file into fp16.
1. Finally, the result is printed, and we can see that the requested `.onnx` files have been produced.

All without writing a single line of code or learning how to use any of the underlying ecosystem modules 🚀

## How It Works

The `turnkey` CLI provides a set of `Tools` that users can invoke in a `Sequence`. The first `Tool` takes the input (`-i`), performs some action, and passes its state to the next `Tool` in the `Sequence`.

You can read the `Sequence` out like a sentence. For example, the demo command above was:

```
> turnkey -i bert.py discover export-pytorch optimize-ort convert-fp16
```

Which you can read like:

> Use `turnkey` on `bert.py` to `discover` the model, `export` the `pytorch` to ONNX, `optimize` the ONNX with `ort`, and `convert` the ONNX to `fp16`.

You can configure each `Tool` by passing it arguments. For example, `export-pytorch --opset 18` would set the opset of the resulting ONNX model to 18.

A full command with an arguments could look like:

```
> turnkey -i bert.py discover export-pytorch --opset 18 optimize-ort conver-fp16
```

Learn more by running `turnkey -h`. To learn about a specific tool, run `turnkey <tool> -h`, for example `turnkey export-pytorch -h`.

## Learn More

- [Installation guide](https://github.com/onnx/turnkeyml/blob/main/docs/install.md)
- [Tutorials](https://github.com/onnx/turnkeyml/tree/main/examples/cli)
- [User guide](https://github.com/onnx/turnkeyml/blob/main/docs/tools_user_guide.md)
- [Code organization guide](https://github.com/onnx/turnkeyml/blob/main/docs/code.md)

## Mass Evaluation

`turnkey` is used in multiple projects where many hundreds of models are being evaluated. For example, the [ONNX Model Zoo](https://github.com/onnx/models) was created using `turnkey`.

We provide several helpful tools to facilitate this kind of mass-evaluation.

### Wildcard Input

`turnkey` will iterate over multiple inputs if you pass it a wildcard input. 

For example, to export ~1000 built-in models to ONNX:

```
> turnkey models/*/*.py discover export-pytorch
```

### Results Cache

All build results, such as `.onnx` files, are collected into a cache directory, which you can learn about with `turnkey cache -h`.

### Generating Reports

`turnkey` collects statistics about each model and build into the corresponding build directory in the cache. Use `turnkey report -h` to see how those statistics can be exported into a CSV file.


## Extensibility

### Models

[![transformers](https://img.shields.io/github/directory-file-count/onnx/turnkeyml/models/transformers?label=transformers)](https://github.com/onnx/turnkeyml/tree/main/models/transformers "Transformer models")
[![graph_convolutions](https://img.shields.io/github/directory-file-count/onnx/turnkeyml/models/graph_convolutions?label=graph_convolutions)](https://github.com/onnx/turnkeyml/tree/main/models/graph_convolutions "Graph Convolution models")
[![torch_hub](https://img.shields.io/github/directory-file-count/onnx/turnkeyml/models/torch_hub?label=torch_hub)](https://github.com/onnx/turnkeyml/tree/main/models/torch_hub "Models from Torch Hub")
[![torchvision](https://img.shields.io/github/directory-file-count/onnx/turnkeyml/models/torchvision?label=torchvision)](https://github.com/onnx/turnkeyml/tree/main/models/torchvision "Models from Torch Vision")
[![timm](https://img.shields.io/github/directory-file-count/onnx/turnkeyml/models/timm?label=timm)](https://github.com/onnx/turnkeyml/tree/main/models/timm "Pytorch Image Models")

This repository is home to a diverse corpus of hundreds of models, which are meant to be a convenient input to `turnkey -i <model>.py discover`. We are actively working on increasing the number of models in our model library. You can see the set of models in each category by clicking on the corresponding badge.

Evaluating a new model is as simple as taking a Python script that instantiates and invokes a PyTorch `torch.nn.module` and call `turnkey` on it. Read about model contributions [here](https://github.com/onnx/turnkeyml/blob/main/docs/contribute.md#contributing-a-model).

### Plugins

The build tool has built-in support for a variety of interoperable `Tools`. If you need more, the TurnkeyML plugin API lets you add your own installable  tools with any functionality you like:

```
> pip install -e my_custom_plugin
> turnkey -i my_model.py discover export-pytorch my-custom-tool --my-args
```

All of the built-in `Tools` are implemented against the plugin API. Check out the [example plugins](https://github.com/onnx/turnkeyml/tree/main/examples/cli/plugins) and the [plugin API guide](https://github.com/onnx/turnkeyml/blob/main/docs/contribute.md#contributing-a-plugin) to learn more about creating an installable plugin.

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/onnx/turnkeyml/blob/main/docs/contribute.md).

## Maintainers

This project is sponsored by the [ONNX Model Zoo](https://github.com/onnx/models) special interest group (SIG). It is maintained by @danielholanda @jeremyfowers @ramkrishna @vgodsoe in equal measure. You can reach us by filing an [issue](https://github.com/onnx/turnkeyml/issues).

## License

This project is licensed under the [Apache 2.0 License](https://github.com/onnx/turnkeyml/blob/main/LICENSE).

## Attribution

TurnkeyML used code from other open source projects as a starting point (see [NOTICE.md](NOTICE.md)). Thank you Philip Colangelo, Derek Elkins, Jeremy Fowers, Dan Gard, Victoria Godsoe, Mark Heaps, Daniel Holanda, Brian Kurtz, Mariah Larwood, Philip Lassen, Andrew Ling, Adrian Macias, Gary Malik, Sarah Massengill, Ashwin Murthy, Hatice Ozen, Tim Sears, Sean Settle, Krishna Sivakumar, Aviv Weinstein, Xueli Xao, Bill Xing, and Lev Zlotnik for your contributions to that work.
