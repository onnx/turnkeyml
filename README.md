# ONNX TurnkeyML

[![Turnkey tests](https://github.com/onnx/turnkeyml/actions/workflows/test_turnkey.yml/badge.svg)](https://github.com/onnx/turnkeyml/tree/main/test "Check out our tests")
[![OS - Windows | Linux](https://img.shields.io/badge/OS-windows%20%7C%20linux-blue)](https://github.com/onnx/turnkeyml/blob/main/docs/install.md "Check out our instructions")
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](docs/install.md)

TurnkeyML provides a no-code CLI for ONNX workflows. Use `turnkey` to discover
PyTorch models, export them to ONNX, and run tools in configurable sequences.

## Installation

The current source requires **Python 3.11** and is tested on Windows and Linux.
To install it from a source checkout:

```bash
git clone https://github.com/onnx/turnkeyml.git
cd turnkeyml
python -m pip install -e .
```

See the [installation guide](docs/install.md) for environment setup and optional
model and Slurm dependencies.

## Quick start

From the repository root, discover and export the included self-test model:

```bash
turnkey -i models/selftest/linear.py discover export-pytorch
```

List available tools and get help for a specific tool:

```bash
turnkey -h
turnkey export-pytorch -h
```

<div align="center">
  <img src="https://download.amd.com/images/tkml_640x480_1.gif" alt="Turnkey Demo" title="Turnkey CLI">
</div>

## Learn more

- [Turnkey guide](docs/turnkey/README.md): workflows and concepts.
- [Tools user guide](docs/turnkey/tools_user_guide.md): CLI and API reference.
- [Examples](examples/turnkey/README.md) and [models](models/readme.md).
- [Plugins](plugins/README.md): extend Turnkey with custom tools and devices.

## Lemonade SDK

Lemonade has moved to [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade)
and is no longer included in this repository. Install it separately with
`python -m pip install lemonade-sdk`. See the new repository for its
documentation and [releases](https://github.com/lemonade-sdk/lemonade/releases).

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/onnx/turnkeyml/blob/main/docs/contribute.md).

## Maintainers

This project is sponsored by the [ONNX Model Zoo](https://github.com/onnx/models) special interest group (SIG). It is maintained by @danielholanda @jeremyfowers @ramkrishna @vgodsoe in equal measure. You can reach us by filing an [issue](https://github.com/onnx/turnkeyml/issues) or emailing `turnkeyml at amd dot com`.

## License

This project is licensed under the [Apache 2.0 License](https://github.com/onnx/turnkeyml/blob/main/LICENSE).

## Attribution

TurnkeyML used code from other open source projects as a starting point (see [NOTICE.md](NOTICE.md)). Thank you Philip Colangelo, Derek Elkins, Jeremy Fowers, Dan Gard, Victoria Godsoe, Mark Heaps, Daniel Holanda, Brian Kurtz, Mariah Larwood, Philip Lassen, Andrew Ling, Adrian Macias, Gary Malik, Sarah Massengill, Ashwin Murthy, Hatice Ozen, Tim Sears, Sean Settle, Krishna Sivakumar, Aviv Weinstein, Xueli Xao, Bill Xing, and Lev Zlotnik for your contributions to that work.

