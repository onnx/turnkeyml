# Welcome to ONNX TurnkeyML

[![Lemonade tests](https://github.com/onnx/turnkeyml/actions/workflows/test_lemonade.yml/badge.svg)](https://github.com/onnx/turnkeyml/tree/main/test "Check out our tests")
[![Turnkey tests](https://github.com/onnx/turnkeyml/actions/workflows/test_turnkey.yml/badge.svg)](https://github.com/onnx/turnkeyml/tree/main/test "Check out our tests")
[![OS - Windows | Linux](https://img.shields.io/badge/OS-windows%20%7C%20linux-blue)](https://github.com/onnx/turnkeyml/blob/main/docs/install.md "Check out our instructions")
[![Made with Python](https://img.shields.io/badge/Python-3.8,3.10-blue?logo=python&logoColor=white)](https://github.com/onnx/turnkeyml/blob/main/docs/install.md "Check out our instructions")

We are on a mission to make it easy to use the most important tools in the ONNX ecosystem. TurnkeyML accomplishes this by providing a full SDK for LLMs with the Lemonade SDK, as well as a no-code CLIs for general ONNX workflows with `turnkey`.

## 🍋 Lemonade SDK: Quickly serve, benchmark and deploy LLMs

# The Lemonade SDK Project has moved to: https://github.com/lemonade-sdk/lemonade

## The new PyPI package name is `lemonade-sdk`

## The new Lemonade_Server_Installer.exe is at: https://github.com/lemonade-sdk/lemonade/releases

## Please migrate to the new repository and package as soon as possible.

For example:
```
    pip uninstall turnkeyml
    pip install lemonade-sdk[YOUR_EXTRAS]
    e.g., pip install lemonade-sdk[llm-oga-hybrid]
```

## Thank you for using Lemonade SDK!

## 🔑 Turnkey: A Complementary Tool for ONNX Workflows

While Lemonade focuses on LLMs, [Turnkey](https://github.com/onnx/turnkeyml/blob/main/docs/turnkey/README.md) is a no-code CLI designed for general ONNX workflows, such as exporting and optimizing CNNs and Transformers.

To see the list of supported tools, using the following command:

```bash
turnkey -h
```

<div align="center">
  <img src="https://download.amd.com/images/tkml_640x480_1.gif" alt="Turnkey Demo" title="Turnkey CLI">
</div>

### [Click here to get started with `turnkey`.](https://github.com/onnx/turnkeyml/blob/main/docs/turnkey/README.md)

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/onnx/turnkeyml/blob/main/docs/contribute.md).

## Maintainers

This project is sponsored by the [ONNX Model Zoo](https://github.com/onnx/models) special interest group (SIG). It is maintained by @danielholanda @jeremyfowers @ramkrishna @vgodsoe in equal measure. You can reach us by filing an [issue](https://github.com/onnx/turnkeyml/issues) or emailing `turnkeyml at amd dot com`.

## License

This project is licensed under the [Apache 2.0 License](https://github.com/onnx/turnkeyml/blob/main/LICENSE).

## Attribution

TurnkeyML used code from other open source projects as a starting point (see [NOTICE.md](NOTICE.md)). Thank you Philip Colangelo, Derek Elkins, Jeremy Fowers, Dan Gard, Victoria Godsoe, Mark Heaps, Daniel Holanda, Brian Kurtz, Mariah Larwood, Philip Lassen, Andrew Ling, Adrian Macias, Gary Malik, Sarah Massengill, Ashwin Murthy, Hatice Ozen, Tim Sears, Sean Settle, Krishna Sivakumar, Aviv Weinstein, Xueli Xao, Bill Xing, and Lev Zlotnik for your contributions to that work.

