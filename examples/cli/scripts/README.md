# CLI PyTorch Examples

You can try out `turnkey` with PyTorch models, via the `discover` tool, like so:

```bash
cd turnkeyml/examples/cli/scripts
turnkey -i hello_world.py discover export-pytorch
```
This will discover the model within `hello_world.py` and export it to ONNX.

See the docstring in each script for more information.
- `hello_world.py`: example with a single-layer PyTorch model.
- `max_depth.py`: example with a multi-layer PyTorch model, to try out the `discover --max-depth DEPTH` option.
- `multiple_invocations.py` and `two_models.py`: examples where more than one model invocation will be discovered.
