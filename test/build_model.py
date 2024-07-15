import os
import argparse
import unittest
import torch
import onnx
import numpy as np
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import save_model
from onnxmltools.utils import load_model
from turnkeyml import build_model
import turnkeyml.build.export as export
import turnkeyml.build.stage as stage
import turnkeyml.common.filesystem as fs
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build


class SmallPytorchModel(torch.nn.Module):
    def __init__(self):
        super(SmallPytorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        output = self.fc(x)
        return output


class AnotherSimplePytorchModel(torch.nn.Module):
    def __init__(self):
        super(AnotherSimplePytorchModel, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        return output


base_dir = os.path.dirname(os.path.abspath(__file__))
cache_location = os.path.join(base_dir, "generated", "build_model_cache")

# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
tiny_pytorch_model = AnotherSimplePytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)


def basic_pytorch_sequence():
    return stage.Sequence(stages={export.ExportPytorchModel(): []})


def basic_onnx_sequence(input: str):
    return stage.Sequence(stages={export.LoadOnnx(): ["--input", input]})


# Run build_model() and get results
def full_compilation_pytorch_model():
    build_name = "full_compilation_pytorch_model"
    state = build_model(
        sequence=basic_pytorch_sequence(),
        model=pytorch_model,
        inputs=inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.FunctionStatus.SUCCESSFUL


def full_compilation_onnx_model():
    build_name = "full_compilation_onnx_model"
    onnx_file = "small_onnx_model.onnx"
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        onnx_file,
        opset_version=build.DEFAULT_ONNX_OPSET,
        input_names=["input"],
        output_names=["output"],
    )
    state = build_model(
        sequence=basic_onnx_sequence(input=onnx_file),
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.FunctionStatus.SUCCESSFUL


def scriptmodule_functional_check():
    build_name = "scriptmodule_functional_check"
    x = torch.rand(10)
    forward_input = x
    input_dict = {"forward": forward_input}
    pytorch_module = torch.jit.trace_module(pytorch_model, input_dict)
    state = build_model(
        sequence=basic_pytorch_sequence(),
        model=pytorch_module,
        inputs=inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.FunctionStatus.SUCCESSFUL


def custom_stage():
    build_name = "custom_stage"

    class MyCustomStage(stage.Stage):
        unique_name = "funny-fp16"

        def __init__(self, funny_saying):
            super().__init__(
                monitor_message="Funny FP16 conversion",
            )

            self.funny_saying = funny_saying

        @staticmethod
        def parser(add_help: bool = True) -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(
                description="Parser for a test stage",
                add_help=add_help,
            )

            return parser

        def fire(self, state):
            input_onnx = state.results
            output_onnx = os.path.join(export.onnx_dir(state), "custom.onnx")
            fp32_model = load_model(input_onnx)
            fp16_model = convert_float_to_float16(fp32_model)
            save_model(fp16_model, output_onnx)

            print(f"funny message: {self.funny_saying}")

            state.results = output_onnx

            return state

    my_custom_stage = MyCustomStage(
        funny_saying="Is a fail whale a fail at all if it makes you smile?"
    )
    my_sequence = stage.Sequence(
        stages={
            export.ExportPytorchModel(): [],
            export.OptimizeOnnxModel(): [],
            my_custom_stage: [],
        },
    )

    state = build_model(
        sequence=my_sequence,
        model=pytorch_model,
        inputs=inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )

    return state.build_status == build.FunctionStatus.SUCCESSFUL


class FullyCustomStage(stage.Stage):
    unique_name = "fully-custom"

    def __init__(self, saying, name):
        super().__init__(
            monitor_message=f"Running {name}",
        )

        self.saying = saying

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Parser for a test stage",
            add_help=add_help,
        )

        return parser

    def fire(self, state):
        print(self.saying)

        state.results = "great stuff"

        return state


class FullyCustomStage1(FullyCustomStage):
    unique_name = "fully-custom-1"


class FullyCustomStage2(FullyCustomStage):
    unique_name = "fully-custom-2"


def custom_sequence():
    build_name = "custom_sequence"
    stage_1_name = "fully-custom"
    stage_2_name = "fully-custom-1"
    stage_3_name = "fully-custom-2"
    stage_1_msg = "Developer Velocity is"
    stage_2_msg = "Innovating"
    stage_3_msg = "Faster than ever"

    stage_1 = FullyCustomStage(stage_1_msg, stage_1_name)
    stage_2 = FullyCustomStage1(stage_2_msg, stage_2_name)
    stage_3 = FullyCustomStage2(stage_3_msg, stage_3_name)

    my_sequence = stage.Sequence(stages={stage_1: [], stage_2: [], stage_3: []})

    build_model(
        sequence=my_sequence,
        model=pytorch_model,
        inputs=inputs,
        build_name=build_name,
        monitor=False,
        rebuild="always",
        cache_dir=cache_location,
    )

    log_1_path = os.path.join(cache_location, build_name, f"log_{stage_1_name}.txt")
    log_2_path = os.path.join(cache_location, build_name, f"log_{stage_2_name}.txt")
    log_3_path = os.path.join(cache_location, build_name, f"log_{stage_3_name}.txt")

    with open(log_1_path, "r", encoding="utf8") as f:
        log_1 = f.readlines()[1]

    with open(log_2_path, "r", encoding="utf8") as f:
        log_2 = f.readlines()[1]

    with open(log_3_path, "r", encoding="utf8") as f:
        log_3 = f.readlines()[1]

    return stage_1_msg in log_1 and stage_2_msg in log_2 and stage_3_msg in log_3


def rebuild_always():
    """
    This function checks to see if the build_name.yaml file has been modified.
    If rebuild="always" the build_name_state.yaml file will have been modified along with
        the rest of the files in model/build_name due to a forced rebuild.
    If rebuild="never" the build_name_state.yaml file should *not* have been modified and
        the rest of the files in model/build_name will remain untouched and the
        model will be loaded from cache.
    To pass this test:
        between build 1 and build 2 the build_name_state.yaml file will be modified and
            therefor have different file modification timestamps
        between build 2 and build 3 the build_name_state.yaml file will *not* be modified
            resulting in identical modification timestamps.
    """
    build_name = "rebuild"
    build_timestamps = {}
    build_purpose_to_rebuild_setting = {
        "initial": "always",
        "rebuild": "always",
        "load": "never",
    }

    # Build Initial model, rebuild, and load from cache
    for build_purpose, rebuild_setting in build_purpose_to_rebuild_setting.items():
        build_model(
            sequence=basic_pytorch_sequence(),
            model=pytorch_model,
            inputs=inputs,
            build_name=build_name,
            rebuild=rebuild_setting,
            monitor=False,
            cache_dir=cache_location,
        )

        yaml_file_path = build.state_file(cache_location, build_name)

        # Read the the file modification timestamp
        if os.path.isfile(yaml_file_path):
            build_timestamps[build_purpose] = os.path.getmtime(yaml_file_path)
        else:
            msg = f"""
            The rebuild_always test attempted to load a state.yaml file
            at {yaml_file_path} but couldn't find one.
            """
            raise ValueError(msg)

    # Did the second build Rebuild?
    if build_timestamps["initial"] != build_timestamps["rebuild"]:
        rebuild = True
    else:
        rebuild = False

    # Was the third build skipped and the model loaded from cache?
    if build_timestamps["rebuild"] == build_timestamps["load"]:
        load = True
    else:
        load = False

    return rebuild and load


def rebuild_if_needed():
    """
    This function checks to see if the build_name.yaml file has been modified.
    If rebuild="always" the build_name_state.yaml file will have been modified along with
        the rest of the files in model/build_name due to a forced rebuild.
    If rebuild="if_needed" the build_name_state.yaml file should *not* have been modified and
        the rest of the files in model/build_name will remain untouched and the
        model will be loaded from cache.
    To pass this test:
        between build 1 and build 2 the build_name_state.yaml file will *not* be modified
            resulting in identical modification timestamps.
    We also toss in a state.save() call to make sure that doesn't break the cache.
    """
    build_name = "rebuild"
    build_timestamps = {}
    build_purpose_to_rebuild_setting = {
        "initial": "always",
        "load": "if_needed",
    }

    # Build Initial model, rebuild, and load from cache
    for build_purpose, rebuild_setting in build_purpose_to_rebuild_setting.items():
        state = build_model(
            sequence=basic_pytorch_sequence(),
            model=pytorch_model,
            inputs=inputs,
            build_name=build_name,
            rebuild=rebuild_setting,
            monitor=False,
            cache_dir=cache_location,
        )

        if build_purpose == "initial":
            state.save()

        yaml_file_path = build.state_file(cache_location, build_name)

        # Read the the file modification timestamp
        if os.path.isfile(yaml_file_path):
            build_timestamps[build_purpose] = os.path.getmtime(yaml_file_path)
        else:
            msg = f"""
            The rebuild_always test attempted to load a state.yaml file
            at {yaml_file_path} but couldn't find one.
            """
            raise ValueError(msg)

    # Was the third build skipped and the model loaded from cache?
    if build_timestamps["initial"] == build_timestamps["load"]:
        load = True
    else:
        load = False

    return load


def illegal_onnx_opset():
    build_name = "illegal_onnx_opset"
    onnx_file = "illegal_onnx_opset.onnx"
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        onnx_file,
        opset_version=(build.MINIMUM_ONNX_OPSET - 1),
        input_names=["input"],
        output_names=["output"],
    )
    build_model(
        sequence=basic_onnx_sequence(input=onnx_file),
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        fs.rmdir(cache_location)

        return super().setUp()

    def test_000_rebuild_always(self):
        assert rebuild_always()

    def test_001_rebuild_if_needed(self):
        assert rebuild_if_needed()

    def test_002_full_compilation_pytorch_model(self):
        assert full_compilation_pytorch_model()

    def test_005_full_compilation_onnx_model(self):
        assert full_compilation_onnx_model()

    def test_009_custom_stage(self):
        assert custom_stage()

    def test_011_custom_sequence(self):
        assert custom_sequence()

    def test_012_illegal_onnx_opset(self):
        self.assertRaises(exp.StageError, illegal_onnx_opset)
        if os.path.exists("illegal_onnx_opset.onnx"):
            os.remove("illegal_onnx_opset.onnx")

    def test_013_set_onnx_opset(self):
        build_name = "full_compilation_pytorch_model"

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        sequence = stage.Sequence(
            stages={
                export.ExportPytorchModel(): ["--opset", str(user_opset)],
                export.OptimizeOnnxModel(): [],
            }
        )

        state = build_model(
            sequence=sequence,
            model=pytorch_model,
            inputs=inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
        )

        assert state.build_status == build.FunctionStatus.SUCCESSFUL

        onnx_model = onnx.load(state.results)
        model_opset = getattr(onnx_model.opset_import[0], "version", None)
        assert user_opset == model_opset

    def test_014_export_only(self):
        build_name = "export_only"

        state = build_model(
            sequence=basic_pytorch_sequence(),
            model=pytorch_model,
            inputs=inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
        )

        assert state.build_status == build.FunctionStatus.SUCCESSFUL
        assert os.path.exists(export.base_onnx_file(state))
        assert not os.path.exists(export.opt_onnx_file(state))

    def test_015_receive_onnx(self):
        """
        Manually export an ONNX file with an opset other than the default
        Then make sure that the state file correctly reflects that opset
        """
        build_name = "receive_onnx"
        onnx_file = f"{build_name} + .onnx"
        user_opset = build.MINIMUM_ONNX_OPSET

        # Make sure we are using an non-default ONNX opset
        assert user_opset != build.DEFAULT_ONNX_OPSET

        # Create ONNX file
        torch.onnx.export(
            pytorch_model,
            input_tensor,
            onnx_file,
            opset_version=user_opset,
            input_names=["input"],
            output_names=["output"],
        )

        # Build the ONNX file
        state = build_model(
            sequence=basic_onnx_sequence(input=onnx_file),
            build_name=build_name,
            rebuild="always",
            monitor=False,
        )

        # Make sure the build was successful
        assert state.build_status == build.FunctionStatus.SUCCESSFUL

        # Get ONNX file's opset
        onnx_model = onnx.load(onnx_file)
        model_opset = getattr(onnx_model.opset_import[0], "version", None)

        # Make sure the ONNX file matches the opset we asked for
        assert user_opset == model_opset

    def test_017_inputs_conversion(self):
        custom_sequence_fp32 = stage.Sequence(
            stages={
                export.ExportPytorchModel(): [],
                export.OptimizeOnnxModel(): [],
            },
        )

        custom_sequence_fp16 = stage.Sequence(
            stages={
                export.ExportPytorchModel(): [],
                export.OptimizeOnnxModel(): [],
                export.ConvertOnnxToFp16(): [],
            },
        )

        # Build model using fp32 inputs
        build_name = "custom_sequence_fp32"
        build_model(
            sequence=custom_sequence_fp32,
            model=pytorch_model,
            inputs=inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
        )

        inputs_path = os.path.join(cache_location, build_name, "inputs.npy")
        assert np.load(inputs_path, allow_pickle=True)[0]["x"].dtype == np.float32

        # Build model using fp16 inputs
        build_name = "custom_sequence_fp16"
        build_model(
            sequence=custom_sequence_fp16,
            model=pytorch_model,
            inputs=inputs,
            build_name="custom_sequence_fp16",
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
        )

        inputs_path = os.path.join(cache_location, build_name, "inputs.npy")
        assert np.load(inputs_path, allow_pickle=True)[0]["x"].dtype == np.float16


if __name__ == "__main__":
    unittest.main()
