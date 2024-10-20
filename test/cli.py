"""
Tests focused on the command-level functionality of turnkey CLI
"""

import os
import shutil
import glob
import csv
from typing import List, Union
import unittest
from unittest.mock import patch
import sys
import io
from contextlib import redirect_stdout
import yaml
import platform
import torch
from turnkeyml.cli.cli import main as turnkeycli
import turnkeyml.tools.report as report
import turnkeyml.common.filesystem as fs
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exceptions
import turnkeyml.common.test_helpers as common
from turnkeyml.common.test_helpers import assert_success_of_builds


def bash(cmd: str) -> List[str]:
    """
    Emulate behavior of bash terminal when listing files
    """
    return glob.glob(cmd)


def flatten(lst: List[Union[str, List[str]]]) -> List[str]:
    """
    Flatten List[Union[str, List[str]]] into a List[str]
    """
    flattened = []
    for element in lst:
        if isinstance(element, list):
            flattened.extend(element)
        else:
            flattened.append(element)
    return flattened


class SmallPytorchModel(torch.nn.Module):
    def __init__(self):
        super(SmallPytorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        output = self.fc(x)
        return output


# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        fs.rmdir(cache_dir)
        fs.rmdir(new_cache_dir)

        return super().setUp()

    def test_001_cli_single(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    def test_002_search_multiple(self):
        # Test the first model in the corpus
        test_scripts = list(common.test_scripts_dot_py.keys())

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_scripts[0]),
            os.path.join(corpus_dir, test_scripts[1]),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_scripts[0], test_scripts[1]], cache_dir)

    def test_003_cli_build_dir(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        test_scripts = common.test_scripts_dot_py.keys()

        testargs = [
            "turnkey",
            "-i",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        assert_success_of_builds(test_scripts, cache_dir)

    def test_004_cli_list(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to list
        testargs = [
            "turnkey",
            "-i",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "--cache-dir",
                cache_dir,
                "cache",
                "--list",
                "--all",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name in f.getvalue(), f"{script_name} {f.getvalue()}"

    def test_005_cli_delete(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus
        # - test_cli_list

        # Build the test corpus so we have builds to delete
        testargs = [
            "turnkey",
            "-i",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "--cache-dir",
                cache_dir,
                "cache",
                "--list",
                "--all",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name in f.getvalue()

        # Delete the builds
        testargs = [
            "turnkey",
            "--cache-dir",
            cache_dir,
            "cache",
            "--delete",
            "--all",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Make sure the builds are gone
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "--cache-dir",
                cache_dir,
                "cache",
                "--list",
                "--all",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name not in f.getvalue()

    def test_006_cli_stats(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to print
        testargs = [
            "turnkey",
            "-i",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can print the builds in the cache
        for test_script in common.test_scripts_dot_py.keys():
            test_script_path = os.path.join(corpus_dir, test_script)
            builds, script_name = fs.get_builds_from_file(cache_dir, test_script_path)

            for build_name in builds:
                # Make sure each build can be accessed with `turnkey cache stats`
                with redirect_stdout(io.StringIO()) as f:
                    testargs = [
                        "turnkey",
                        "--cache-dir",
                        cache_dir,
                        "cache",
                        "--stats",
                        "--build-names",
                        build_name,
                    ]
                    with patch.object(sys, "argv", testargs):
                        turnkeycli()

                    assert script_name in f.getvalue()

                # Make sure the stats YAML file contains the fields
                # required for producing a report
                stats_file = os.path.join(
                    build.output_dir(cache_dir, build_name), "turnkey_stats.yaml"
                )
                with open(stats_file, "r", encoding="utf8") as stream:
                    stats_dict = yaml.load(stream, Loader=yaml.FullLoader)

                assert isinstance(stats_dict["hash"], str), stats_dict["hash"]
                assert isinstance(stats_dict["parameters"], int), stats_dict[
                    "parameters"
                ]
                assert isinstance(
                    stats_dict["onnx_input_dimensions"], dict
                ), stats_dict["onnx_input_dimensions"]
                assert isinstance(
                    stats_dict["onnx_model_information"], dict
                ), stats_dict["onnx_model_information"]
                assert isinstance(stats_dict["onnx_ops_counter"], dict), stats_dict[
                    "onnx_ops_counter"
                ]
                assert isinstance(stats_dict["system_info"], dict), stats_dict[
                    "system_info"
                ]

                # Make sure the turnkey_stats has the expected ONNX opset
                assert (
                    stats_dict["onnx_model_information"]["opset"]
                    == build.DEFAULT_ONNX_OPSET
                ), stats_dict["onnx_model_information"]["opset"]

                # Make sure the turnkey_stats has the necessary fields used in the onnx model zoo
                assert isinstance(stats_dict["author"], str), stats_dict["author"]
                assert isinstance(stats_dict["model_name"], str), stats_dict[
                    "model_name"
                ]
                assert isinstance(stats_dict["task"], str), stats_dict["task"]

    def test_007_cli_version(self):
        # Get the version number
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "version",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        # Make sure we get back a 3-digit number
        assert len(f.getvalue().split(".")) == 3

    def test_008_cli_turnkey_args(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        # Set as many turnkey args as possible
        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    # TODO: Investigate why this test is only failing on Windows CI
    @unittest.skipIf(platform.system() == "Windows", "Windows CI only failure")
    def test_08_cli_onnx_opset(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "--opset",
            str(user_opset),
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds(
            [test_script], cache_dir, check_perf=False, check_opset=user_opset
        )

    def test_09_cli_process_isolation(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "-i",
                os.path.join(corpus_dir, test_script),
                "--cache-dir",
                cache_dir,
                "--process-isolation",
                "discover",
                "export-pytorch",
                "--opset",
                "17",
                "optimize-ort",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

            assert_success_of_builds([test_script], cache_dir, check_perf=False)

    @unittest.skipIf(
        platform.system() == "Windows",
        "Skipping, as torch.compile is not supported on Windows"
        "Revisit when torch.compile for Windows is supported",
    )
    def test_010_skip_compiled(self):
        test_script = "compiled.py"
        testargs = [
            "turnkey",
            "-i",
            os.path.join(extras_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        builds_found = assert_success_of_builds([test_script], cache_dir)

        # Compile.py contains two Pytorch models.
        # One of those is compiled and should be skipped.
        assert builds_found == 1

    def test_011_invalid_file_type(self):
        # Ensure that we get an error when running turnkey with invalid input_files
        with self.assertRaises(SystemExit):
            testargs = [
                "turnkey",
                "-i",
                "gobbledegook",
                "discover",
                "export-pytorch",
                "optimize-ort",
            ]
            with patch.object(sys, "argv", flatten(testargs)):
                turnkeycli()

    def test_012_cli_export_only(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir, check_onnx_file_count=1)

    def test_013_cli_onnx_model(self):
        """
        Manually export an ONNX file, then feed it into the CLI
        """
        build_name = "receive_onnx"
        onnx_file = os.path.join(corpus_dir, f"{build_name}.onnx")

        # Create ONNX file
        torch.onnx.export(
            pytorch_model,
            input_tensor,
            onnx_file,
            opset_version=build.DEFAULT_ONNX_OPSET,
            input_names=["input"],
            output_names=["output"],
        )

        testargs = [
            "turnkey",
            "-i",
            onnx_file,
            "--cache-dir",
            cache_dir,
            "load-onnx",
            "convert-fp16",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([build_name], cache_dir)

    def test_014_cli_onnx_model_opset(self):
        """
        Manually export an ONNX file with a non-default opset, then feed it into the CLI
        """
        build_name = "receive_onnx_opset"
        onnx_file = os.path.join(corpus_dir, f"{build_name}.onnx")
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

        testargs = [
            "turnkey",
            "-i",
            onnx_file,
            "--cache-dir",
            cache_dir,
            "load-onnx",
            "convert-fp16",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([build_name], cache_dir)

    def test_015_non_existent_file(self):
        # Ensure we get an error when loading a non existent file
        with self.assertRaises(exceptions.ArgError):
            filename = "thou_shall_not_exist.py"
            with redirect_stdout(io.StringIO()) as f:
                testargs = [
                    "turnkey",
                    "-i",
                    filename,
                    "discover",
                    "export-pytorch",
                    "optimize-ort",
                ]
                with patch.object(sys, "argv", testargs):
                    turnkeycli()

    def test_016_non_existent_file_prefix(self):
        # Ensure we get an error when loading a non existent file
        with self.assertRaises(exceptions.ArgError):
            file_prefix = "non_existent_prefix_*.py"
            with redirect_stdout(io.StringIO()) as f:
                testargs = [
                    "turnkey",
                    "-i",
                    file_prefix,
                    "discover",
                    "export-pytorch",
                    "optimize-ort",
                ]
                with patch.object(sys, "argv", testargs):
                    turnkeycli()

    def test_017_input_text_file(self):
        """
        Ensure that we can intake .txt files
        """

        testargs = [
            "turnkey",
            "-i",
            os.path.join(extras_dir, "selected_models.txt"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        builds_found = assert_success_of_builds(["linear.py", "linear2.py"], cache_dir)
        assert (
            builds_found == 3
        ), f"Expected 3 builds (1 for linear.py, 2 for linear2.py), but got {builds_found}."

    def test_018_cli_timeout(self):
        """
        Make sure that the --timeout option and its associated reporting features work.

        timeout.py is designed to take 20s to discover, which gives us the
        opportunity to kill it with a timeout.
        """

        testargs = [
            "turnkey",
            "-i",
            os.path.join(extras_dir, "timeout.py"),
            "--cache-dir",
            cache_dir,
            "--process-isolation",
            "--timeout",
            "10",
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        testargs = [
            "turnkey",
            "report",
            "--input-caches",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Read generated CSV file and make sure the build was killed by the timeout
        summary_csv_path = report.get_report_name()
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Check the summary for "killed", but also accept the edge case that
        # the build timed out before the stats.yaml was created
        try:
            timeout_summary = summary[0]

            assert timeout_summary["build_status"] == "timeout", timeout_summary[
                "build_status"
            ]
        except IndexError:
            # Edge case where the CSV is empty because the build timed out before
            # the stats.yaml was created, which in turn means the CSV is empty
            pass
        except KeyError:
            # Edge case where the CSV only contains a key for "error_log"
            assert "timeout" in timeout_summary["error_log"]

    def test_019_cli_report(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        test_scripts = common.test_scripts_dot_py.keys()

        # Build the test corpus so we have builds to report
        testargs = [
            "turnkey",
            "-i",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        testargs = [
            "turnkey",
            "report",
            "--input-caches",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Read generated CSV file
        summary_csv_path = report.get_report_name()
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Check if csv file contains all expected rows and columns
        expected_cols = [
            "model_name",
            "author",
            "class",
            "parameters",
            "hash",
            "selected_sequence_of_tools",
        ]
        linear_summary = summary[1]
        assert len(summary) == len(test_scripts)
        for elem in expected_cols:
            assert (
                elem in linear_summary
            ), f"Couldn't find expected key {elem} in results spreadsheet"

        # Check whether all rows we expect to be populated are actually populated
        assert (
            linear_summary["model_name"] == "linear2"
        ), f"Wrong model name found {linear_summary['model_name']}"
        assert (
            linear_summary["author"] == "turnkey"
        ), f"Wrong author name found {linear_summary['author']}"
        assert (
            linear_summary["class"] == "TwoLayerModel"
        ), f"Wrong class found {linear_summary['model_class']}"
        assert (
            linear_summary["hash"] == "80b93950"
        ), f"Wrong hash found {linear_summary['hash']}"

        # Make sure the report.get_dict() API works
        result_dict = report.get_dict(
            summary_csv_path,
            [
                "selected_sequence_of_tools",
                "tool_duration:discover",
                "tool_duration:export-pytorch",
                "tool_duration:optimize-ort",
                "tool_status:discover",
                "tool_status:export-pytorch",
                "tool_status:optimize-ort",
            ],
        )
        for result in result_dict.values():
            # All of the models should have exported to ONNX and optimized the ONNX model
            for tool in ["export-pytorch", "optimize-ort"]:
                assert tool in result["selected_sequence_of_tools"]
                duration = result[f"tool_duration:{tool}"]
                status = result[f"tool_status:{tool}"]
                assert (
                    status == "successful"
                ), f"Unexpected status {status} for tool '{tool}'"
                try:
                    assert (
                        float(duration) > 0
                    ), f"Tool {tool} has invalid duration '{duration}'"
                except ValueError:
                    # Catch the case where the value is not numeric
                    assert False, f"Tool {tool} has invalid duration {duration}"

    def test_020_cli_onnx_verify(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "verify-exporter",
            "export-pytorch",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    def test_021_cli_fp16_convert(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
            "optimize-ort",
            "convert-fp16",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    def test_022_cli_cache_move(self):

        test_script = list(common.test_scripts_dot_py.keys())[0]

        # Build a model into the default cache location
        testargs = [
            "turnkey",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "discover",
            "export-pytorch",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Move the cache to a new location
        shutil.move(cache_dir, new_cache_dir)

        # Get the build state file in its new location
        selected_build = fs.get_available_builds(new_cache_dir)[-1]
        state_file_path = os.path.join(
            new_cache_dir, selected_build, f"{selected_build}_state.yaml"
        )

        # Build the cached build in its new location
        testargs = [
            "turnkey",
            "-i",
            state_file_path,
            "load-build",
            "optimize-ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        test_script = selected_build + ".py"
        assert_success_of_builds([test_script], new_cache_dir, check_perf=False)


if __name__ == "__main__":
    # Create a cache directory a directory with test models
    cache_dir, corpus_dir = common.create_test_dir("cli")
    new_cache_dir = f"{cache_dir}2"

    extras_dir = os.path.join(corpus_dir, "extras")
    os.makedirs(extras_dir, exist_ok=True)

    for key, value in common.extras_python(corpus_dir).items():
        file_path = os.path.join(extras_dir, key)

        with open(file_path, "w", encoding="utf") as f:
            f.write(value)

    unittest.main()
