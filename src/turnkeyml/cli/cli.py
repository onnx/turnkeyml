import argparse
import sys
import os
import textwrap as _textwrap
import re
from typing import List
import turnkeyml.common.filesystem as fs
from turnkeyml.build.stage import Stage, Sequence
from turnkeyml.build.stage_plugins import SUPPORTED_STAGES

from turnkeyml.cli.spawn import DEFAULT_TIMEOUT_SECONDS
from turnkeyml.files_api import benchmark_files
from turnkeyml.analyze.status import Verbosity
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
from turnkeyml.common.management_tools import ManagementTool


class PreserveWhiteSpaceWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __add_whitespace(self, idx, amount, text):
        if idx == 0:
            return text
        return (" " * amount) + text

    def _split_lines(self, text, width):
        textRows = text.splitlines()
        for idx, line in enumerate(textRows):
            search = re.search(r"\s*[0-9\-]{0,}\.?\s*", line)
            if line.strip() == "":
                textRows[idx] = " "
            elif search:
                whitespace_needed = search.end()
                lines = [
                    self.__add_whitespace(i, whitespace_needed, x)
                    for i, x in enumerate(_textwrap.wrap(line, width))
                ]
                textRows[idx] = lines

        return [item for sublist in textRows for item in sublist]


def _stage_list_help(stages: List[Stage], subclass) -> str:
    help = ""
    for stage_class in stages:
        if issubclass(stage_class, subclass):
            help = (
                help
                + f" * {stage_class.unique_name}: {stage_class.parser().description}\n"
            )

    return help


def _check_extension(choices, file_name, error_func):
    _, extension = os.path.splitext(file_name.split("::")[0])
    if extension[1:].lower() not in choices:
        error_func(
            f"input_files must end with .py, .onnx, or .txt (got '{file_name}')\n"
        )
    return file_name


def main():

    stage_parsers = {stage.unique_name: stage.parser() for stage in SUPPORTED_STAGES}
    stage_classes = {stage.unique_name: stage for stage in SUPPORTED_STAGES}

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Turnkey build of AI models. "
        "This utility runs stages in a sequence. "
        "To use it, provide a list of stages and "
        "their arguments.",
        formatter_class=PreserveWhiteSpaceWrapRawTextHelpFormatter,
    )

    # Sort stages into categories and format for the help menu
    eval_stage_choices = _stage_list_help(SUPPORTED_STAGES, Stage)
    mgmt_stage_choices = _stage_list_help(SUPPORTED_STAGES, ManagementTool)

    parser.add_argument(
        "stages",
        metavar="stage --stage-args [stage --stage-args...]",
        nargs="?",
        help=f"""\
Available stages that can be sequenced together to perform an evaluation. 

Call `turnkey STAGE -h` to learn more about each stage.

Stage choices: 
{eval_stage_choices}

Management tool choices:
{mgmt_stage_choices}""",
        choices=stage_parsers.keys(),
    )

    parser.add_argument(
        "-i",
        "--input-files",
        nargs="+",
        help="One or more script (.py), ONNX (.onnx), or input list (.txt) files to be benchmarked",
        type=lambda file: _check_extension(("py", "onnx", "txt"), file, parser.error),
    )

    parser.add_argument(
        "-d",
        "--cache-dir",
        help="Build cache directory where the resulting build directories will "
        f"be stored (defaults to {fs.DEFAULT_CACHE_DIR})",
        required=False,
        default=fs.DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--lean-cache",
        dest="lean_cache",
        help="Delete all build artifacts except for log files when the command completes",
        action="store_true",
    )

    parser.add_argument(
        "--labels",
        dest="labels",
        help="Only benchmark the scripts that have the provided labels",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--rebuild",
        choices=build.REBUILD_OPTIONS,
        dest="rebuild",
        help=f"Sets the cache rebuild policy (defaults to {build.DEFAULT_REBUILD_POLICY})",
        required=False,
        default=build.DEFAULT_REBUILD_POLICY,
    )

    slurm_or_processes_group = parser.add_mutually_exclusive_group()

    slurm_or_processes_group.add_argument(
        "--use-slurm",
        dest="use_slurm",
        help="Execute on Slurm instead of using local compute resources",
        action="store_true",
    )

    slurm_or_processes_group.add_argument(
        "--process-isolation",
        dest="process_isolation",
        help="Isolate evaluating each input into a separate process",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Build timeout, in seconds, after which a build will be canceled "
        f"(default={DEFAULT_TIMEOUT_SECONDS}). Only "
        "applies when --process-isolation or --use-slurm is also used.",
    )

    default_verbosity = Verbosity.AUTO.value
    parser.add_argument(
        "--verbosity",
        choices=[field.value for field in Verbosity],
        default=default_verbosity,
        help="Verbosity of the status updates printed to the command line "
        f"(default={default_verbosity}). '{Verbosity.DYNAMIC.value}': "
        "take over the terminal, updating "
        " it with a summary of all turnkey information. "
        f"'{Verbosity.STATIC.value}': print each evaluation as it takes place and "
        "never clear the terminal.",
    )

    # Break sys.argv into categories based on which stages were invoked
    # Arguments that are passed prior to invoking a stage are categorized as
    # global arguments that should be used to initialize the state.
    current_stage = "globals"
    stages_invoked = {current_stage: []}
    cmd = sys.argv[1:]
    while len(cmd):
        if cmd[0] in stage_parsers.keys():
            current_stage = cmd.pop(0)
            stages_invoked[current_stage] = []
        else:
            stages_invoked[current_stage].append(cmd.pop(0))

    # Do one pass of parsing to figure out if -h was used
    global_args = vars(parser.parse_args(stages_invoked["globals"]))
    # Remove "stages" from global args because it was just there
    # as a placeholder
    global_args.pop("stages")
    parser.parse_args(stages_invoked["globals"])
    # Remove globals from the list since its already been parsed
    stages_invoked.pop("globals")
    evaluation_stages = []
    management_stages = []
    for cmd, argv in stages_invoked.items():
        stage_parsers[cmd].parse_args(argv)

        # Keep track of whether the stages are ManagementStage or not,
        # since ManagementStages are mutually exclusive with evaluation
        # stages
        if issubclass(stage_classes[cmd], ManagementTool):
            management_stages.append(cmd)
        else:
            evaluation_stages.append(cmd)

    if len(management_stages) > 0 and len(evaluation_stages) > 0:
        raise exp.ArgError(
            "This call to turnkey invoked both management and "
            "evaluation stages, however each call to turnkey "
            "is only allowed to invoke one or the other. "
            f"Management stages: {management_stages};"
            f"Evaluation stages: {evaluation_stages}."
        )

    if len(management_stages) == 0 and len(evaluation_stages) == 0:
        raise exp.ArgError(
            "Calls to tunrkey are required to call at least "
            "one stage or management tool."
        )

    # Convert stage names into Stage instaces
    stage_instances = {
        stage_classes[cmd](): argv for cmd, argv in stages_invoked.items()
    }

    if len(evaluation_stages) > 0:
        # Run the evaluation stages as a build
        sequence = Sequence(stages=stage_instances)
        benchmark_files(sequence=sequence, **global_args)
    else:
        # Run the management stages
        for management_stage, argv in stage_instances.items():
            # Support "~" in the cache_dir argument
            parsed_cache_dir = os.path.expanduser(global_args[fs.Keys.CACHE_DIR])
            management_stage.parse_and_run(parsed_cache_dir, argv)


if __name__ == "__main__":
    main()
