import argparse
import sys
import os
import textwrap as _textwrap
import re
from typing import List
import turnkeyml.common.filesystem as fs
from turnkeyml.build.stage import Stage, ManagementStage, Sequence
from turnkeyml.build.stage_plugins import SUPPORTED_STAGES
from turnkeyml.files_api import benchmark_files


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

    parser.add_argument(
        "stages",
        metavar="stage --stage-args [stage --stage-args...]",
        nargs="?",
        help=f"""\
Available stages that can be sequenced together to perform an evaluation. 

Call `turnkey STAGE -h` to learn more about each stage.

Stage choices: 
{eval_stage_choices}""",
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
    for cmd, argv in stages_invoked.items():
        stage_parsers[cmd].parse_args(argv)

        # Keep track of whether the stages are ManagementStage or not,
        # since ManagementStages are mutually exclusive with evaluation
        # stages
        if not issubclass(stage_classes[cmd], ManagementStage):
            evaluation_stages.append(cmd)

    # Convert stage names into Stage instaces
    stage_instances = {
        stage_classes[cmd](): argv for cmd, argv in stages_invoked.items()
    }

    for cmd, argv in stages_invoked.items():
        print(f"Invoking {cmd}, {argv}")

    sequence = Sequence(stages=stage_instances)

    benchmark_files(**global_args, sequence=sequence)


if __name__ == "__main__":
    main()
