import argparse
import sys
import os
import textwrap as _textwrap
import re
from difflib import get_close_matches
from typing import List
import turnkeyml.common.filesystem as fs
from turnkeyml.sequence import Sequence
from turnkeyml.tools import Tool
from turnkeyml.sequence.tool_plugins import SUPPORTED_TOOLS
from turnkeyml.cli.spawn import DEFAULT_TIMEOUT_SECONDS
from turnkeyml.files_api import benchmark_files
import turnkeyml.common.build as build
import turnkeyml.common.printing as printing
from turnkeyml.tools.management_tools import ManagementTool


class CustomArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        self.print_usage()
        printing.log_error(message)
        self.exit(2)


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


def _tool_list_help(tools: List[Tool], subclass) -> str:
    help = ""
    for tool_class in tools:
        if issubclass(tool_class, subclass):
            help = (
                help
                + f" * {tool_class.unique_name}: {tool_class.parser().description}\n"
            )

    return help


def _check_extension(
    choices: List[str], file_name: str, error_func: callable, tool_names: List[str]
):
    _, extension = os.path.splitext(file_name.split("::")[0])
    if not extension:
        close_matches = get_close_matches(file_name, tool_names)
        if close_matches:
            # Mispelled tool names can be picked up as input files, so we check
            # for this case here and try to provide a better suggestion
            error_func(
                f"unrecognized argument '{file_name}', did you mean '{close_matches[0]}'?"
            )
        else:
            error_func(
                f"{file_name} was recognized as an argument to `--input-files`, "
                "however it is not a file name (no file extension). If it was "
                "meant to be a tool name, please check whether that tool is "
                "available and correctly spelled in the list of available tools "
                "when calling `turnkey -h`."
            )
    if extension[1:].lower() not in choices:
        error_func(
            f"input_files must end with .py, .onnx, or .txt (got '{file_name}')\n"
        )
    return file_name


def main():

    tool_parsers = {tool.unique_name: tool.parser() for tool in SUPPORTED_TOOLS}
    tool_classes = {tool.unique_name: tool for tool in SUPPORTED_TOOLS}

    # Define the argument parser
    parser = CustomArgumentParser(
        description="Turnkey build of AI models. "
        "This utility runs tools in a sequence. "
        "To use it, provide a list of tools and "
        "their arguments.",
        formatter_class=PreserveWhiteSpaceWrapRawTextHelpFormatter,
    )

    # Sort tools into categories and format for the help menu
    eval_tool_choices = _tool_list_help(SUPPORTED_TOOLS, Tool)
    mgmt_tool_choices = _tool_list_help(SUPPORTED_TOOLS, ManagementTool)

    parser.add_argument(
        "tools",
        metavar="tool --tool-args [tool --tool-args...]",
        nargs="?",
        help=f"""\
Available tools that can be sequenced together to perform an evaluation. 

Call `turnkey TOOL -h` to learn more about each tool.

Tool choices: 
{eval_tool_choices}

Management tool choices:
{mgmt_tool_choices}""",
        choices=tool_parsers.keys(),
    )

    parser.add_argument(
        "-i",
        "--input-files",
        nargs="+",
        help="One or more script (.py), ONNX (.onnx), or input list (.txt) files to be benchmarked",
        type=lambda file: _check_extension(
            ("py", "onnx", "txt"), file, parser.error, tool_classes
        ),
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

    # run as if "-h" was passed if no parameters are passed
    if len(sys.argv) == 1:
        sys.argv.append("-h")

    # Break sys.argv into categories based on which tools were invoked
    # Arguments that are passed prior to invoking a tool are categorized as
    # global arguments that should be used to initialize the state.
    current_tool = "globals"
    tools_invoked = {current_tool: []}
    cmd = sys.argv[1:]
    while len(cmd):
        if cmd[0] in tool_parsers.keys():
            # Make sure each tool was only called once
            if cmd[0] in tools_invoked.keys():
                parser.error(
                    "A single call to turnkey can only invoke each tool once, "
                    f"however this call invokes tool {cmd[0]} multiple times."
                )
            current_tool = cmd.pop(0)
            tools_invoked[current_tool] = []
        else:
            tools_invoked[current_tool].append(cmd.pop(0))

    # Do one pass of parsing to figure out if -h was used
    global_args = vars(parser.parse_args(tools_invoked["globals"]))
    # Remove "tools" from global args because it was just there
    # as a placeholder
    global_args.pop("tools")
    parser.parse_args(tools_invoked["globals"])
    # Remove globals from the list since its already been parsed
    tools_invoked.pop("globals")
    evaluation_tools = []
    management_tools = []
    for cmd, argv in tools_invoked.items():
        tool_parsers[cmd].parse_args(argv)

        # Keep track of whether the tools are ManagementTool or not,
        # since ManagementTools are mutually exclusive with evaluation
        # tools
        if issubclass(tool_classes[cmd], ManagementTool):
            management_tools.append(cmd)
        else:
            evaluation_tools.append(cmd)

    if len(management_tools) > 0 and len(evaluation_tools) > 0:
        parser.error(
            "This call to turnkey invoked both management and "
            "evaluation tools, however each call to turnkey "
            "is only allowed to invoke one or the other. "
            f"Management tools: {management_tools};"
            f"Evaluation tools: {evaluation_tools}."
        )

    if len(management_tools) == 0 and len(evaluation_tools) == 0:
        parser.error(
            "Calls to tunrkey are required to call at least "
            "one tool or management tool."
        )

    # Convert tool names into Tool instaces
    tool_instances = {tool_classes[cmd](): argv for cmd, argv in tools_invoked.items()}

    if len(evaluation_tools) > 0:
        # Run the evaluation tools as a build
        sequence = Sequence(tools=tool_instances)
        benchmark_files(sequence=sequence, **global_args)
    else:
        # Run the management tools
        for management_tool, argv in tool_instances.items():
            # Support "~" in the cache_dir argument
            parsed_cache_dir = os.path.expanduser(global_args[fs.Keys.CACHE_DIR])
            management_tool.parse_and_run(parsed_cache_dir, argv)


if __name__ == "__main__":
    main()
