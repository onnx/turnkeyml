"""
This script is an example of a tool.py file for Tool Plugin. Such a tool.py 
can be used to redefine the build phase of the turnkey CLI, benchmark_files(),
and build_model() to have any custom behavior.

In this the tool simply passes the build state to the next
tool in the sequence (i.e., this example is a no-op). 

After you install the plugin, you can tell `turnkey` to use this sequence with:

    turnkey -i INPUT_SCRIPT export-pytorch example-plugin-tool 
"""

import argparse
from multiprocessing import Process
from turnkeyml.tools.management_tools import ManagementTool
from turnkeyml.state import State


def print_message(message):
    print(message)


class ExamplePluginTool(ManagementTool):
    """
    Example of a Tool installed by a plugin. Note that this docstring appears
    in the help menu when `turnkey example-plugin-tool -h` is called.
    """

    unique_name = "example-plugin-tool"

    def __init__(self):
        super().__init__()

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="This is an examples tool from the Tool Plugins example",
            add_help=add_help,
        )

        return parser

    def run(self, cache_dir: str):

        # Run using multiprocessing
        message = f"Running using multiprocessing. Cache dir is {cache_dir}"
        process = Process(target=print_message, args=(message,))
        process.start()
        process.join()
