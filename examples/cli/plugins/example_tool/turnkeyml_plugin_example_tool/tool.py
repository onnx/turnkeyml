"""
This script is an example of a tool.py file for Tool Plugin. Such a tool.py 
can be used to redefine the build phase of the turnkey CLI, benchmark_files(),
and build_model() to have any custom behavior.

In this example tool.py file we are simply passing the build state to the next
tool in the sequence (i.e., this example is a no-op). 

After you install the plugin, you can tell `turnkey` to use this sequence with:

    turnkey -i INPUT_SCRIPT export-pytorch exmaple-plugin-tool 
"""

import argparse
from turnkeyml.tools import Tool
from turnkeyml.state import State


class ExamplePluginTool(Tool):

    unique_name = "example-plugin-tool"

    def __init__(self):
        super().__init__(
            monitor_message="Special step expected by CombinedExampleRT",
        )

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="This is an examples tool from the Tool Plugins example",
            add_help=add_help,
        )

        return parser

    def run(self, state: State):
        return state
