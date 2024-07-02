"""
This script is an example of a stage.py file for Stage Plugin. Such a stage.py 
can be used to redefine the build phase of the turnkey CLI, benchmark_files(),
and build_model() to have any custom behavior.

In this example stage.py file we are simply passing the build state to the next
stage in the sequence (i.e., this example is a no-op). 

After you install the plugin, you can tell `turnkey` to use this sequence with:

    turnkey -i INPUT_SCRIPT export-pytorch exmaple-plugin-stage 
"""

import argparse
from turnkeyml.build.stage import Stage
import turnkeyml.common.filesystem as fs


class ExamplePluginStage(Stage):

    unique_name = "example-plugin-stage"

    def __init__(self):
        super().__init__(
            monitor_message="Special step expected by CombinedExampleRT",
        )

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="This is an examples stage from the Stage Plugins example",
            add_help=add_help,
        )

        return parser

    def fire(self, state: fs.State):
        return state
