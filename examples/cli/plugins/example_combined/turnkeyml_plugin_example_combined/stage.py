import argparse
from turnkeyml.tools import Tool
from turnkeyml.sequence.state import State

combined_seq_name = "example-combined-seq"


class CombinedExampleStage(Tool):
    """
    This is an empty Stage that we include in our example that provides both
    a sequence and a runtime in a single plugin package.
    """

    unique_name = "combined-example-stage"

    def __init__(self):
        super().__init__(
            monitor_message="Special step expected by CombinedExampleRT",
        )

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="This is an examples stage from the combined example",
            add_help=add_help,
        )

        return parser

    def fire(self, state: State):
        return state
