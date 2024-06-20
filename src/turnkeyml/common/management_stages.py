import argparse
import abc
import turnkeyml.common.filesystem as fs
from turnkeyml.common.filesystem import State
from turnkeyml.version import __version__ as turnkey_version


class ManagementTool(abc.ABC):
    """
    Intended for management functions, such as managing the cache
    or printing the version number.
    """

    @staticmethod
    @abc.abstractmethod
    def parser() -> argparse.ArgumentParser:
        """
        Static method that returns an ArgumentParser that defines the command
        line interface for this Stage.
        """

    # pylint: disable=unused-argument
    def parse(self, args, known_only=True) -> argparse.Namespace:
        """
        Run the parser and return a Namespace of keyword arguments that the user
        passed to the Stage via the command line.

        Tools should extend this function only if they require specific parsing
        logic.

        Args:
            args: command line arguments passed from the CLI.
            known_only: this argument allows the CLI framework to
                incrementally parse complex commands.
        """

        if known_only:
            parsed_args = self.__class__.parser().parse_args(args)
        else:
            parsed_args, _ = self.__class__.parser().parse_known_args(args)

        return parsed_args

    @abc.abstractmethod
    def run(self):
        """
        Execute the functionality of the Tool.
        """

    def parse_and_run(self, args, known_only=True):
        """
        Helper function to parse CLI arguments into the args expected
        by run(), and then forward them into the fire() method.
        """

        parsed_args = self.parse(args, known_only)
        self.run(**parsed_args.__dict__)


class Version(ManagementTool):
    """
    Simply prints the version number of the turnkeyml installation.
    """

    unique_name = "version"

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Print the turnkeyml version number",
            add_help=add_help,
        )

        return parser

    def run(self):
        print(turnkey_version)


class Cache(ManagementTool):
    """
    Wraps a set of tools for managing the turnkey build cache.
    """

    unique_name = "cache"

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Manage the turnkey build cache "
            f"(default location {fs.DEFAULT_CACHE_DIR}, select with the --cache-dir option)",
            add_help=add_help,
        )

        build_selection_group = parser.add_mutually_exclusive_group(required=True)

        build_selection_group.add_argument(
            "--build-names",
            nargs="+",
            help="Name of the specific build(s) to be operated upon, within the cache directory",
        )

        build_selection_group.add_argument(
            "--all",
            "-a",
            help="Operate on all the builds in the cache",
            action="store_true",
        )

        action_group = parser.add_mutually_exclusive_group(required=True)

        action_group.add_argument(
            "--list",
            "-l",
            action="store_true",
            help="List all of the builds in the cache",
        )

        action_group.add_argument(
            "--stats",
            "-s",
            action="store_true",
            help="Print the collected stats for the selected build(s)",
        )

        action_group.add_argument(
            "--delete",
            action="store_true",
            help="Permanently delete the selected build(s)",
        )

        action_group.add_argument(
            "--clean",
            action="store_true",
            help="Remove the build artifacts from the selected build(s)",
        )

        return parser

    def run(self):
        pass
