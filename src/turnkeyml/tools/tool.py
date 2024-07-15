import abc
import sys
import time
import os
import argparse
from typing import List, Tuple, Dict
from multiprocessing import Process
import psutil
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as fs
from turnkeyml.sequence.state import State


def _spinner(message):
    try:
        parent_process = psutil.Process(pid=os.getppid())
        while parent_process.status() == psutil.STATUS_RUNNING:
            for cursor in ["   ", ".  ", ".. ", "..."]:
                time.sleep(0.5)
                status = f"      {message}{cursor}\r"
                sys.stdout.write(status)
                sys.stdout.flush()
    except psutil.NoSuchProcess:
        # If the parent process stopped existing, we can
        # safely assume the spinner no longer needs to spin
        # NOTE: this only seems to be needed on Windows
        pass


def _name_is_file_safe(name: str):
    """
    Make sure the name can be used in a filename
    """

    allowed_in_unique_name = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )

    if len(name) == 0:
        msg = """
        Stage __init__() was passed a unique_name with no length. A
        uniquely identifying unique_name is required.
        """
        raise ValueError(msg)

    for char in name:
        if char not in allowed_in_unique_name:
            msg = f"""
            Stage __init__() was passed a unique_name:
            {name}
            with illegal characters. The unique_name must be safe to
            use in a filename, meaning it can only use characters: {allowed_in_unique_name}
            """
            raise ValueError(msg)


class StageParser(argparse.ArgumentParser):
    def error(self, message):
        if message.startswith("unrecognized arguments"):
            unrecognized = message.split(": ")[1]
            if not unrecognized.startswith("-"):
                # This was probably a misspelled stage name
                message = message + (
                    f". If `{unrecognized}` was intended to invoke "
                    "a stage, please run `turnkey -h` and check the spelling and "
                    "availability of that stage."
                )
        self.print_usage()
        printing.log_error(message)
        self.exit(2)


class Tool(abc.ABC):

    unique_name: str

    @classmethod
    def helpful_parser(cls, description: str, **kwargs):
        epilog = (
            f"`{cls.unique_name}` is a Stage. It is intended to be invoked as "
            "part of a sequence of Stages, for example: `turnkey -i INPUTS stage-one "
            "stage-two stage-three`"
        )

        return StageParser(
            prog=f"turnkey {cls.unique_name}",
            description=description,
            epilog=epilog,
            **kwargs,
        )

    def status_line(self, successful, verbosity):
        """
        Print a line of status information for this Stage into the monitor.
        """
        if verbosity:
            # Only use special characters when the terminal encoding supports it
            if sys.stdout.encoding == "utf-8":
                success_tick = "✓"
                fail_tick = "×"
            else:
                success_tick = "+"
                fail_tick = "x"

            if successful is None:
                # Initialize the message
                printing.logn(f"      {self.monitor_message}   ")
            elif successful:
                # Print success message
                printing.log(f"    {success_tick} ", c=printing.Colors.OKGREEN)
                printing.logn(self.monitor_message + "   ")
            else:
                # successful == False, print failure message
                printing.log(f"    {fail_tick} ", c=printing.Colors.FAIL)
                printing.logn(self.monitor_message + "   ")

    def __init__(
        self,
        monitor_message,
    ):
        _name_is_file_safe(self.__class__.unique_name)

        self.status_key = f"{fs.Keys.STAGE_STATUS}:{self.__class__.unique_name}"
        self.duration_key = f"{fs.Keys.STAGE_DURATION}:{self.__class__.unique_name}"
        self.monitor_message = monitor_message
        self.progress = None
        self.logfile_path = None
        self.stages = None
        # Stages can provide a list of keys that can be found in
        # evaluation stats. Those key:value pairs will be presented
        # in the status at the end of the build.
        self.status_stats = []

    @abc.abstractmethod
    def fire(self, state: State) -> State:
        """
        Developer-defined function to fire the stage.
        In less punny terms, this is the function that
        build_model() will run to implement a model-to-model
        transformation on the flow to producing a Model.
        """

    @staticmethod
    @abc.abstractmethod
    def parser() -> argparse.ArgumentParser:
        """
        Static method that returns an ArgumentParser that defines the command
        line interface for this Stage.
        """

    # pylint: disable=unused-argument
    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        """
        Run the parser and return a Namespace of keyword arguments that the user
        passed to the Stage via the command line.

        Stages should extend this function only if they require specific parsing
        logic, for example decoding the name of a data type into a data type class.

        Args:
            state: the same state passed into the run method of the Stage, useful if
                the parse decoding logic needs to take the state into account.
            args: command line arguments passed from the CLI.
            known_only: this argument allows the CLI framework to
                incrementally parse complex commands.
        """

        if known_only:
            parsed_args = self.__class__.parser().parse_args(args)
        else:
            parsed_args, _ = self.__class__.parser().parse_known_args(args)

        return parsed_args

    def parse_and_fire(self, state: State, args, known_only=True) -> Dict:
        """
        Helper function to parse CLI arguments into the args expected
        by fire(), and then forward them into the fire() method.
        """

        parsed_args = self.parse(state, args, known_only)
        return self.fire_helper(state, **parsed_args.__dict__)

    def fire_helper(self, state: State, **kwargs) -> Tuple[State, int]:
        """
        Wraps the user-defined .fire method with helper functionality.
        Specifically:
            - Provides a path to a log file
            - Redirects the stdout of the stage to that log file
            - Monitors the progress of the stage on the command line,
                including in the event of an exception
        """

        # Set the build status to INCOMPLETE to indicate that a Stage
        # started running. This allows us to test whether the Stage exited
        # unexpectedly, before it was able to set ERROR
        state.build_status = build.FunctionStatus.INCOMPLETE

        self.logfile_path = os.path.join(
            build.output_dir(state.cache_dir, state.build_name),
            f"log_{self.unique_name}.txt",
        )

        if state.monitor:
            self.progress = Process(target=_spinner, args=[self.monitor_message])
            self.progress.start()

        try:
            # Execute the build stage
            with build.Logger(self.monitor_message, self.logfile_path):
                state = self.fire(state, **kwargs)

        except Exception:  # pylint: disable=broad-except
            self.status_line(
                successful=False,
                verbosity=state.monitor,
            )
            state.build_status = build.FunctionStatus.ERROR
            raise

        else:
            self.status_line(successful=True, verbosity=state.monitor)

            # Stages should not set build.FunctionStatus.SUCCESSFUL for the whole build,
            # as that is reserved for Sequence.launch()
            if state.build_status == build.FunctionStatus.SUCCESSFUL:
                raise exp.ToolError(
                    "TurnkeyML Stages are not allowed to set "
                    "`state.build_status == build.FunctionStatus.SUCCESSFUL`, "
                    "however that has happened. If you are a plugin developer, "
                    "do not do this. If you are a user, please file an issue at "
                    "https://github.com/onnx/turnkeyml/issues."
                )

        finally:
            if state.monitor:
                self.progress.terminate()

        return state
