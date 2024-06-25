import abc
import sys
import time
import os
import argparse
import traceback
from typing import List, Tuple, Dict
from multiprocessing import Process
import psutil
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as fs
import turnkeyml.analyze.status as status


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


def _store_traceback(invocation_info: status.UniqueInvocationInfo):
    """
    Store the traceback from an exception into invocation_info so that
    we can print it during the status update.
    """

    exc_type, exc_value, exc_traceback = sys.exc_info()
    invocation_info.traceback = traceback.format_exception(
        exc_type, exc_value, exc_traceback
    )

    # Remove line breaks and sequences of spaces from status message
    invocation_info.status_message = " ".join(invocation_info.status_message.split())


class Stage(abc.ABC):

    unique_name: str

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
    def fire(self, state: fs.State) -> fs.State:
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
    def parse(self, state: fs.State, args, known_only=True) -> argparse.Namespace:
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

    def parse_and_fire(self, state: fs.State, args, known_only=True) -> Dict:
        """
        Helper function to parse CLI arguments into the args expected
        by fire(), and then forward them into the fire() method.
        """

        parsed_args = self.parse(state, args, known_only)
        return self.fire_helper(state, **parsed_args.__dict__)

    def fire_helper(self, state: fs.State, **kwargs) -> Tuple[fs.State, int]:
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
                raise exp.StageError(
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


def _rewind_stdout(lines: int = 1):
    """
    Helper function for the command line monitor. Moves the cursor up a
    certain number of lines in the terminal, corresponding to the
    status line for a Stage, so that we can update the status of
    that Stage.
    """
    rewind_stdout_one_line = "\033[1A"
    rewind_multiple_lines = rewind_stdout_one_line * lines
    print(rewind_multiple_lines, end="")


class Sequence:
    """
    Helper class to launch and manage build stages.
    """

    def __init__(
        self,
        stages: Dict[Stage, List[str]],
    ):

        self.stages = stages

        # Make sure all the stage names are unique
        self.stage_names = [stage.__class__.unique_name for stage in self.stages.keys()]

        if len(self.stage_names) != len(set(self.stage_names)):
            msg = f"""
            All Stages in a Sequence must have unique unique_names, however Sequence
            received duplicates in the list of names: {self.stage_names}
            """
            raise ValueError(msg)

    def show_monitor(self, state: fs.State, verbosity: bool):
        """
        Displays the monitor on the terminal. The purpose of the monitor
        is to show the status of each stage (success, failure, not started yet,
        or in-progress).
        """

        if verbosity:
            print()

            printing.logn(
                f'Building "{state.build_name}"',
                c=printing.Colors.BOLD,
            )

            for stage in self.stages:
                stage.status_line(successful=None, verbosity=True)

            _rewind_stdout(len(self.stages))

    def launch(self, state: fs.State) -> fs.State:
        """
        Executes a launch sequence.
        In less punny terms, this method is called by the top-level
        build_model() function to iterate over all of the Stages required for a build.
        Builds are defined by self.stages in a top-level Sequence, and self.stages
        can include both Stages and Sequences (ie, sequences can be nested).
        """

        if state.build_status == build.FunctionStatus.SUCCESSFUL:
            msg = """
            build_model() is running a build on a model that already built successfully, which
            should not happen because the build should have loaded from cache or rebuilt from scratch.
            If you are using custom Stages and Sequences then you have some debugging to do. Otherwise,
            please file an issue at https://github.com/onnx/turnkeyml/issues
            """
            raise exp.Error(msg)

        # Collect telemetry for the build
        state.save_stat(
            fs.Keys.SELECTED_SEQUENCE_OF_STAGES,
            self.stage_names,
        )

        # At the beginning of a sequence no stage has started
        for stage in self.stages:
            state.save_stat(stage.status_key, build.FunctionStatus.NOT_STARTED)
            state.save_stat(stage.duration_key, "-")

        # Run the build
        for stage, argv in self.stages.items():
            start_time = time.time()

            try:

                # Set status as incomplete, since stage just started
                state.save_stat(stage.status_key, build.FunctionStatus.INCOMPLETE)

                # Collect telemetry about the stage
                state.current_build_stage = stage.unique_name

                # Run the stage
                state = stage.parse_and_fire(state, argv)

                # Save the state so that it can be assessed for a cache hit
                state.save()

            # Broad exception is desirable as we want to capture
            # all exceptions (including those we can't anticipate)
            except Exception as e:  # pylint: disable=broad-except

                if os.environ.get("TURNKEY_DEBUG") == "True":
                    # It may be useful to raise the exception here, since
                    # if any of the subsequent lines of code raise another
                    # exception it will be very hard to root cause e.
                    raise e

                # Update Stage and build status
                state.save_stat(stage.status_key, build.FunctionStatus.ERROR)
                state.save_stat(fs.Keys.BUILD_STATUS, build.FunctionStatus.ERROR)

                # Save the log file for the failed stage to stats for easy reference
                stats = fs.Stats(state.cache_dir, state.build_name)
                stats.save_eval_error_log(stage.logfile_path)

                # Advance the cursor below the monitor so
                # we can print an error message
                stage_depth_in_sequence = len(
                    self.stage_names
                ) - self.stage_names.index(
                    stage.unique_name  # pylint: disable=undefined-loop-variable
                )
                stdout_lines_to_advance = stage_depth_in_sequence - 2
                cursor_down = "\n" * stdout_lines_to_advance

                print(cursor_down)

                if vars(state).get("invocation_info"):
                    state.invocation_info.status_message = f"Error: {e}."
                    state.invocation_info.status_message_color = printing.Colors.WARNING

                    _store_traceback(state.invocation_info)
                else:
                    printing.log_error(e)

            else:
                # Update Stage Status
                state.save_stat(stage.status_key, build.FunctionStatus.SUCCESSFUL)

                if vars(state).get("models_found") and vars(state).get(
                    "invocation_info"
                ):
                    state.invocation_info.status_message = (
                        "Stages successfully executed!"
                    )
                    state.invocation_info.status_message_color = printing.Colors.OKGREEN

            finally:
                # Store stage duration
                execution_time = time.time() - start_time
                state.save_stat(stage.duration_key, execution_time)

        if vars(state).get("models_found") and vars(state).get("invocation_info"):

            # Present status statistics from the Stages
            for stage in self.stages:
                state.invocation_info.stats_keys += stage.status_stats

            print()
            status.update(
                state.models_found,
                state.build_name,
                state.invocation_info,
                cache_dir=state.cache_dir,
            )

        if state.lean_cache:
            printing.log_info("Removing build artifacts...")
            fs.clean_output_dir(state.cache_dir, state.build_name)

        state.current_build_stage = None
        state.build_status = build.FunctionStatus.SUCCESSFUL
        state.save()

        return state

    def status_line(self, verbosity):
        """
        Print a status line in the monitor for every Stage in the sequence
        """
        for stage in self.stages:
            stage.status_line(successful=None, verbosity=verbosity)

    @property
    def info(self) -> Dict[str, Dict]:
        """
        Return a dictionary of stage_name:argv for the sequence
        """

        return {
            stage.__class__.unique_name: argv for stage, argv in self.stages.items()
        }
