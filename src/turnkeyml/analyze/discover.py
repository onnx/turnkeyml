import argparse
import copy
import os
import inspect
from typing import Optional, List
import torch
import turnkeyml.build.stage as stage
import turnkeyml.common.exceptions as exp
import turnkeyml.common.filesystem as fs
from turnkeyml.analyze.script import (
    evaluate_script,
    TracerArgs,
)
import turnkeyml.common.printing as printing


default_max_depth = 0


class Discover(stage.Stage):
    """
    Discover the PyTorch models and their corresponding inputs in a python script (.py)
    and send one model/inputs pair onwards into the sequence.

    Expected inputs: None

    Outputs:
        state.results: a PyTorch model instance (torch.nn.Module)
        state.inputs: a dictionary of example inputs to the model's forward function,
            e.g., model(**inputs)
    """

    unique_name = "discover"

    def __init__(self):
        super().__init__(monitor_message="Discovering PyTorch models")

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Discover the PyTorch models in a python script",
            add_help=add_help,
        )

        parser.add_argument(
            "--script-args",
            dest="script_args",
            type=str,
            help="Arguments to pass into the target script(s)",
        )

        parser.add_argument(
            "--max-depth",
            dest="max_depth",
            type=int,
            default=default_max_depth,
            help="Maximum depth to analyze within the model structure of the target script(s)",
        )

        # Hidden argument required by TurnkeyML for any stage that starts a sequence
        parser.add_argument("--input", help=argparse.SUPPRESS)

        return parser

    def parse(self, state: fs.State, args, known_only=True) -> argparse.Namespace:
        parsed_args = super().parse(state, args, known_only)

        file_path, targets, _ = fs.decode_input_arg(parsed_args.input)

        parsed_args.input = file_path
        parsed_args.targets = targets

        return parsed_args

    def fire(
        self,
        state: fs.State,
        input: str = "",
        targets: Optional[List[str]] = None,
        script_args: str = "",
        max_depth: int = default_max_depth,
    ):
        if not input.endswith(".py"):
            raise exp.ArgError(
                "Inputs to the `discover` stage must by python scripts "
                f"(.py files), got {input}",
            )

        if targets is None:
            targets_to_use = []
        else:
            targets_to_use = targets

        tracer_args = TracerArgs(
            input=input,
            script_args=script_args,
            targets=targets_to_use,
            max_depth=max_depth,
        )

        # Discover the models in the python script by executing it with
        # a tracer enabled
        models_found = evaluate_script(tracer_args)

        # Count the amount of build-able model invocations discovered
        # If there is only 1, pass it to the next build stage. Otherwise,
        # print all the invocations and suggest that the user select one.
        count = 0
        for model_info in models_found.values():
            for (
                invocation_hash,
                invocation_info,
            ) in model_info.unique_invocations.items():
                count += 1

                # Set the same status for all invocations at first
                # The next code block will be responsible for the selected
                # invocation.

                invocation_info.status_message = (
                    "Discovered; select with `-i "
                    f"{os.path.basename(model_info.file)}::{invocation_hash}"
                )
                invocation_info.status_message_color = printing.Colors.OKCYAN

        for model_info in models_found.values():
            for invocation_info in model_info.unique_invocations.values():
                if invocation_info.is_target or (
                    len(targets_to_use) == 0 and count > 1
                ):
                    if len(targets_to_use) == 0 and count > 1:
                        invocation_info.auto_selected = True

                    # Save stats about the model
                    state.save_stat(
                        fs.Keys.HASH,
                        model_info.hash,
                    )
                    state.save_stat(
                        fs.Keys.MODEL_NAME,
                        tracer_args.script_name,
                    )
                    state.save_stat(
                        fs.Keys.PARAMETERS,
                        model_info.params,
                    )

                    state.save_stat(
                        fs.Keys.CLASS,
                        type(model_info.model).__name__,
                    )

                    state.results = model_info.model
                    # FIXME: we should be able to get rid of state.model now
                    state.model = model_info.model

                    # Organize the inputs to python model instances
                    args, kwargs = invocation_info.inputs
                    inputs = {}
                    for k in kwargs.keys():
                        if torch.is_tensor(kwargs[k]):
                            inputs[k] = torch.tensor(kwargs[k].detach().numpy())
                        else:
                            inputs[k] = copy.deepcopy(kwargs[k])

                    # Convert all positional arguments into keyword arguments
                    if args != ():

                        forward_function = model_info.model.forward
                        all_args = list(
                            inspect.signature(forward_function).parameters.keys()
                        )
                        for i in range(len(args)):
                            if torch.is_tensor(args[i]):
                                inputs[all_args[i]] = torch.tensor(
                                    args[i].detach().numpy()
                                )
                            else:
                                inputs[all_args[i]] = args[i]
                    state.inputs = inputs
                    state.invocation_info = invocation_info
                    state.models_found = models_found

                    return state

        return state
