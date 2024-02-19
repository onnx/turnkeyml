import os
import dataclasses
import math
import platform
from typing import Dict, Union, List
from turnkeyml.common import printing
import turnkeyml.common.build as build
from turnkeyml.analyze.util import ModelInfo, BasicInfo


def update(
    models_found: Dict[str, ModelInfo],
    build_name: str,
    cache_dir: str,
) -> None:
    """
    Prints all models and submodels found
    """

    if os.environ.get("TURNKEY_DEBUG") != "True":
        if platform.system() != "Windows":
            os.system("clear")
        else:
            os.system("cls")

    printing.logn(
        "\nModels discovered during profiling:\n",
        c=printing.Colors.BOLD,
    )
    recursive_print(models_found, build_name, cache_dir, None, None, [])


def recursive_print(
    models_found: Dict[str, ModelInfo],
    build_name: str,
    cache_dir: str,
    parent_model_hash: Union[str, None] = None,
    parent_invocation_hash: Union[str, None] = None,
    script_names_visited: List[str] = False,
) -> None:
    script_names_visited = []

    for model_hash in models_found.keys():
        model_visited = False
        model_info = models_found[model_hash]
        invocation_idx = 0
        for invocation_hash in model_info.unique_invocations.keys():
            unique_invocation = model_info.unique_invocations[invocation_hash]

            if (
                parent_model_hash == model_info.parent_hash
                and unique_invocation.executed > 0
                and (
                    model_info.unique_invocations[invocation_hash].parent_hash
                    == parent_invocation_hash
                )
            ):
                print_file_name = False
                if model_info.script_name not in script_names_visited:
                    script_names_visited.append(model_info.script_name)
                    if model_info.depth == 0:
                        print_file_name = True

                unique_invocation.print(
                    build_name=build_name,
                    cache_dir=cache_dir,
                    print_file_name=print_file_name,
                    invocation_idx=invocation_idx,
                    model_visited=model_visited,
                    multiple_unique_invocations=len(model_info.unique_invocations) > 1,
                )
                model_visited = True
                invocation_idx += 1

                if print_file_name:
                    script_names_visited.append(model_info.script_name)

                recursive_print(
                    models_found,
                    build_name,
                    cache_dir,
                    parent_model_hash=model_hash,
                    parent_invocation_hash=invocation_hash,
                    script_names_visited=script_names_visited,
                )
