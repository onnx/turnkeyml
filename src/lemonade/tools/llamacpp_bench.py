import argparse
import statistics
from statistics import StatisticsError
import tqdm
from turnkeyml.state import State
from lemonade.cache import Keys
from lemonade.tools.llamacpp import LlamaCppAdapter
from lemonade.tools.bench import Bench


class LlamaCppBench(Bench):

    unique_name = "llama-cpp-bench"

    def __init__(self):
        super().__init__()

        # Additional statistics generated by this bench tool
        self.status_stats += [
            Keys.STD_DEV_TOKENS_PER_SECOND,
        ]
        self.std_dev_token_generation_tokens_per_second_list = []

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark a Llamacpp model",
            add_help=add_help,
        )

        parser = Bench.parser(parser)

        return parser

    def run_prompt(
        self,
        state: State,
        report_progress_fn,
        prompt: str,
        iterations: int,
        warmup_iterations: int,
        output_tokens: int,
    ) -> State:
        """
        Benchmark llama.cpp model that was loaded by LoadLlamaCpp.
        """

        if self.first_run_prompt:

            if not hasattr(state, "model") or not isinstance(
                state.model, LlamaCppAdapter
            ):
                raise Exception(
                    f"{self.__class__.unique_name} requires a LlamaCppAdapter model to be "
                    "loaded first. Please run load-llama-cpp before this tool."
                )

        iteration_tokens_per_second = []
        iteration_time_to_first_token = []

        for iteration in tqdm.tqdm(
            range(iterations + warmup_iterations),
            desc="iterations",
            disable=iterations < 2,
        ):
            try:
                # Use the adapter's generate method which already has the timeout
                # and error handling
                raw_output, stderr = state.model.generate(prompt, return_raw=True)

                # Parse the timing information from the output
                ms_per_token = None
                time_to_first_token_ms = None
                input_tokens = None

                # Look for timing in both stdout and stderr
                for output in [raw_output, stderr]:
                    for line in output.splitlines():
                        if "llama_perf_context_print:        eval time =" in line:
                            parts = line.split("(")[1].strip()
                            parts = parts.split(",")
                            ms_per_token = float(
                                parts[0].split("ms per token")[0].strip()
                            )
                        if "llama_perf_context_print: prompt eval time =" in line:
                            parts = line.split("=")[1].split("/")
                            time_to_first_token_ms = float(
                                parts[0].split("ms")[0].strip()
                            )
                            input_tokens = int(parts[1].split("tokens")[0].strip())

                if ms_per_token is None or time_to_first_token_ms is None:
                    error_msg = (
                        "Could not find timing information in llama.cpp output.\n"
                    )
                    error_msg += "Raw output:\n" + raw_output + "\n"
                    error_msg += "Stderr:\n" + stderr
                    raise Exception(error_msg)

                # When output_tokens is set to 1 for accuracy tests, ms_per_token tends to 0
                # and causes a divide-by-zero error. Set tokens_per_second to 0 in such cases
                # as performance data for generating a few tokens is not relevant.
                tokens_per_second = 0
                if output_tokens > 5 and ms_per_token > 0:
                    tokens_per_second = 1000 / ms_per_token
                time_to_first_token = time_to_first_token_ms / 1000

                if iteration > warmup_iterations - 1:
                    iteration_tokens_per_second.append(tokens_per_second)
                    iteration_time_to_first_token.append(time_to_first_token)

                report_progress_fn((iteration + 1) / (warmup_iterations + iterations))

            except Exception as e:
                error_msg = f"Failed to run benchmark: {str(e)}"
                raise Exception(error_msg)

        self.input_ids_len_list.append(input_tokens)
        mean_time_to_first_token = statistics.mean(iteration_time_to_first_token)
        self.mean_time_to_first_token_list.append(mean_time_to_first_token)
        self.prefill_tokens_per_second_list.append(
            input_tokens / mean_time_to_first_token
        )
        self.token_generation_tokens_per_second_list.append(
            statistics.mean(iteration_tokens_per_second)
        )
        try:
            self.std_dev_time_to_first_token_list.append(
                statistics.stdev(iteration_time_to_first_token)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_time_to_first_token_list.append(None)
        try:
            self.std_dev_token_generation_tokens_per_second_list.append(
                statistics.stdev(iteration_tokens_per_second)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_token_generation_tokens_per_second_list.append(None)

    def save_stats(self, state):
        super().save_stats(state)

        # Save additional statistics
        if not all(
            element is None
            for element in self.std_dev_token_generation_tokens_per_second_list
        ):
            state.save_stat(
                Keys.STD_DEV_TOKENS_PER_SECOND,
                self.get_item_or_list(
                    self.std_dev_token_generation_tokens_per_second_list
                ),
            )
