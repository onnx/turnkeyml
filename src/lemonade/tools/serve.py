import argparse
import asyncio
import statistics
import time
from threading import Thread, Event
import logging
import traceback

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch  # pylint: disable=unused-import
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from tabulate import tabulate

from openai.types.completion import Completion, CompletionChoice
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.model import Model

from turnkeyml.tools.management_tools import ManagementTool
import lemonade.api as lemonade_api
from lemonade_install.install import ModelManager

# Set to a high number to allow for interesting experiences in real apps
# Tests should use the max_new_tokens argument to set a lower value
DEFAULT_MAX_NEW_TOKENS = 1500

DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"

LOCAL_MODELS = ModelManager().downloaded_models_enabled


class GeneratorThread(Thread):
    """
    Thread class designed for use with streaming generation within
    an LLM server. It needs access to the streamer in order to order
    to help the completions APIs escape the "for text in streamer" loop.
    It also provides exception handling that works nicely with HTTP
    servers by providing the stack trace and making the exception
    information available to the main thread.
    """

    def __init__(self, streamer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None
        self.streamer = streamer

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:  # pylint: disable=broad-except
            self.exception = e
            logging.error(f"Exception raised in generate thread: {e}")
            traceback.print_exc()
            self.streamer.done()


# Custom huggingface-style stopping criteria to allow
# us to halt streaming in-progress generations
class StopOnEvent(StoppingCriteria):
    """
    Custom stopping criteria that halts text generation when a specified event is set.

    This allows for external control of generation, such as stopping a generation
    before it reaches the maximum token limit.
    """

    def __init__(self, stop_event: Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


class LoadConfig(BaseModel):
    """
    Configuration for loading a language model.

    Specifies the model checkpoint, generation parameters,
    and hardware/framework configuration (recipe) for model loading.
    """

    checkpoint: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    recipe: str = "hf-cpu"
    # Indicates the maximum prompt length allowed for that specific
    # checkpoint + recipe combination
    max_length: int = None


class CompletionRequest(BaseModel):
    """
    Request model for text completion API endpoint.

    Contains a prompt, a model identifier, and a streaming
    flag to control response delivery.
    """

    prompt: str
    model: str
    echo: bool = False
    stream: bool = False
    stop: list[str] | str | None = None


class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completion API endpoint.

    Contains a list of chat messages, a model identifier,
    and a streaming flag to control response delivery.
    """

    messages: list[dict]
    model: str
    stream: bool = False
    stop: list[str] | str | None = None


class Server(ManagementTool):
    """
    Open a web server that apps can use to communicate with the LLM.

    The server exposes these endpoints:
    - /api/v0/load: load a model checkpoint
    - /api/v0/unload: unload a model checkpoint
    - /api/v0/health: check whether a model is loaded and ready to serve.
    - /api/v0/stats: performance statistics for the generation.
    - /api/v0/halt: stop an in-progress generation from make more tokens.
    - /api/v0/completions: completion responses using HTTP chunked transfer encoding.
    - /api/v0/chat/completions: chat completion responses using HTTP chunked transfer encoding.
    - /api/v0/models: list all available models.
    """

    unique_name = "serve"

    def __init__(self):
        super().__init__()

        # Initialize FastAPI app
        self.app = FastAPI()

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Set up custom routes
        self.app.post("/api/v0/load")(self.load_llm)
        self.app.post("/api/v0/unload")(self.unload_llm)
        self.app.get("/api/v0/health")(self.health)
        self.app.get("/api/v0/halt")(self.halt_generation)
        self.app.get("/api/v0/stats")(self.send_stats)
        self.app.post("/api/v0/completions")(self.completions)

        # Set up OpenAI-compatible routes
        self.app.post("/api/v0/chat/completions")(self.chat_completions)
        self.app.post("/api/v0/completions")(self.completions)
        self.app.get("/api/v0/models")(self.models)

        # Performance stats that are set during /ws and can be
        # fetched in /stats
        self.time_to_first_token = None
        self.tokens_per_second = None
        self.input_tokens = None
        self.output_tokens = None
        self.decode_token_times = None

        # Store debug logging state
        self.debug_logging_enabled = logging.getLogger().isEnabledFor(logging.DEBUG)

        # Flag that tells the LLM to stop generating text and end the response
        self.stop_event = Event()

        self.llm_loaded: LoadConfig = None
        self.tokenizer = None

        # Placeholders for model and configs
        self.model = None
        self.max_new_tokens = None

        # Initialize semaphore for tracking active generations
        self.max_concurrent_generations = 1
        self._generate_semaphore = asyncio.Semaphore(self.max_concurrent_generations)

        # Curated list of "Instruct" and "Chat" models.
        self.local_models = LOCAL_MODELS

        # Add lock for load/unload operations
        self._load_lock = asyncio.Lock()

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Launch an industry-standard LLM server",
            add_help=add_help,
        )

        parser.add_argument(
            "--checkpoint",
            required=False,
            type=str,
            help="Name of the model checkpoint to load (optional)",
        )
        parser.add_argument(
            "--max-new-tokens",
            required=False,
            type=int,
            default=DEFAULT_MAX_NEW_TOKENS,
            help=f"Number of new tokens the LLM should make (default: {DEFAULT_MAX_NEW_TOKENS})",
        )
        parser.add_argument(
            "--port",
            required=False,
            type=int,
            default=DEFAULT_PORT,
            help=f"Port number to run the server on (default: {DEFAULT_PORT})",
        )
        parser.add_argument(
            "--log-level",
            required=False,
            type=str,
            default=DEFAULT_LOG_LEVEL,
            choices=["critical", "error", "warning", "info", "debug", "trace"],
            help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
        )

        return parser

    def run(
        self,
        # ManagementTool has a required cache_dir arg, but
        # we always use the default cache directory
        _=None,
        checkpoint: str = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        port: int = DEFAULT_PORT,
        log_level: str = DEFAULT_LOG_LEVEL,
    ):

        # Define TRACE level
        logging.TRACE = 9  # Lower than DEBUG which is 10
        logging.addLevelName(logging.TRACE, "TRACE")

        # Add a convenience function at the module level
        def trace(message, *args, **kwargs):
            logging.log(logging.TRACE, message, *args, **kwargs)

        logging.trace = trace

        # Configure logging to match uvicorn's format
        logging_level = getattr(logging, log_level.upper())
        logging.basicConfig(
            level=logging_level,
            format="%(levelprefix)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add uvicorn's log formatter
        logging.root.handlers[0].formatter = uvicorn.logging.DefaultFormatter(
            fmt="%(levelprefix)s %(message)s",
            use_colors=True,
        )

        # Ensure the log level is properly set
        logging.getLogger().setLevel(logging_level)

        # Update debug logging state after setting log level
        self.debug_logging_enabled = logging.getLogger().isEnabledFor(logging.DEBUG)

        if self.debug_logging_enabled:
            # Print the elapsed time for each request
            self.setup_middleware_timer()

        # Only load the model when starting the server if checkpoint was provided
        if checkpoint:
            config = LoadConfig(
                checkpoint=checkpoint,
                max_new_tokens=max_new_tokens,
            )
            asyncio.run(self.load_llm(config))

        uvicorn.run(self.app, host="localhost", port=port, log_level=log_level)

    async def _show_telemetry(self):
        """
        Show telemetry data in debug mode.
        """
        # Exit early if debug logging is disabled or no telemetry data is available
        if not self.debug_logging_enabled or self.tokens_per_second is None:
            return

        # Prepare telemetry data (transposed format)
        telemetry = [
            ["Input tokens", self.input_tokens],
            ["Output tokens", self.output_tokens],
            ["TTFT (s)", f"{self.time_to_first_token:.2f}"],
            ["TPS", f"{self.tokens_per_second:.2f}"],
        ]

        table = tabulate(
            telemetry, headers=["Metric", "Value"], tablefmt="fancy_grid"
        ).split("\n")

        # Show telemetry in debug while complying with uvicorn's log indentation
        logging.debug("\n          ".join(table))

    async def completions(self, completion_request: CompletionRequest):
        """
        Stream completion responses using HTTP chunked transfer encoding.
        """
        if completion_request.model:

            # Get model config
            if completion_request.model in self.local_models:
                model_config = self.local_models[completion_request.model]
                lc = LoadConfig(**model_config)
            else:
                # If the model is not built-in, we assume it corresponds to a checkpoint
                lc = LoadConfig(checkpoint=completion_request.model)

        # Load the model if it's different from the currently loaded one
        await self.load_llm(lc)

        # Check if the model supports reasoning
        reasoning_first_token = self.local_models[completion_request.model]["reasoning"]

        # If the model supports reasoning, we:
        # 1. add a <think> tag to the model's context
        # 2. ensure that the first token is a <think> token
        text = completion_request.prompt
        if reasoning_first_token:
            text += "<think>"

        if completion_request.stream:

            if completion_request.echo:
                logging.warning(
                    "`Echo` parameter is not supported for streaming completions"
                )

            # Stream the response
            async def generate():
                # Declare it's the same variable from outside scope
                # This is necessary because the variable is modified
                # in the inner function
                nonlocal reasoning_first_token

                async for token in self._generate_tokens(text, completion_request.stop):
                    choice = CompletionChoice(
                        text=("<think>" + token if reasoning_first_token else token),
                        index=0,
                        finish_reason="stop",
                        logprobs=None,
                    )

                    completion = Completion(
                        id="0",
                        choices=[choice],
                        model=self.llm_loaded.checkpoint,
                        object="text_completion",
                        created=int(time.time()),
                    )

                    # Format as SSE
                    reasoning_first_token = False
                    yield f"data: {completion.model_dump_json()}\n\n".encode("utf-8")

                # Send the [DONE] marker
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # If streaming is not requested, collect all generated tokens into a single response
        else:
            full_response = text if completion_request.echo else ""
            async for token in self._generate_tokens(text, completion_request.stop):
                full_response += token

            choice = CompletionChoice(
                text=full_response,
                index=0,
                finish_reason="stop",
                logprobs=None,
            )

            return Completion(
                id="0",
                choices=[choice],
                model=self.llm_loaded.checkpoint,
                object="text_completion",
                created=int(time.time()),
            )

    async def chat_completions(self, chat_completion_request: ChatCompletionRequest):
        """
        Stream chat completion responses using HTTP chunked transfer encoding.
        """

        # Get model config
        if chat_completion_request.model in self.local_models:
            model_config = self.local_models[chat_completion_request.model]
            lc = LoadConfig(**model_config)
        else:
            # If the model is not built-in, we assume it corresponds to a checkpoint
            lc = LoadConfig(checkpoint=chat_completion_request.model)

        # Load the model if it's different from the currently loaded one
        await self.load_llm(lc)

        # Convert chat messages to text using the model's chat template
        if self.tokenizer.chat_template:
            # Use the model's built-in chat template if available
            messages_dict = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in chat_completion_request.messages
            ]
            text = self.tokenizer.apply_chat_template(
                messages_dict, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback to a standardized template if the model doesn't provide one
            logging.warning("No chat template found. Using default template.")
            formatted_messages = []
            for msg in chat_completion_request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                role_marker = "<|assistant|>" if role == "assistant" else "<|user|>"
                formatted_messages.append(f"{role_marker}\n{content} <|end|>")
            text = "\n".join(formatted_messages) + "\n<|assistant|>"

        # If the model supports reasoning, we:
        # 1. add a <think> tag to the model's context
        # 2. ensure that the first token is a <think> token
        reasoning_first_token = self.local_models[chat_completion_request.model][
            "reasoning"
        ]
        if reasoning_first_token:
            text += "<think>"

        if chat_completion_request.stream:

            # Stream the response
            async def generate():
                # Declare it's the same variable from outside scope
                # This is necessary because the variable is modified
                # in the inner function
                nonlocal reasoning_first_token

                async for token in self._generate_tokens(
                    text, chat_completion_request.stop
                ):

                    # Create a ChatCompletionChunk
                    chunk = ChatCompletionChunk.model_construct(
                        id="0",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=self.llm_loaded.checkpoint,
                        choices=[
                            Choice.model_construct(
                                index=0,
                                delta=ChoiceDelta(
                                    content=(
                                        "<think>" + token
                                        if reasoning_first_token
                                        else token
                                    ),
                                    function_call=None,
                                    role="assistant",
                                    tool_calls=None,
                                    refusal=None,
                                ),
                                finish_reason=None,
                                logprobs=None,
                            )
                        ],
                    )

                    # Format as SSE
                    reasoning_first_token = False
                    yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")

                # Send the [DONE] marker
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # If streaming is not requested, collect all generated tokens into a single response
        else:
            full_response = "<think>" if reasoning_first_token else ""
            async for token in self._generate_tokens(
                text, chat_completion_request.stop
            ):
                full_response += token

            ccm = ChatCompletionMessage(
                content=full_response,
                role="assistant",
                refusal=None,
                audio=None,
                function_call=None,
                tool_calls=None,
            )

            choice = Choice(
                finish_reason="stop",
                index=0,
                message=ccm,
                logprobs=None,
            )

            return ChatCompletion(
                id="0",
                choices=[choice],
                model=self.llm_loaded.checkpoint,
                object="chat.completion",
                created=int(time.time()),
            )

    async def _generate_tokens(self, message: str, stop: list[str] | str | None = None):
        """
        Core streaming completion logic, separated from response handling.
        Returns an async generator that yields tokens.
        """
        model = self.model
        tokenizer = self.tokenizer

        # Reset the early-exit flag before we start each generation
        self.stop_event.clear()

        input_ids = tokenizer(message, return_tensors="pt").input_ids

        # Process stop sequences
        stop_sequences = []
        if stop is not None:
            if isinstance(stop, str):
                stop_sequences = [stop]
            else:
                stop_sequences = stop[:4]  # Limit to 4 sequences as per spec

        # Set up the generation parameters
        if "oga-" in self.llm_loaded.recipe:
            from lemonade.tools.ort_genai.oga import OrtGenaiStreamer

            streamer = OrtGenaiStreamer(tokenizer)
            self.input_tokens = len(input_ids)
        else:
            # Huggingface-like models
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
            )
            self.input_tokens = len(input_ids[0])

        if (
            self.llm_loaded.max_length
            and self.input_tokens > self.llm_loaded.max_length
        ):
            # This is the exact same exception message raised by OGA when max_length is exceeded
            raise RuntimeError(
                f"prompt tokens ({self.input_tokens}) cannot be greater "
                f"than model context_length ({self.llm_loaded.max_length})"
            )

        # Log the input tokens early to avoid this not showing due to potential crashes
        logging.debug(f"Input Tokens: {self.input_tokens}")
        logging.trace(f"Input Message: {message}")

        stopping_criteria = StoppingCriteriaList([StopOnEvent(self.stop_event)])

        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 1,
            "pad_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }

        # Initialize performance variables
        generation_start_time = time.perf_counter()
        first_token = True
        self.decode_token_times = []
        self.output_tokens = 0

        # Begin generation
        thread = GeneratorThread(
            streamer, target=model.generate, kwargs=generation_kwargs
        )
        thread.start()

        # Acquire the generation semaphore
        await self._generate_semaphore.acquire()
        active_generations = (
            self.max_concurrent_generations
            - self._generate_semaphore._value  # pylint: disable=protected-access
        )

        logging.debug(f"Active generations: {active_generations}")

        try:
            # Generate the response using streaming
            new_text = ""
            for new_text in streamer:
                # Yield control back to the event loop
                # This gives the FastAPI server a chance to send the chunks to the client
                await asyncio.sleep(0)

                # Capture performance stats about this token
                self.output_tokens = self.output_tokens + 1
                if first_token:
                    self.time_to_first_token = (
                        time.perf_counter() - generation_start_time
                    )
                    first_token = False
                else:
                    self.decode_token_times.append(
                        time.perf_counter() - next_token_start_time
                    )
                next_token_start_time = time.perf_counter()

                # Remove the EOS token from the response if needed
                if hasattr(self.tokenizer, "eos_token"):
                    new_text = new_text.replace(self.tokenizer.eos_token, "")

                # Check for stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in new_text:
                            # Make sure we yield the text up to before the stop sequence
                            new_text = new_text[: new_text.find(stop_seq)]
                            self.stop_event.set()

                yield new_text

                # Allow the user to finish the response early
                if self.stop_event.is_set():
                    logging.info("Stopping generation early.")
                    break

            self.tokens_per_second = 1 / statistics.mean(self.decode_token_times)

        finally:
            thread.join()

            # Release the semaphore when generation is complete (or if an error occurs)
            self._generate_semaphore.release()
            active_generations = (
                self.max_concurrent_generations
                - self._generate_semaphore._value  # pylint: disable=protected-access
            )

            # Check if an exception occurred in the generation thread
            # If it did, raise it as an HTTPException so that the client
            # knows they wont be getting a completion
            if thread.exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Completion failure {thread.exception}",
                )

            # Display telemetry if in debug mode
            await self._show_telemetry()

    async def send_stats(self):
        """
        Send performance statistics to the client.
        """
        return {
            "time_to_first_token": self.time_to_first_token,
            "tokens_per_second": self.tokens_per_second,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "decode_token_times": self.decode_token_times,
        }

    async def halt_generation(self):
        """
        Allow the client to halt an in-progress generation.
        """

        self.stop_event.set()

        return {
            "terminated": True,
        }

    async def health(self):
        """
        Report server health information to the client.
        """
        self.stop_event.set()

        return {
            "status": "ok",
            "model_loaded": (self.llm_loaded.checkpoint if self.llm_loaded else None),
        }

    async def load_llm(self, config: LoadConfig):
        try:
            await self._load_lock.acquire()

            # Acquire all generate locks
            for _ in range(self.max_concurrent_generations):
                await self._generate_semaphore.acquire()

            if self.llm_loaded and config.checkpoint == self.llm_loaded.checkpoint:
                return {
                    "status": "success",
                    "message": f"Model already loaded: {config.checkpoint}",
                }

            # Unload the current model if needed
            if self.llm_loaded:
                await self.unload_llm(require_lock=False)

            self.max_new_tokens = config.max_new_tokens
            logging.info(f"Loading llm: {config.checkpoint}")
            try:
                self.model, self.tokenizer = lemonade_api.from_pretrained(
                    checkpoint=config.checkpoint, recipe=config.recipe
                )

                self.max_new_tokens = config.max_new_tokens
                self.llm_loaded = config

                return {
                    "status": "success",
                    "message": f"Loaded model: {config.checkpoint}",
                }
            except Exception:  # pylint: disable=broad-exception-caught
                self.llm_loaded = None
                self.tokenizer = None
                self.model = None
                logging.exception(f"Tried to load LLM {config.checkpoint} and failed")

                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"model {config.checkpoint} not found",
                )

        finally:
            self._load_lock.release()

            # Release all generate locks
            for _ in range(self.max_concurrent_generations):
                self._generate_semaphore.release()

    async def unload_llm(self, require_lock: bool = True):
        try:
            if require_lock:
                await self._load_lock.acquire()

                # Acquire all generate locks
                for _ in range(self.max_concurrent_generations):
                    await self._generate_semaphore.acquire()

            self.llm_loaded = None
            self.tokenizer = None
            self.model = None
            return {"status": "success", "message": "Unloaded model"}
        except Exception as e:  # pylint: disable=broad-exception-caught
            return {
                "status": "error",
                "message": f"Failed to unload model: {str(e)}",
            }
        finally:
            if require_lock:
                self._load_lock.release()

                # Release all generate locks
                for _ in range(self.max_concurrent_generations):
                    self._generate_semaphore.release()

    async def models(self):
        """
        Return a list of available models in OpenAI-compatible format.
        """
        models_list = []
        for model in self.local_models:
            m = Model(
                id=model,
                owned_by="lemonade",
                object="model",
                created=int(time.time()),
            )
            models_list.append(m)

        return {"object": "list", "data": models_list}

    def setup_middleware_timer(self):
        logging.info("Middleware set up")

        @self.app.middleware("http")
        async def log_request_time(request: Request, call_next):
            """
            Log the request processing time for any request
            """

            start_time = time.perf_counter()
            response = await call_next(request)
            request_time = time.perf_counter() - start_time
            logging.debug(f"Total request time: {request_time:.4f} seconds")
            return response
