import argparse
import os
import time
import statistics
from threading import Thread, Event
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import uvicorn
import matplotlib.pyplot as plt
import turnkeyml.common.build as build
from turnkeyml.state import State
from turnkeyml.tools import Tool
from lemonade.tools.adapter import ModelAdapter, TokenizerAdapter
from lemonade.cache import Keys

DEFAULT_GENERATE_PARAMS = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.7,
}

DEFAULT_SERVER_PORT = 8000
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_N_TRIALS = 1

END_OF_STREAM = "</s>"


def sanitize_string(input_string):
    return input_string.encode("charmap", "ignore").decode("charmap")


def sanitize_text(text):
    if isinstance(text, str):
        return sanitize_string(text)
    elif isinstance(text, list):
        return [sanitize_string(item) for item in text]
    else:
        raise TypeError("Input must be a string or a list of strings.")


def positive_int(x):
    """Conversion function for argparse"""
    i = int(x)
    if i < 1:
        raise ValueError("Non-positive values are not allowed")
    return i


class LLMPrompt(Tool):
    """
    Send a prompt to an LLM instance and print the response to the screen.

    Required input state:
        - state.model: LLM instance that supports the generate() method.
        - state.tokenizer: LLM tokenizer instance that supports the __call__() (ie, encode)
            and decode() methods.

    Output state produced:
        - "response": text response from the LLM.
    """

    unique_name = "llm-prompt"

    def __init__(self):
        super().__init__(monitor_message="Prompting LLM")

        self.status_stats = [
            Keys.PROMPT_TOKENS,
            Keys.PROMPT,
            Keys.RESPONSE_TOKENS,
            Keys.RESPONSE,
            Keys.RESPONSE_LENGTHS_HISTOGRAM,
        ]

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Prompt an LLM and print the result",
            add_help=add_help,
        )

        parser.add_argument(
            "--prompt",
            "-p",
            help="Input prompt to the LLM. Two formats are supported. "
            "1) str: use a user-provided prompt string "
            "2) path/to/prompt.txt: load the prompt from a .txt file.",
            required=True,
        )

        parser.add_argument(
            "--max-new-tokens",
            "-m",
            default=DEFAULT_MAX_NEW_TOKENS,
            type=int,
            help=f"Maximum number of new tokens in the response "
            f"(default is {DEFAULT_MAX_NEW_TOKENS})",
        )

        parser.add_argument(
            "--n-trials",
            "-n",
            default=DEFAULT_N_TRIALS,
            type=positive_int,
            help=f"Number of responses the LLM will generate for the prompt "
            f"(useful for testing, default is {DEFAULT_N_TRIALS})",
        )

        return parser

    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        """
        Helper function to parse CLI arguments into the args expected
        by run()
        """

        parsed_args = super().parse(state, args, known_only)

        # Decode prompt arg into a string prompt
        if parsed_args.prompt.endswith(".txt") and os.path.exists(parsed_args.prompt):
            with open(parsed_args.prompt, "r", encoding="utf-8") as f:
                parsed_args.prompt = f.read()
        else:
            # No change to the prompt
            pass

        return parsed_args

    def run(
        self,
        state: State,
        prompt: str = "Hello",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        n_trials: int = DEFAULT_N_TRIALS,
    ) -> State:

        model: ModelAdapter = state.model
        tokenizer: TokenizerAdapter = state.tokenizer

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        if isinstance(input_ids, (list, str)):
            # OGA models return a list of tokens
            # Our llama.cpp adapter returns a string
            len_tokens_in = len(input_ids)
        else:
            # HF models return a 2-D tensor
            len_tokens_in = input_ids.shape[1]

        len_tokens_out = []
        response_texts = []
        for trial in range(n_trials):
            if n_trials > 1:
                self.set_percent_progress(100.0 * trial / n_trials)

            # Get the response from the LLM, which may include the prompt in it
            response = model.generate(
                input_ids, max_new_tokens=max_new_tokens, **DEFAULT_GENERATE_PARAMS
            )

            # Flatten the input and response
            input_ids_array = (
                input_ids if isinstance(input_ids, (list, str)) else input_ids[0]
            )
            response_array = response if isinstance(response, str) else response[0]

            # Separate the prompt from the response
            len_tokens_out.append(len(response_array) - len_tokens_in)

            input_token = 0
            while (
                input_token < len_tokens_in
                and input_ids_array[input_token] == response_array[input_token]
            ):
                input_token += 1

            # Only decode the actual response (not the prompt)
            response_text = tokenizer.decode(
                response_array[input_token:], skip_special_tokens=True
            ).strip()
            response_texts.append(response_text)

        state.response = response_texts

        if n_trials == 1:
            len_tokens_out = len_tokens_out[0]
            response_texts = response_texts[0]
        else:
            self.set_percent_progress(None)

            # Plot data
            plt.figure()
            plt.hist(len_tokens_out, bins=20)
            plt.xlabel("Response Length (tokens)")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Response Lengths\n{state.build_name}")
            figure_path = os.path.join(
                build.output_dir(state.cache_dir, state.build_name),
                "response_lengths.png",
            )
            plt.savefig(figure_path)
            state.save_stat(Keys.RESPONSE_LENGTHS_HISTOGRAM, figure_path)

        state.save_stat(Keys.PROMPT_TOKENS, len_tokens_in)
        state.save_stat(Keys.PROMPT, prompt)
        state.save_stat(Keys.RESPONSE_TOKENS, len_tokens_out)
        state.save_stat(Keys.RESPONSE, sanitize_text(response_texts))

        return state


# Custom huggingface-style stopping criteria to allow
# us to halt streaming in-progress generations
class StopOnEvent(StoppingCriteria):
    def __init__(self, stop_event: Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


class Serve(Tool):
    """
    Open a web server that apps can use to communicate with the LLM.

    There are two ways to perform generations with the server:
    - Send an http request to "http://localhost:8000/generate" and
        receive back a response with the complete prompt.
    - Open a WebSocket with "ws://localhost:8000" and receive a
        streaming response to the prompt.

    The server also exposes these helpful endpoints:
    - /health: check whether a model is loaded and ready to serve.
    - /stats: performance statistics for the generation.
    - /halt: stop an in-progress generation from make more tokens.

    The WebSocket functionality is demonstrated by the webpage served at
    http://localhost:8000, which you can visit with a web browser after
    opening the server.

    Required input state:
        - state.model: model instance serve. Must be compatible with the
            huggingface TextIteratorStreamer.
        - state.tokenizer: tokenizer instance used to generate inputs for the
            model. Must be compatible with the huggingface TextIteratorStreamer.
        - state.checkpoint: name of the checkpoint used to load state.model.

    Output state produced: None
    """

    unique_name = "serve"

    def __init__(self):
        # Disable the build logger since the server is interactive
        super().__init__(
            monitor_message="Launching LLM Server",
            enable_logger=False,
        )

        # Performance stats that are set during /ws and can be
        # fetched in /stats
        self.time_to_first_token = None
        self.tokens_per_second = None
        self.input_tokens = None
        self.output_tokens = None
        self.decode_token_times = None

        # Flag that tells the LLM to stop generating text and end the response
        self.stop_event = Event()

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Open an HTTP server for the model",
            add_help=add_help,
        )

        parser.add_argument(
            "--max-new-tokens",
            required=False,
            type=int,
            default=300,
            help="Number of new tokens the LLM should make (default: 300)",
        )

        return parser

    def run(
        self,
        state: State,
        max_new_tokens: int = 300,
    ) -> State:

        # Disable the build monitor since the server is persistent and interactive
        if self.progress:
            self.progress.terminate()
            print("\n")

        app = FastAPI()

        # Load the model and tokenizer
        model = state.model
        tokenizer = state.tokenizer

        class Message(BaseModel):
            text: str

        html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Chat</title>
            </head>
            <body>
                <h1>Lemonade Chat</h1>
                <form action="" onsubmit="sendMessage(event)">
                    <input type="text" id="messageText" autocomplete="off"/>
                    <button type="submit">Send</button>
                </form>
                <button onclick="showStats()">Show Stats</button>
                <button onclick="halt()">Halt</button>
                <button onclick="health()">Health</button>
                <p id="allMessages"></p> <!-- Use a <p> element to display all messages -->
                <p id="statsMessage"></p> <!-- Use a <p> element to display stats message -->
                <script>
                    const messageQueue = []; // Store incoming messages
                    const allMessagesContainer = document.getElementById('allMessages'); // Get the container element
                    const statsMessageContainer = document.getElementById('statsMessage'); // Get the stats message container
                    var ws = new WebSocket("ws://localhost:8000/ws");
                    ws.onmessage = function(event) {
                        const message = event.data;
                        messageQueue.push(message); // Add the received message to the queue
                        displayAllMessages(); // Display all messages
                    };
                    function displayAllMessages() {
                        if (messageQueue.length > 0) {
                            const allMessages = messageQueue.join(' '); // Concatenate all messages
                            allMessagesContainer.textContent = allMessages; // Set the content of the container
                        }
                    }
                    function sendMessage(event) {
                        var input = document.getElementById("messageText")
                        ws.send(input.value)
                        input.value = ''
                        event.preventDefault()
                    }
                    function showStats() {
                        fetch('/stats')
                        .then(response => response.json())
                        .then(data => {
                            statsMessageContainer.textContent = JSON.stringify(data); // Display the stats message
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    }
                    function halt() {
                        fetch('/halt')
                        .then(response => response.json())
                        .then(data => {
                            statsMessageContainer.textContent = JSON.stringify(data); // Display the stats message
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    }
                    function health() {
                        fetch('/health')
                        .then(response => response.json())
                        .then(data => {
                            statsMessageContainer.textContent = JSON.stringify(data); // Display the stats message
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    }
                </script>
            </body>
        </html>
        """

        @app.get("/")
        async def get():
            return HTMLResponse(html)

        @app.post("/generate")
        async def generate_response(message: Message):
            input_ids = tokenizer(message.text, return_tensors="pt").input_ids
            response = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **DEFAULT_GENERATE_PARAMS,
            )
            generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

            # Remove the input prompt from the generated text
            generated_text = generated_text.replace(message.text, "").strip()

            return {"response": generated_text}

        @app.websocket("/ws")
        async def stream_response(websocket: WebSocket):
            """
            Receive a prompt string, and then stream the response back
            over a websocket.
            """

            await websocket.accept()
            while True:

                try:
                    message = await websocket.receive_text()
                except WebSocketDisconnect:
                    print("Client closed connection")
                    break

                # Reset the early-exit flag before we start each generation
                self.stop_event.clear()

                input_ids = tokenizer(message, return_tensors="pt").input_ids

                # Set up the generation parameters
                if isinstance(model, ModelAdapter) and model.type == "ort-genai":
                    # Onnxruntime-genai models
                    import lemonade.tools.ort_genai.oga as oga

                    streamer = oga.OrtGenaiStreamer(tokenizer)

                    self.input_tokens = len(input_ids)

                else:
                    # Huggingface-like models
                    streamer = TextIteratorStreamer(
                        tokenizer,
                        skip_prompt=True,
                    )

                    self.input_tokens = len(input_ids[0])

                # Enable sending a signal into the generator thread to stop
                # the generation early
                stopping_criteria = StoppingCriteriaList([StopOnEvent(self.stop_event)])

                generation_kwargs = {
                    "input_ids": input_ids,
                    "streamer": streamer,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "stopping_criteria": stopping_criteria,
                    **DEFAULT_GENERATE_PARAMS,
                }

                # Initialize performance variables
                generation_start_time = time.perf_counter()
                first_token = True
                self.decode_token_times = []
                self.output_tokens = 0

                # Begin generation
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                # Generate the response using streaming
                new_text = ""
                for new_text in streamer:

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

                    # Print the decoded value to the terminal for debugging purposes
                    print(new_text, end="", flush=True)

                    # Send the generated text to the client
                    await asyncio.sleep(0.001)  # Add a small delay (adjust as needed)
                    await websocket.send_text(new_text)

                    # Allow the user to finish the response early
                    if self.stop_event.is_set():
                        print("Stopping generation early.")
                        break

                if new_text != END_OF_STREAM:
                    await websocket.send_text(END_OF_STREAM)

                self.tokens_per_second = 1 / statistics.mean(self.decode_token_times)
                print("\n")
                thread.join()

        @app.get("/stats")
        async def send_stats():
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

        @app.get("/halt")
        async def halt_generation():
            """
            Allow the client to halt an in-progress generation.
            """

            self.stop_event.set()

            return {
                "terminated": True,
            }

        @app.get("/health")
        async def health():
            """
            Report server health information to the client.
            """

            self.stop_event.set()

            return {
                "model_loaded": state.checkpoint,
            }

        uvicorn.run(app, host="localhost", port=DEFAULT_SERVER_PORT)

        return state
