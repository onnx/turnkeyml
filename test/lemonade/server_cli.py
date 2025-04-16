"""
Usage: python server_cli.py

This will launch the lemonade server and test the CLI.

If you get the `ImportError: cannot import name 'TypeIs' from 'typing_extensions'` error:
    1. pip uninstall typing_extensions
    2. pip install openai
"""

import unittest
import subprocess
import asyncio
import socket
import time
from threading import Thread
import sys
import io
import httpx
from server import kill_process_on_port, PORT
from turnkeyml import __version__ as version_number

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError as e:
    raise ImportError("You must `pip install openai` to run this test", e)


class Testing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """
        Start lemonade server process
        """
        print("\n=== Starting new test ===")

        # Ensure we kill anything using the test port before and after the test
        kill_process_on_port(PORT)
        self.addCleanup(kill_process_on_port, PORT)

    def test_001_version(self):
        result = subprocess.run(
            ["lemonade-server-dev", "--version"], capture_output=True, text=True
        )

        # Check that the stdout ends with the version number (some apps rely on this)
        assert result.stdout.strip().endswith(
            version_number
        ), f"Expected stdout to end with '{version_number}', but got: '{result.stdout}'"

    def test_002_serve_and_status(self):

        # First, ensure we can correctly detect that the server is not running
        result = subprocess.run(
            ["lemonade-server-dev", "status"],
            capture_output=True,
            text=True,
        )
        assert (
            result.stdout == "Server is not running\n"
        ), f"{result.stdout} {result.stderr}"

        # Now, start the server
        NON_DEFAULT_PORT = PORT + 1
        process = subprocess.Popen(
            ["lemonade-server-dev", "serve", "--port", str(NON_DEFAULT_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the server to start by checking the port
        start_time = time.time()
        while True:
            if time.time() - start_time > 60:
                raise TimeoutError("Server failed to start within 60 seconds")
            try:
                conn = socket.create_connection(("localhost", NON_DEFAULT_PORT))
                conn.close()
                break
            except socket.error:
                time.sleep(1)

        # Wait a few other seconds after the port is available
        time.sleep(20)

        # Now, ensure we can correctly detect that the server is running
        result = subprocess.run(
            ["lemonade-server-dev", "status"],
            capture_output=True,
            text=True,
        )
        assert (
            result.stdout == f"Server is running on port {NON_DEFAULT_PORT}\n"
        ), f"Expected stdout to end with '{NON_DEFAULT_PORT}', but got: '{result.stdout}' {result.stderr}"

        # Close the server
        process.terminate()
        process.wait()


if __name__ == "__main__":
    unittest.main()
