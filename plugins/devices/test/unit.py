"""
Miscellaneous unit tests
"""

import unittest
import os
import turnkeyml.common.build as build
import turnkeyml_plugin_devices.common.run.performance as performance
import turnkeyml_plugin_devices.common.run.plugin_helpers as plugin_helpers


class Testing(unittest.TestCase):

    def test_000_device_class(self):
        family = "family"
        part = "part"
        config = "config"
        device_str = f"{family}::{part}::{config}"
        my_device = performance.Device(device_str)

        assert my_device.family == family
        assert my_device.part == part
        assert my_device.config == config
        assert str(my_device) == device_str

    def test_001_subprocess_logger(self):
        """
        Ensure the subprocess logger stores both stdout and stderr and is also fail-safe
        """

        # Initialize messages and commands used in the test
        logfile_path = "unit_log_subprocess.txt"
        inside_stdout_msg = "This stdout should be inside the log file"
        inside_sterr_msg = "This stderr should be inside the log file"
        outside_stdout_msg = "This stdout should be outside the log file"
        outside_stderr_msg = "This stderr should be outside the log file"
        traceback_error_msg = "Tracebacks should be inside the log file"
        inside_stdout_cmd = f"print('{inside_stdout_msg}',file=sys.stdout)"
        inside_sterr_cmd = f"print('{inside_sterr_msg}',file=sys.stderr)"
        traceback_error_cmd = f"raise ValueError('{traceback_error_msg}')"

        # Perform basic test (no exceptions inside logger)
        cmd = ["python", "-c", f"import sys\n{inside_stdout_cmd}\n{inside_sterr_cmd}"]
        plugin_helpers.logged_subprocess(cmd=cmd, log_file_path=logfile_path)

        # Make sure we captured everything we intended to capture
        with open(logfile_path, "r", encoding="utf-8") as file:
            log_contents = file.read()
        assert inside_stdout_msg in log_contents
        assert inside_sterr_msg in log_contents

        # Perform test with exceptions inside the logger
        cmd = [
            "python",
            "-c",
            f"import sys\n{inside_stdout_cmd}\n{inside_sterr_cmd}\n{traceback_error_cmd}",
        ]
        with self.assertRaises(plugin_helpers.CondaError):
            plugin_helpers.logged_subprocess(cmd=cmd, log_file_path=logfile_path)

        # Make sure we captured everything we intended to capture
        with open(logfile_path, "r", encoding="utf-8") as file:
            log_contents = file.read()
        assert inside_stdout_msg in log_contents
        assert inside_sterr_msg in log_contents
        assert traceback_error_msg in log_contents

        # Ensure subprocess correctly receives the environment
        subprocess_env = os.environ.copy()
        expected_env_var_value = "Expected Value"
        subprocess_env["TEST_ENV_VAR"] = expected_env_var_value
        cmd = ["python", "-c", f'import os\nprint(os.environ["TEST_ENV_VAR"])']
        plugin_helpers.logged_subprocess(
            cmd=cmd, log_file_path=logfile_path, env=subprocess_env
        )
        with open(logfile_path, "r", encoding="utf-8") as file:
            log_contents = file.read()
        assert expected_env_var_value in log_contents

        # Test log_to_std_streams
        cmd = [
            "python",
            "-c",
            f'print("{outside_stdout_msg}")\nprint("{outside_stderr_msg}")',
        ]
        with build.Logger("", logfile_path):
            plugin_helpers.logged_subprocess(
                cmd=cmd, log_to_std_streams=True, log_to_file=False
            )
        with open(logfile_path, "r", encoding="utf-8") as file:
            log_contents = file.read()
        assert outside_stdout_msg in log_contents
        assert outside_stderr_msg in log_contents


if __name__ == "__main__":
    unittest.main()
