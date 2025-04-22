# PEEL: PowerShell Enhanced with Enhanced Lemonade (LLM) Functionality

**Overview:**

PEEL is a Windows Terminal application that extends the capabilities of PowerShell by integrating with Lemonade Server. It offers seamless access to LLM-powered assistance directly within the terminal, providing contextual help based on the terminal's scrollback.

**Key Features:**

1.  **PEEL Shell in Windows Terminal:**
    *   PEEL will register itself as a new shell option within Windows Terminal, similar to how "Windows PowerShell", "Command Prompt", and "Azure Cloud Shell" appear in the new tab dropdown.
    *   It should have a distinctive icon (the one provided) to identify it within the Windows Terminal UI.
    * The default behavior of the shell should be the same as powershell.
    *   **Icon**: The PEEL icon will be `turnkeyml/img/favicon.ico`.
    * **API Version**: PEEL should use the v1 API of lemonade server

2.  **`Lemonade-Install` Cmdlet:**
    *   **Functionality:** Downloads and executes the latest Lemonade Server Installer (`Lemonade_Server_Installer.exe`) from the official TurnkeyML GitHub releases page: `https://github.com/onnx/turnkeyml/releases/latest/download/Lemonade_Server_Installer.exe`.
    *   **Behavior:** This cmdlet should be able to handle scenarios where:
        *   The installer is already present.
        *   The installer download fails (e.g., due to network issues).
        * The installer process throws an error
    *   **Outcome:** After successful installation, the `lemonade-server` command should be available in the PEEL shell.
3.  **`Get-Aid` Cmdlet:**
    *   **Functionality:** Captures the terminal's scrollback history, sends it to the Lemonade Server via the streaming chat completions API, and displays the LLM's response in a streaming fashion directly within the terminal.
    *   **Invocation:** It is intended to help get past the error.
    *   **API Interaction:** Uses the streaming chat completions API described in the `server_spec.md`.
    *   **Scrollback:** The `Get-Aid` cmdlet should capture the last N lines of the terminal's scrollback buffer (where N is a configurable number or a reasonable default, like 50). It can do this by checking the user's console's history.
    * **Server**: The `Get-Aid` cmdlet should connect to the server in localhost:8080

    * **Error handling**: if the connection to the server fails or is not running, notify the user.
    *   **LLM Selection:** By default, it will invoke the "standard" LLM model in Lemonade Server.
    *   **Model**: It will invoke Llama-3.2-3B-Instruct-Hybrid
    *   **User Experience:** The output from the LLM should be clearly distinguished from the normal terminal output, perhaps using different colors or formatting.
4.  **`Get-More-Aid` Cmdlet:**
    *   **Functionality:** Similar to `Get-Aid`, but invokes a "larger" LLM model configured in Lemonade Server.
    *   **Model**: It will invoke Qwen-1.5-7B-Chat-Hybrid
    *   **User Experience:** Should have similar formatting as `Get-Aid`.
5.  **`Get-Maximum-Aid` Cmdlet:**
    *   **Functionality:** Similar to `Get-Aid`, but invokes the "largest" available LLM model configured in Lemonade Server.
    *   **Model**: It will invoke DeepSeek-R1-Distill-Qwen-7B-Hybrid
    *   **User Experience:** Should have similar formatting as `Get-Aid`.
6. **Installation:**
    * Users should download the PEEL module (`peel` directory) to their local machine.
    * Then run `install.ps1` from that directory in powershell.
    * After that, PEEL should appear as an option in the Windows Terminal.


**Implementation Details:**




*   **Language:** PowerShell
*   **Location:** New `peel` directory under `examples/lemonade/server/`.
*   **File Structure**
```
examples/lemonade/server/
    └── peel/
        ├── peel.psd1 # Module manifest
        ├── peel.psm1 # PowerShell module implementation
        ├── install.ps1 # Installation script
        └── assets/
            └── peel-icon.png # PEEL icon (provided)
```
**Next Steps:**

I think we should implement this in the following steps.
1. **Set up the Folder Structure:** Create the required file structure under `examples/lemonade/server/peel`.
2. **Create the Powershell Module Manifest:** This file `peel.psd1` will specify the metadata for the module.
3. **Create the Powershell Module:** Create the main module file `peel.psm1`.
4. **Create the Install script:** Create the `install.ps1` script to install the module.
5. **Lemonade-Install cmdlet:** Implement the cmdlet that downloads and run the installer.
6. **Get-Aid cmdlet:** Implement the cmdlet that send the terminal's scrollback.
7. **Get-More-Aid and Get-Maximum-Aid:** Implement this similar to `Get-Aid`.
8. **Register the app in the terminal:** Find the way of adding this app to the windows terminal options.