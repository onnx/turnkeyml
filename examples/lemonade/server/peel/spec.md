# PEEL: PowerShell Enhanced by Embedded Lemonade (LLM) Functionality

**Overview:**

PEEL is a Windows Terminal application that extends the capabilities of PowerShell by integrating with Lemonade Server. It offers seamless access to LLM-powered assistance directly within the terminal, providing contextual help based on the terminal's scrollback.

**Key Features:**

1.  **PEEL Shell in Windows Terminal:**
    *   PEEL will register itself as a new shell option within Windows Terminal, similar to how "Windows PowerShell", "Command Prompt", and "Azure Cloud Shell" appear in the new tab dropdown.
    *   It should have a distinctive icon (the one provided) to identify it within the Windows Terminal UI.
    *   The default behavior of the shell should be the same as powershell.
    *   **Icon**: The PEEL icon will be `turnkeyml/img/favicon.ico`.
    *   **API Version**: PEEL should use the v1 API of Lemonade Server

2.  **Cmdlets:**
    *   **Install-Lemonade**: Downloads and runs the Lemonade Server installer.
    *   **Get-Aid**: Sends the last 50 lines of terminal history to Lemonade Server using the Llama-3.2-3B-Instruct-Hybrid model.
    *   **Get-MoreAid**: Like Get-Aid, but uses the Qwen-1.5-7B-Chat-Hybrid model.
    *   **Get-MaximumAid**: Like Get-Aid, but uses the DeepSeek-R1-Distill-Qwen-7B-Hybrid model.
    *   All cmdlets stream the LLM's response back to the terminal.
    *   **User Experience:** The output from the LLM should be clearly distinguished from the normal terminal output, perhaps using different colors or formatting.

3. **Installation:**
    * Users should download the PEEL module (`peel` directory) to their local machine.
    * Then run `install.ps1` from that directory in PowerShell.
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

**Cmdlet Details:**

- **Install-Lemonade**: Downloads and executes the latest Lemonade Server Installer from the official TurnkeyML GitHub releases page.
- **Get-Aid**: Captures the last 50 lines of terminal history and sends them to Lemonade Server using the Llama-3.2-3B-Instruct-Hybrid model.
- **Get-MoreAid**: Same as Get-Aid, but uses the Qwen-1.5-7B-Chat-Hybrid model.
- **Get-MaximumAid**: Same as Get-Aid, but uses the DeepSeek-R1-Distill-Qwen-7B-Hybrid model.

All cmdlets stream the LLM's response and display it in the terminal.