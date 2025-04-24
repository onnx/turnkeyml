# PEEL: PowerShell Enhanced by Embedded Lemonade (LLM) Functionality

## Overview

🍋 PEEL is a PowerShell module that extends the capabilities of PowerShell by integrating with [Lemonade Server](https://github.com/onnx/turnkeyml). It offers seamless access to LLM-powered assistance directly within the terminal, providing contextual help based on the terminal's scrollback.

PEEL currently requires a Ryzen AI 300-series PC running Windows 11. 

## Key Features

PEEL adds the following PowerShell cmdlets.

Get Aid cmdlets:
* **Get-Aid**: Sends the last 50 lines of terminal history to Lemonade Server using the Llama-3.2-3B-Instruct-Hybrid model.
* **Get-MoreAid**: Like Get-Aid, but uses the Qwen-1.5-7B-Chat-Hybrid model.
* **Get-MaximumAid**: Like Get-Aid, but uses the DeepSeek-R1-Distill-Qwen-7B-Hybrid model.

Helper cmdlet:
* **Install-Lemonade**: Downloads and runs the Lemonade Server installer (GUI mode). This is just included to help you install Lemonade Server, in case you don't already have it.

## PEEL Shell Usage (Recommended)

The recommended way to use PEEL is via the **PEEL shell profile in Windows Terminal**. The installer automatically registers a PEEL profile, which launches PowerShell with the required environment variable and module import. This ensures full functionality, including automatic transcript capture for LLM assistance.

- **Transcript Recording:**
  - PEEL records a transcript of your shell session (commands and outputs) to a temporary file. This transcript is used to provide context to the LLM when you run `Get-Aid` and related commands.
  - The transcript is stored in your system's temporary directory and is unique to each PEEL shell session.

- **Environment Variable:**
  - The PEEL shell sets the `PEEL_SHELL` environment variable automatically.
  - If you want to use PEEL features in any PowerShell session (not just the Windows Terminal PEEL profile), you can manually set the `PEEL_SHELL` environment variable in your user or system environment variables, or in your PowerShell profile script:
    ```powershell
    $env:PEEL_SHELL = '1'
    Import-Module peel
    ```

- **Why use the PEEL shell?**
  - Ensures transcript-based context is available for LLM commands.
  - Prevents accidental use in non-PEEL shells, which would not capture the full scrollback.

## Installation
 * Clone this repository.
 * In PowerShell, run `install.ps1` from the same directory as this document.
 * The new PEEL shell, and its cmdlets, are now available in Windows Terminal.
 * Run the `Install-Lemonade` cmdlet to get Lemonade Server, if you don't have it already.

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
        ├── favicon.ico # Icon for Windows Terminal
```

## Usage Example

First, run some command that doesn't work, like:

```PowerShell
git pull-request
```

This will produce an error message like:

```PowerShell
git: 'pull-request' is not a git command. See 'git --help'.
```

Use the `Get-Aid` command to send the scrollback to an LLM and get help:

```PowerShell
Get-Aid
```

The whole terminal session ends up looking like this:

```PowerShell
PS C:\Users\user> git pull-request
git: 'pull-request' is not a git command. See 'git --help'.
PS C:\Users\user> Get-Aid

Lemonade Server Response:
---------------------------
The last command executed was `git pull-request`. This is not a valid Git command. The correct command to pull changes from a remote repository is `git pull`.

The error message suggests that you should see the Git documentation for more information on how to use the `git` command.

If you meant to pull changes from a remote repository, you can try running `git pull <repository-name> <branch-name>` (e.g., `git pull origin master`).

If you meant to use a different command, please let me know and I'll do my best to help.
```

## Portions Licensed as Follows

\> This project was inspired by [`wut-cli`](https://github.com/shobrook/wut) its system prompt is based on the `wut` `EXPLAIN_PROMPT` system prompt.

MIT License

Copyright (c) 2024 Jonathan Shobrook

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.