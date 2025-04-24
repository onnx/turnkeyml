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
 * The new cmdlets are now available.
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

Use the `Get-MoreAid` command to send the scrollback to an LLM and get help:

```PowerShell
Get-MoreAid
```

The whole terminal session ends up looking like this:

```PowerShell
(base) PS C:\Users\user> git pull-request
git: 'pull-request' is not a git command. See 'git --help'.
(base) PS C:\Users\user> Get-MoreAid
lemonade-server status output: Server is running on port 8000
Waiting for Lemonade Server to become ready (timeout: 40 seconds)...
Health check attempt 1 of 20: http://localhost:8000/api/v0/health
Lemonade Server is ready.
Capturing terminal history...
Sending history to Lemonade Server...

Lemonade Server Response:
---------------------------
The `git pull-request` command is not recognized as a built-in Git command. It seems you might be referring to GitHub's workflow or a misunderstanding. When you want to create a pull request on GitHub, you would typically use the following steps:

1. First, make sure you are in the correct repository directory using `cd [repository-name]`.
2. Check out the latest changes from the remote branch with `git pull origin [branch-name]`. Replace `[branch-name]` with the name of the branch you want to merge.
3. If there are any changes you want to contribute, stage them with `git add .`, then commit them with a meaningful message using `git commit -m "Your commit message"`.
4. Finally, go to the GitHub repository's page, click on the "New pull request" button, and fill out the details, including selecting the base branch and the branch you just committed.

If you're encountering an error or need assistance with this process, please provide the specific error message or the context of your command, and I'll be happy to help you interpret the output and suggest a solution.
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