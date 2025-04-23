<#
    .SYNOPSIS
    This file contains the cmdlets for the PEEL (PowerShell Enhanced by Embedded Lemonade) module.
    
    Contains the implementation of Install-Lemonade, Get-Aid, Get-MoreAid, and Get-MaximumAid
#>

# --- PEEL shell detection logic ---
if ($env:PEEL_SHELL) {
    $global:PEEL_SHELL = $true
}
# --- end PEEL shell detection logic ---

# --- PEEL transcript logic ---
# Only start transcript if running in a PEEL shell
if ($env:PEEL_SHELL) {
    $global:PEELTranscriptPath = Join-Path $env:TEMP ("peel_transcript_" + $PID + "_" + [guid]::NewGuid().ToString() + ".txt")
    try {
        if (-not (Get-Variable -Name TranscriptEnabled -Scope Global -ErrorAction SilentlyContinue)) {
            $global:TranscriptEnabled = $false
        }
        if (-not $global:TranscriptEnabled) {
            Start-Transcript -Path $global:PEELTranscriptPath | Out-Null
            $global:TranscriptEnabled = $true
        }
    } catch {
        Write-Warning "Could not start transcript: $($_.Exception.Message)"
    }
}
# --- end transcript logic ---

function Install-Lemonade {
    <#
        .SYNOPSIS
        Downloads and executes the latest Lemonade Server Installer.

        .DESCRIPTION
        This cmdlet downloads the Lemonade Server Installer from the official TurnkeyML GitHub releases page
        and executes it. It handles scenarios where the installer is already present, the download fails,
        or the installer execution fails.
    #>
    [CmdletBinding()]
    param()

    $installerUrl = "https://github.com/onnx/turnkeyml/releases/latest/download/Lemonade_Server_Installer.exe"
    $installerPath = "$env:TEMP\Lemonade_Server_Installer.exe"

    Write-Host "Checking if Lemonade Server Installer is already present..."
    if (Test-Path $installerPath) {
        Write-Host "Lemonade Server Installer found at $installerPath"
    } else {
        Write-Host "Downloading Lemonade Server Installer from $installerUrl..."
        Write-Host "COMMAND: Invoke-WebRequest -Uri '$installerUrl' -OutFile '$installerPath' -ErrorAction Stop"
        try {
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -ErrorAction Stop
            Write-Host "Lemonade Server Installer downloaded successfully to $installerPath"
        } catch {
            Write-Error "Failed to download Lemonade Server Installer: $($_.Exception.Message)"
            return
        }
    }

    Write-Host "Executing Lemonade Server Installer..."
    Write-Host "Please complete the installation manually using the GUI."
    Write-Host "COMMAND: Start-Process -FilePath '$installerPath'"
    try {
        Start-Process -FilePath $installerPath
        Write-Host "Lemonade Server Installer launched. Please complete the installation in the GUI."
    } catch {
        Write-Error "Failed to execute Lemonade Server Installer: $($_.Exception.Message)"
    }
}

function Ensure-LemonadeServer {
    [CmdletBinding()]
    param(
        [int]$Port = 8000,
        [int]$MaxTries = 10,
        [int]$SleepSeconds = 2
    )
    $ServerUrl = "http://localhost:$Port/api/v0/chat/completions"
    $spinner = @('|', '/', '-', '\')
    $spinIndex = 0
    $isInstalled = $false
    $isRunning = $false
    $status = $null

    # Spinner while checking lemonade-server status (using Start-Job)
    Write-Host ""  # Blank line before spinner
    $spinnerMessage = "Getting LLM Aid..."
    $statusJob = Start-Job -ScriptBlock {
        try {
            & lemonade-server status 2>&1
        } catch {
            $null
        }
    }
    while ($statusJob.State -eq 'Running') {
        $spinChar = $spinner[$spinIndex % $spinner.Length]
        [Console]::Write("`r $spinChar $spinnerMessage   ")
        Start-Sleep -Milliseconds 80
        $spinIndex++
    }
    [Console]::Write("`r" + (' ' * 60) + "`r")
    $status = Receive-Job $statusJob
    Remove-Job $statusJob
    if ($status -match "Server is running on port $Port") {
        $isInstalled = $true
        $isRunning = $true
    } elseif ($status -match "Server is not running") {
        $isInstalled = $true
        $isRunning = $false
    }
    if (-not $isInstalled) {
        Write-Error "Lemonade Server is not installed. To use this cmdlet, please run Install-Lemonade to set up Lemonade Server first."
        return $false
    }
    if (-not $isRunning) {
        try {
            $proc = Start-Process -FilePath "lemonade-server" -ArgumentList "serve --port $Port" -WindowStyle Hidden -PassThru -ErrorAction Stop
            Start-Sleep -Seconds 2
            if ($proc.HasExited) {
                Write-Error "Lemonade Server failed to start. The port $Port may already be in use. Try closing other applications using this port or specify a different port."
                return $false
            }
        } catch {
            Write-Error "Failed to start Lemonade Server: $($_.Exception.Message)"
            return $false
        }
    }
    # Spinner animation while waiting for server health
    $spinIndex = 0
    $healthUrl = "http://localhost:$Port/api/v0/health"
    $ready = $false
    $totalTries = $MaxTries
    Write-Host ""  # Blank line before spinner
    while (-not $ready -and $totalTries -gt 0) {
        $spinChar = $spinner[$spinIndex % $spinner.Length]
        [Console]::Write("`r $spinChar Getting LLM Aid...   ")
        try {
            $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 80
        $spinIndex++
        $totalTries--
    }
    [Console]::Write("`r" + (' ' * 60) + "`r")
    if ($ready) {
        return $true
    } else {
        Write-Error "Lemonade Server did not become ready in time."
        return $false
    }
}

function Invoke-AidCore {
    [CmdletBinding()]
    param(
        [string]$Model,
        [int]$Port = 8000,
        [int]$ScrollbackLines = 50
    )
    # Prevent running unless in a PEEL shell
    if (-not ($env:PEEL_SHELL -or $global:PEEL_SHELL)) {
        Write-Error "This command can only be run in a PEEL shell. Please launch the PEEL shell from Windows Terminal."
        return
    }
    $ServerUrl = "http://localhost:$Port/api/v0/chat/completions"
    if (-not (Ensure-LemonadeServer -Port $Port)) {
        Write-Error "Lemonade Server is not available. Exiting."
        return
    }
    # --- Use transcript scrollback instead of just command history ---
    $transcriptPath = $global:PEELTranscriptPath
    if ((Test-Path $transcriptPath)) {
        $scrollback = Get-Content $transcriptPath -Raw
        # Optionally trim to last N lines/characters if needed
        $maxChars = 8000
        if ($scrollback.Length -gt $maxChars) {
            $scrollback = $scrollback.Substring($scrollback.Length - $maxChars)
        }
    } else {
        $scrollback = (Get-History -Count $ScrollbackLines).CommandLine
        if ($null -eq $scrollback) {
            $scrollback = ""
        } elseif ($scrollback -is [System.Collections.IEnumerable] -and -not ($scrollback -is [string])) {
            $scrollback = $scrollback -join "`n"
        } else {
            $scrollback = [string]$scrollback
        }
    }
    # Ensure scrollback is a string (defensive)
    if ($null -eq $scrollback) { $scrollback = "" }
    $scrollback = [string]$scrollback  # Explicitly cast to string

    $body = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = @"
<assistant>
You are a command-line assistant whose job is to explain the output of the most recently executed command in the terminal.
Your goal is to help users understand (and potentially fix) things like stack traces, error messages, logs, or any other confusing output from the terminal.
</assistant>

<instructions>

- Receive the last command and its output (from the transcript scrollback) as context.
- Do not discuss the fact that the transcript is a transcript, focus on the last command.
- Explain the output of the last command.
- Use a clear, concise, and informative tone.
- If the output is an error or warning, e.g. a stack trace or incorrect command, identify the root cause and suggest a fix.
- Otherwise, if the output is something else, e.g. logs or a web response, summarize the key points.

</instructions>
"@ },
            @{ role = "user"; content = $scrollback }
        )
        stream = $true
    } | ConvertTo-Json -Depth 5

    try {
        Add-Type -AssemblyName System.Net.Http
        $handler = New-Object System.Net.Http.HttpClientHandler
        $client = New-Object System.Net.Http.HttpClient($handler)
        $uri = $ServerUrl
        $request = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, "application/json")
        $httpRequest = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
        $httpRequest.Content = $request
        $response = $client.SendAsync($httpRequest, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        $stream = $response.Content.ReadAsStreamAsync().Result
        $reader = New-Object System.IO.StreamReader($stream)
        $firstResponse = $false
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine().Trim()
            if ($line -eq "" -or $line -eq "data: [DONE]" -or $line -eq "[DONE]") { continue }
            if ($line.StartsWith("data: ")) { $line = $line.Substring(6) }
            try {
                $data = $line | ConvertFrom-Json -ErrorAction Stop
                if ($data.choices -and $data.choices[0].delta.content) {
                    if (-not $firstResponse) {
                        [Console]::Write("`r" + (' ' * 60) + "`r")
                        Write-Host "Lemonade Server Response:" -ForegroundColor DarkCyan
                        Write-Host "---------------------------" -ForegroundColor DarkCyan
                        $firstResponse = $true
                    }
                    Write-Host $data.choices[0].delta.content -NoNewline -ForegroundColor Green
                }
            } catch {
                # Ignore lines that aren't valid JSON
                continue
            }
        }
        if (-not $firstResponse) {
            [Console]::Write("`r" + (' ' * 60) + "`r")
        }
        Write-Host ""
    } catch {
        Write-Error "Failed to connect to Lemonade Server: $($_.Exception.Message)"
    }
}

function Get-Aid {
    <#
        .SYNOPSIS
        Explains the output of your most recent terminal command using an LLM.

        .DESCRIPTION
        Captures the last 50 lines of the terminal's scrollback history, sends it to Lemonade Server via the streaming chat completions API, and displays the LLM's response in a streaming fashion within the terminal. Uses the model Llama-3.2-3B-Instruct-Hybrid.
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "Llama-3.2-3B-Instruct-Hybrid"
}

function Get-MoreAid {
    <#
        .SYNOPSIS
        Explains the output of your most recent terminal command using an LLM.

        .DESCRIPTION
        Captures the last 50 lines of the terminal's scrollback history, sends it to Lemonade Server via the streaming chat completions API, and displays the LLM's response in a streaming fashion within the terminal. Uses the model Qwen-1.5-7B-Chat-Hybrid.
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "Qwen-1.5-7B-Chat-Hybrid"
}

function Get-MaximumAid {
    <#
        .SYNOPSIS
        Explains the output of your most recent terminal command using an LLM.

        .DESCRIPTION
        Captures the last 50 lines of the terminal's scrollback history, sends it to Lemonade Server via the streaming chat completions API, and displays the LLM's response in a streaming fashion within the terminal. Uses the largest available model: DeepSeek-R1-Distill-Qwen-7B-Hybrid.
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "DeepSeek-R1-Distill-Qwen-7B-Hybrid"
}

Export-ModuleMember -Function Install-Lemonade, Get-Aid, Get-MoreAid, Get-MaximumAid


