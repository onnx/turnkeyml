<#
    .SYNOPSIS
    This file contains the cmdlets for the PEEL (PowerShell Enhanced by Embedded Lemonade) module.
    
    Contains the implementation of Install-Lemonade, Get-Aid, Get-MoreAid, and Get-MaximumAid
#>

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
        Write-Host ("`r $spinChar $spinnerMessage ") -NoNewline -ForegroundColor Yellow
        Start-Sleep -Milliseconds 120
        $spinIndex++
    }
    $status = Receive-Job $statusJob
    Remove-Job $statusJob
    Write-Host ("`r" + (' ' * 60) + "`r") -NoNewline
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
        Write-Host ("`r $spinChar Getting LLM Aid... ") -NoNewline -ForegroundColor Yellow
        try {
            $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 200
        $spinIndex++
        $totalTries--
    }
    # Clear spinner line
    Write-Host ("`r" + (' ' * 60) + "`r") -NoNewline
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
    $ServerUrl = "http://localhost:$Port/api/v0/chat/completions"
    if (-not (Ensure-LemonadeServer -Port $Port)) {
        Write-Error "Lemonade Server is not available. Exiting."
        return
    }
    $history = (Get-History -Count $ScrollbackLines).CommandLine | Out-String
    $body = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = @"
You are a command-line assistant whose job is to explain the output of the most recently executed command in the terminal.
Your goal is to help users understand (and potentially fix) things like stack traces, error messages, logs, or any other confusing output from the terminal.

- Receive the last command in the terminal history and the previous commands before it as context.
- Explain the output of the last command.
- Use a clear, concise, and informative tone.
- If the output is an error or warning, e.g. a stack trace or incorrect command, identify the root cause and suggest a fix.
- Otherwise, if the output is something else, e.g. logs or a web response, summarize the key points.
"@ },
            @{ role = "user"; content = $history }
        )
        stream = $true
    } | ConvertTo-Json
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
        # Spinner until first response
        $spinner = @('|', '/', '-', '\')
        $spinIndex = 0
        $firstResponse = $false
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine().Trim()
            if ($line -eq "" -or $line -eq "data: [DONE]" -or $line -eq "[DONE]") { continue }
            if ($line.StartsWith("data: ")) { $line = $line.Substring(6) }
            try {
                $data = $line | ConvertFrom-Json -ErrorAction Stop
                if ($data.choices -and $data.choices[0].delta.content) {
                    if (-not $firstResponse) {
                        # Clear spinner line and print response header
                        Write-Host ("`r" + (' ' * 60) + "`r") -NoNewline
                        Write-Host "Lemonade Server Response:" -ForegroundColor DarkCyan
                        Write-Host "---------------------------" -ForegroundColor DarkCyan
                        $firstResponse = $true
                    }
                    Write-Host $data.choices[0].delta.content -NoNewline -ForegroundColor Green
                }
            } catch {
                # Ignore lines that aren't valid JSON
                if (-not $firstResponse) {
                    $spinChar = $spinner[$spinIndex % $spinner.Length]
                    Write-Host ("`r $spinChar Waiting for Lemonade Server response... ") -NoNewline -ForegroundColor Yellow
                    Start-Sleep -Milliseconds 120
                    $spinIndex++
                }
            }
        }
        if ($firstResponse) { Write-Host "" -ForegroundColor DarkCyan }
    } catch {
        Write-Error "Failed to connect to Lemonade Server: $($_.Exception.Message)"
    }
}

function Get-Aid {
    <#
        .SYNOPSIS
        Captures the terminal's scrollback history, sends it to Lemonade Server, and displays the LLM's response.

        .DESCRIPTION
        This cmdlet captures the last 50 lines of the terminal's scrollback history, sends it to the
        Lemonade Server via the streaming chat completions API, and displays the LLM's response in
        a streaming fashion within the terminal.
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "Llama-3.2-3B-Instruct-Hybrid"
}

function Get-MoreAid {
    <#
        .SYNOPSIS
        Captures the terminal's scrollback history, sends it to Lemonade Server, and displays the LLM's response.

        .DESCRIPTION
        This cmdlet captures the last 50 lines of the terminal's scrollback history, sends it to the
        Lemonade Server via the streaming chat completions API, and displays the LLM's response in
        a streaming fashion within the terminal.
        This uses the model Qwen-1.5-7B-Chat-Hybrid
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "Qwen-1.5-7B-Chat-Hybrid"
}

function Get-MaximumAid {
    <#
        .SYNOPSIS
        Captures the terminal's scrollback history, sends it to Lemonade Server, and displays the LLM's response using the largest model.
        .DESCRIPTION
        This cmdlet captures the last 50 lines of the terminal's scrollback history, sends it to the Lemonade Server via the streaming chat completions API, and displays the LLM's response in a streaming fashion within the terminal. This uses the model DeepSeek-R1-Distill-Qwen-7B-Hybrid.
    #>
    [CmdletBinding()]
    param()
    Invoke-AidCore -Model "DeepSeek-R1-Distill-Qwen-7B-Hybrid"
}

Export-ModuleMember -Function Install-Lemonade, Get-Aid, Get-MoreAid, Get-MaximumAid


