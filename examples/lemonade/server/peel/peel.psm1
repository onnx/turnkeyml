<#
    .SYNOPSIS
    This file contains the cmdlets for the PEEL (PowerShell Enhanced with Enhanced Lemonade) module.
    
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
        try {
            Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -ErrorAction Stop
            Write-Host "Lemonade Server Installer downloaded successfully to $installerPath"
        } catch {
            Write-Error "Failed to download Lemonade Server Installer: $($_.Exception.Message)"
            return
        }
    }

    Write-Host "Executing Lemonade Server Installer..."
    try {
        Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -ErrorAction Stop
        Write-Host "Lemonade Server Installer executed successfully."
    } catch {
        Write-Error "Failed to execute Lemonade Server Installer: $($_.Exception.Message)"
    }
}

function Ensure-LemonadeServer {
    [CmdletBinding()]
    param(
        [string]$InstallerUrl = "https://github.com/onnx/turnkeyml/releases/latest/download/Lemonade_Server_Installer.exe",
        [int]$Port = 8000,
        [int]$MaxTries = 20,
        [int]$SleepSeconds = 1
    )
    $ServerUrl = "http://localhost:$Port/api/v0/chat/completions"
    # 1. Check if Lemonade Server is installed (using CLI)
    $isInstalled = $false
    try {
        $version = & lemonade-server --version 2>&1
        if ($LASTEXITCODE -eq 0) { $isInstalled = $true }
        else { Write-Host "lemonade-server --version output: $version" }
    } catch { Write-Host "Error running lemonade-server --version: $_" }
    if (-not $isInstalled) {
        Write-Host "Lemonade Server not found. Installing..."
        $installerPath = "$env:TEMP\Lemonade_Server_Installer.exe"
        if (!(Test-Path $installerPath)) {
            Write-Host "Downloading Lemonade Server Installer from $InstallerUrl..."
            try {
                Invoke-WebRequest -Uri $InstallerUrl -OutFile $installerPath -ErrorAction Stop
                Write-Host "Lemonade Server Installer downloaded successfully to $installerPath"
            } catch {
                Write-Error "Failed to download Lemonade Server Installer: $($_.Exception.Message)"
                return $false
            }
        }
        Write-Host "Executing Lemonade Server Installer..."
        try {
            Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -ErrorAction Stop
            Write-Host "Lemonade Server Installer executed successfully."
        } catch {
            Write-Error "Failed to execute Lemonade Server Installer: $($_.Exception.Message)"
            return $false
        }
    }
    # 2. Check if Lemonade Server is running (using CLI)
    $isRunning = $false
    try {
        $status = & lemonade-server status 2>&1
        Write-Host "lemonade-server status output: $status"
        if ($status -match "Server is running on port $Port") { $isRunning = $true }
    } catch { Write-Host "Error running lemonade-server status: $_" }
    if (-not $isRunning) {
        Write-Host "Lemonade Server is not running. Starting..."
        try {
            Write-Host "Running: lemonade-server serve --port $Port"
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
    # 3. Wait for server to be ready
    Write-Host "Waiting for Lemonade Server to become ready (timeout: $(${MaxTries} * $SleepSeconds) seconds)..."
    $healthUrl = "http://localhost:$Port/api/v0/health"
    for ($i=1; $i -le $MaxTries; $i++) {
        Write-Host "Health check attempt $i of ${MaxTries}: $healthUrl"
        try {
            $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) {
                Write-Host "Lemonade Server is ready."
                return $true
            } else {
                Write-Host "Health check status code: $($resp.StatusCode)"
            }
        } catch {
            Write-Host "Health check failed: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds $SleepSeconds
    }
    Write-Error "Lemonade Server did not become ready in time after $MaxTries attempts."
    return $false
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
        Write-Error "Lemonade Server is not available. Aborting."
        return
    }
    Write-Host "Capturing terminal history..."
    $history = (Get-History -Count $ScrollbackLines).CommandLine | Out-String
    $payload = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = "You are a helpful assistant that helps the user with powershell" },
            @{ role = "user"; content = "This is the powershell console history : $($history) \n
                Help me with the errors or the next command I need to do." }
        )
        stream = $true
    } | ConvertTo-Json
    Write-Host "Sending history to Lemonade Server..."
    Write-Host "POST URL: $ServerUrl"
    Write-Host "POST BODY: $payload"
    try {
        $response = Invoke-WebRequest -Uri $ServerUrl -Method Post -Headers @{"Content-Type" = "application/json"} -Body $payload -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 404) {
            Write-Host "DEBUG: 404 Not Found from $ServerUrl"
            # Check if the model is available
            Write-Host "Checking available models on Lemonade Server..."
            try {
                $modelsUrl = "http://localhost:" + $Port + "/api/v0/models"
                Write-Host "DEBUG: GET $modelsUrl"
                $modelsResp = Invoke-WebRequest -Uri $modelsUrl -UseBasicParsing -ErrorAction Stop
                Write-Host "DEBUG: Models response: $($modelsResp.Content)"
                $models = ($modelsResp.Content | ConvertFrom-Json).data.id
                if ($models -notcontains $Model) {
                    Write-Error "Model '$Model' is not available on Lemonade Server. Available models: $($models -join ', ')"
                } else {
                    Write-Error "Lemonade Server is running, but the /api/v0/chat/completions endpoint was not found (404). Please check your Lemonade Server version and configuration."
                }
            } catch {
                Write-Error "Failed to fetch available models from Lemonade Server. The /api/v0/chat/completions endpoint may be missing, or the server is not OpenAI-compatible."
            }
            return
        } elseif ($response.StatusCode -ne 200) {
            Write-Host "DEBUG: Non-200 status code: $($response.StatusCode)"
            Write-Error "Lemonade Server returned status code: $($response.StatusCode)"
        }
        Write-Host "DEBUG: Response content: $($response.Content)"
        $jsonLines = ($response.Content -split "`n" )| Where-Object {$_.trim() -ne ""}
        Write-Host "`n" -ForegroundColor DarkCyan
        Write-Host "Lemonade Server Response:" -ForegroundColor DarkCyan
        Write-Host "---------------------------" -ForegroundColor DarkCyan
        foreach ($jsonLine in $jsonLines) {
           $data = ($jsonLine -replace "data: ", "") | ConvertFrom-Json -ErrorAction SilentlyContinue
           if ($data.choices -and $data.choices[0].delta.content) {
               Write-Host $data.choices[0].delta.content -NoNewline -ForegroundColor Green
           }
        }
        Write-Host "`n" -ForegroundColor DarkCyan
    } catch {
        Write-Host "DEBUG: Exception thrown in Invoke-WebRequest: $($_.Exception.Message)"
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


