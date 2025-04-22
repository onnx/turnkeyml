<#
    .SYNOPSIS
    This file contains the cmdlets for the PEEL (PowerShell Enhanced with Enhanced Lemonade) module.
    
    Contains the implementation of Lemonade-Install and Get-Aid
#>



function Lemonade-Install {
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

    # Set parameters
    $serverUrl = "http://localhost:8080/v1/chat/completions"
    $model = "Llama-3.2-3B-Instruct-Hybrid"
    $scrollbackLines = 50

    # Get scrollback history
    Write-Host "Capturing terminal history..."
    $history = (Get-History -Count $scrollbackLines).CommandLine | Out-String

    # Prepare payload
    $payload = @{
        model = $model
        messages = @(
            @{
                role = "system"
                content = "You are a helpful assistant that helps the user with powershell"
            }
            @{
                role = "user"
                content = "This is the powershell console history : $($history) \n
                Help me with the errors or the next command I need to do."
            }
        )
        stream = $true
    } | ConvertTo-Json
    
    # Send request to Lemonade Server and process streaming response
    Write-Host "Sending history to Lemonade Server..."
    try {
        $response = Invoke-WebRequest -Uri $serverUrl -Method Post -Headers @{"Content-Type" = "application/json"} -Body $payload -UseBasicParsing -ErrorAction Stop

        if ($response.StatusCode -ne 200) {
            Write-Error "Lemonade Server returned status code: $($response.StatusCode)"
        }

        $jsonLines = ($response.Content -split "`n" )| Where-Object {$_.trim() -ne ""}
        Write-Host "`n" -ForegroundColor DarkCyan
        Write-Host "Lemonade Server Response:" -ForegroundColor DarkCyan
        Write-Host "---------------------------" -ForegroundColor DarkCyan
        foreach ($jsonLine in $jsonLines) {
           $data = ($jsonLine -replace "data: ", "") | ConvertFrom-Json -ErrorAction SilentlyContinue
           if ($data.choices -and $data.choices[0].delta.content)
           {
               Write-Host $data.choices[0].delta.content -NoNewline -ForegroundColor Green
           }
        }
        Write-Host "`n" -ForegroundColor DarkCyan

    } catch {
        Write-Error "Failed to connect to Lemonade Server: $($_.Exception.Message)"
    }
}

function Get-More-Aid {
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

    # Set parameters
    $serverUrl = "http://localhost:8080/v1/chat/completions"
    $model = "Qwen-1.5-7B-Chat-Hybrid"
    $scrollbackLines = 50

    # Get scrollback history
    Write-Host "Capturing terminal history..."
    $history = (Get-History -Count $scrollbackLines).CommandLine | Out-String

    # Prepare payload
    $payload = @{
        model = $model
        messages = @(
            @{
                role = "system"
                content = "You are a helpful assistant that helps the user with powershell"
            }
            @{
                role = "user"
                content = "This is the powershell console history : $($history) \n
                Help me with the errors or the next command I need to do."
            }
        )
        stream = $true
    } | ConvertTo-Json

    # Send request to Lemonade Server and process streaming response
    Write-Host "Sending history to Lemonade Server..."
    Get-Aid -serverUrl $serverUrl -model $model -history $history -payload $payload
}



# Reusable function for sending history and processing response





Export-ModuleMember -Function Lemonade-Install, Get-Aid


