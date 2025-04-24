function Install-PEELModule {
    param(
        [string]$moduleRoot
    )
    try {
        $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "WindowsPowerShell\Modules\peel"
        # Always copy to allow updates/fixes
        if (!(Test-Path -Path $destinationPath)) {
            New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
        }
        # Copy only peel.psm1 to the module root
        $sourceModule = Join-Path -Path $moduleRoot -ChildPath "peel.psm1"
        if (Test-Path $sourceModule) {
            Copy-Item -Path $sourceModule -Destination $destinationPath -Force
        } else {
            Write-Error "Could not find peel.psm1 in $moduleRoot"
            return $false
        }
        return $true
    }
    catch {
        Write-Error "Error copying PEEL module files: $($_.Exception.Message)"
        return $false
    }
}

# Main script
$ErrorActionPreference = "Stop"
$cmdlets = @()
try {
    $VerbosePreference = "Continue"

    # Get the path of the current script
    $scriptPath = $MyInvocation.MyCommand.Definition
    # Get the directory where the script is located
    $moduleRoot = Split-Path $scriptPath

    # Get the path to favicon.ico in the same folder as the script
    $faviconPath = Join-Path $moduleRoot "favicon.ico"

    # Define the destination path for the module in the current user profile, handle errors with try-catch
    try {
        $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "PowerShell\Modules\peel"
        # Create the destination directory if it does not exist
        if (!(Test-Path -Path $destinationPath)) {
            New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
        }
    }
    catch {
        Write-Error "Error creating PowerShell modules directory: $($_.Exception.Message)"
        return
    }
    $installResult = Install-PEELModule -moduleRoot $moduleRoot

    $importSuccess = $true
    try {
        $oldVerbose = $VerbosePreference
        $VerbosePreference = "SilentlyContinue"
        Import-Module peel -Force -ErrorAction Stop
        $VerbosePreference = $oldVerbose
        # Get exported cmdlets from the module
        $cmdlets = (Get-Command -Module peel | Where-Object { $_.CommandType -eq 'Function' } | Select-Object -ExpandProperty Name)
    } catch {
        Write-Error "Failed to import peel.psm1: $($_.Exception.Message)"
        $importSuccess = $false
    }

    # Register PEEL shell in Windows Terminal (if not already present)
    try {
        $wtSettingsPath = Join-Path $env:LOCALAPPDATA "Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
        if (Test-Path $wtSettingsPath) {
            $settings = Get-Content $wtSettingsPath -Raw | ConvertFrom-Json
            $peelProfile = $settings.profiles.list | Where-Object { $_.name -eq "PEEL" }
            $pwshPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
            $peelCommand = "$pwshPath -NoExit -Command & { `$env:PEEL_SHELL='1'; Import-Module peel }"
            if ($peelProfile) {
                $peelProfile.commandline = $peelCommand
                $peelProfile.icon = $faviconPath
                if ($peelProfile.PSObject.Properties["env"]) {
                    $peelProfile.PSObject.Properties.Remove("env")
                }
            } else {
                $peelProfileObj = [PSCustomObject]@{
                    name = "PEEL"
                    commandline = $peelCommand
                    icon = $faviconPath
                    startingDirectory = "~"
                    hidden = $false
                    guid = "{b1b1b1b1-1111-1111-1111-111111111111}"
                }
                $settings.profiles.list += $peelProfileObj
            }
            $settings | ConvertTo-Json -Depth 100 | Set-Content $wtSettingsPath -Encoding UTF8
        }
    } catch {
        Write-Warning "Could not register PEEL shell in Windows Terminal: $($_.Exception.Message)"
    }

    if ($importSuccess -and $installResult -eq $true) {
        Write-Host "==============================="
        Write-Host " PEEL Module Installation"
        Write-Host "==============================="
        Write-Host "Installed cmdlets:"
        foreach ($cmdlet in $cmdlets) {
            Write-Host "  - $cmdlet"
        }
        Write-Host ""
        Write-Host "To use the Get-Aid cmdlets, open a PEEL shell from Windows Terminal."
        Write-Host ""
        Write-Host "If you don't have Lemonade Server installed, run: Install-Lemonade"
        Write-Host "==============================="
        exit 0
    } else {
        Write-Error "An error occurred while installing or importing the PEEL Module."
        exit 1
    }
} catch {
    Write-Error $($_.Exception.Message)
    exit 1
}

# Note: The PEEL shell sets the PEEL_SHELL environment variable. peel.psm1 should check for this variable to enable transcript logic only in PEEL shells.

