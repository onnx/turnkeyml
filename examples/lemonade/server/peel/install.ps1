function Install-PEELModule {
    param(
        [string]$moduleRoot
    )
    Write-Host "Install-PEELModule: Starting installation..."
    $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "WindowsPowerShell\Modules\peel"
    Write-Host "Install-PEELModule: Destination path set to: $destinationPath"
    try {
        Write-Host "Install-PEELModule: Checking if PEEL module is already installed..."
        # Always copy to allow updates/fixes
        if (!(Test-Path -Path $destinationPath)) {
            Write-Host "Creating PEEL module directory: $destinationPath"
            New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
        }
        # Copy only peel.psm1 to the module root
        $sourceModule = Join-Path -Path $moduleRoot -ChildPath "peel.psm1"
        if (Test-Path $sourceModule) {
            Write-Host "Copying $sourceModule to $destinationPath"
            Copy-Item -Path $sourceModule -Destination $destinationPath -Force
        } else {
            Write-Error "Could not find peel.psm1 in $moduleRoot"
            return $false
        }
        Write-Host "Install-PEELModule: Files copied successfully."
        return $true
    }
    catch {
        Write-Error "Error copying PEEL module files: $($_.Exception.Message)"
        return $false
    }
}

# Main script
Write-Host "Main script: Starting script execution..."
$VerbosePreference = "Continue"
Write-Host "Main script: Verbose preference set to Continue"
# Notify the user that the installation is starting
Write-Host "Starting PEEL installation..."

# Get the path of the current script
Write-Host "Main script: Getting script path..."
$scriptPath = $MyInvocation.MyCommand.Definition
Write-Host "Main script: Script path: $scriptPath"
# Get the directory where the script is located
Write-Host "Main script: Getting module root..."
$moduleRoot = Split-Path $scriptPath

# Define the destination path for the module in the current user profile, handle errors with try-catch
try {
    $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "PowerShell\Modules\peel"
    # Create the destination directory if it does not exist
    if (!(Test-Path -Path $destinationPath)) {
        Write-Host "Creating directory: $destinationPath"
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
    }
}
catch {
    Write-Error "Error creating PowerShell modules directory: $($_.Exception.Message)"
    return
}
Write-Host "Main script: calling Install-PEELModule"
$installResult = Install-PEELModule -moduleRoot $moduleRoot

# Import the module only after copying the new version
$importSuccess = $true
try {
    Import-Module peel -Force -ErrorAction Stop
    Write-Host "Main script: peel.psm1 module imported."
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
        $peelCommand = "$pwshPath -NoExit -Command ""`$env:PEEL_SHELL='1'; Import-Module peel"""
        if ($peelProfile) {
            # Update existing PEEL profile to use correct commandline and remove 'env' property if present
            $peelProfile.commandline = $peelCommand
            if ($peelProfile.PSObject.Properties["env"]) {
                $peelProfile.PSObject.Properties.Remove("env")
            }
        } else {
            $peelProfileObj = [PSCustomObject]@{
                name = "PEEL"
                commandline = $peelCommand
                icon = ""
                startingDirectory = "~"
                hidden = $false
                guid = "{b1b1b1b1-1111-1111-1111-111111111111}"
            }
            $settings.profiles.list += $peelProfileObj
        }
        $settings | ConvertTo-Json -Depth 100 | Set-Content $wtSettingsPath -Encoding UTF8
        Write-Host "PEEL shell profile registered or updated in Windows Terminal."
    } else {
        Write-Warning "Windows Terminal settings.json not found. Skipping PEEL shell registration."
    }
} catch {
    Write-Warning "Could not register PEEL shell in Windows Terminal: $($_.Exception.Message)"
}

if ($importSuccess -and $installResult -eq $true) {
    Write-Host "PEEL module installed successfully!"
    exit 0
} else {
    Write-Error "An error occurred while installing or importing the PEEL Module."
    exit 1
}

# Note: The PEEL shell sets the PEEL_SHELL environment variable. peel.psm1 should check for this variable to enable transcript logic only in PEEL shells.

Write-Host "Main script: Script execution completed."

