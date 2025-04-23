
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

if ($importSuccess -and $installResult -eq $true) {
    Write-Host "PEEL module installed successfully!"
    exit 0
} else {
    Write-Error "An error occurred while installing or importing the PEEL Module."
    exit 1
}

Write-Host "Main script: Script execution completed."

