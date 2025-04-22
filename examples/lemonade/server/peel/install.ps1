powershell
# Get the path of the current script
$scriptPath = $MyInvocation.MyCommand.Definition
# Get the directory where the script is located
$moduleRoot = Split-Path $scriptPath

# Define the destination path for the module in the current user profile
$destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "PowerShell\Modules\peel"

# Check if the module is already installed
if (Test-Path -Path $destinationPath) {
    Write-Host "PEEL module is already installed."
} else {
    # Create the destination directory if it does not exist
    Write-Host "Creating directory: $destinationPath"
    New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

    # Copy the module files to the destination directory
    Write-Host "Copying module files to: $destinationPath"
    Copy-Item -Path (Join-Path -Path $moduleRoot -ChildPath "*") -Destination $destinationPath -Recurse -Force

    # Notify the user
    Write-Host "PEEL module installed successfully to: $destinationPath"
}

# Register PEEL in Windows Terminal profiles
try {
    Write-Host "Registering PEEL in Windows Terminal..."
    $settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

    # Read the settings.json file
    $settingsContent = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Check if the PEEL profile already exists
    $peelProfileExists = $settingsContent.profiles.list | Where-Object { $_.name -eq "PEEL" }

    if ($peelProfileExists) {
        Write-Host "PEEL profile already exists in Windows Terminal."
    } else {
        # Create the PEEL profile configuration
        $newProfile = @{
            "name" = "PEEL"
            "guid" = (New-Guid).Guid.ToString()
            "commandline" = "powershell.exe"
            "startingDirectory" = "%USERPROFILE%"
            "icon" = "$(Join-Path -Path $moduleRoot -ChildPath "../../../img/favicon.ico")"
        }

        # Add the new profile to the settings
        $settingsContent.profiles.list += $newProfile

        # Write the updated configuration back to the settings file
        $settingsContent | ConvertTo-Json -Depth 10 | Set-Content -Path $settingsPath

        Write-Host "PEEL profile added to Windows Terminal successfully."
    }
} catch {
    Write-Error "Failed to configure PEEL in Windows Terminal: $($_.Exception.Message)"
} finally{
    #Notify the user
    if (!(Test-Path -Path $destinationPath)) {
        Write-Host "PEEL module installed successfully to: $destinationPath"
    }
    Write-Host "The PEEL profile will be available in Windows Terminal after the next restart."
}

