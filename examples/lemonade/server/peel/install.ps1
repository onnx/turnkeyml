powershell
# Notify the user that the installation is starting
Write-Host "Starting PEEL installation..."

# Get the path of the current script
$scriptPath = $MyInvocation.MyCommand.Definition
# Get the directory where the script is located
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

# Check if the module is already installed, handle errors with try-catch
try {
    if (Test-Path -Path $destinationPath) {
        Write-Host "PEEL module is already installed."
    } else {
        # Copy the module files to the destination directory
        Write-Host "Copying module files to: $destinationPath"
        Copy-Item -Path (Join-Path -Path $moduleRoot -ChildPath "*") -Destination $destinationPath -Recurse -Force
    }
}
catch {
    Write-Error "Error copying PEEL module files: $($_.Exception.Message)"
    return
}


# Register PEEL in Windows Terminal profiles
try {
    Write-Host "Registering PEEL in Windows Terminal..."
    $settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

    #Check if the file exist
    if (!(Test-Path -Path $settingsPath)) {
        Write-Error "Windows Terminal settings file not found: $settingsPath"
        return
    }

    # Read the settings.json file, handle errors with try-catch
    try {
        $settingsContent = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
    }
    catch {
        Write-Error "Error reading Windows Terminal settings file: $($_.Exception.Message)"
        return
    }

        # Check if the PEEL profile already exists
    $peelProfileExists = $settingsContent.profiles.list | Where-Object { $_.name -eq "PEEL" } | Select-Object -First 1

    if ($peelProfileExists) {
        Write-Host "PEEL profile already exists in Windows Terminal."
        return
    } 
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
} catch {
    Write-Error "Error configuring PEEL in Windows Terminal: $($_.Exception.Message)"
    return
}
# Notify the user of success
Write-Host "PEEL module and Windows Terminal profile installed successfully!"
Write-Host "The PEEL profile will be available in Windows Terminal after the next restart."

