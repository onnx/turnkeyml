powershell
function Install-PEELModule {
    param(
        [string]$moduleRoot
    )
    Write-Host "Install-PEELModule: Starting installation..."
    $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "PowerShell\Modules\peel"
    Write-Host "Install-PEELModule: Destination path set to: $destinationPath"
    try {
        Write-Host "Install-PEELModule: Checking if PEEL module is already installed..."
        if (Test-Path -Path $destinationPath) {
            Write-Host "PEEL module is already installed."
        } else {
            # Copy the module files to the destination directory
            Write-Host "Install-PEELModule: PEEL module not found. Starting copying process..."
            Write-Host "Copying module files to: $destinationPath"
            Write-Host "Install-PEELModule: Copying files..."
            Copy-Item -Path (Join-Path -Path $moduleRoot -ChildPath "*") -Destination $destinationPath -Recurse -Force
            Write-Host "Install-PEELModule: Files copied successfully."
        }
    }
    catch{
        Write-Error "Error copying PEEL module files: $($_.Exception.Message)"
        throw $_
    }
    Write-Host "Install-PEELModule: Installation completed."
}

function Add-PEELToWindowsTerminal {
    Write-Host "Add-PEELToWindowsTerminal: Starting registration..."
    param(
        [string]$moduleRoot
    )
    # Register PEEL in Windows Terminal profiles
    try {
        Write-Host "Registering PEEL in Windows Terminal..."
        $settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
        Write-Host "Add-PEELToWindowsTerminal: Settings path: $settingsPath"
        #Check if the file exist
        if (!(Test-Path -Path $settingsPath)) {
            Write-Host "Add-PEELToWindowsTerminal: Settings file does not exist."
            Write-Error "Windows Terminal settings file not found: $settingsPath"
            throw "Windows Terminal settings file not found: $settingsPath"
        }
        Write-Host "Add-PEELToWindowsTerminal: Settings file found. Reading settings content..."
        $settingsContent = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        Write-Host "Add-PEELToWindowsTerminal: Settings content read."
        # Check if the PEEL profile already exists
        Write-Host "Add-PEELToWindowsTerminal: Checking if PEEL profile already exists..."
        $peelProfileExists = $settingsContent.profiles.list | Where-Object { $_.name -eq "PEEL" }
        Write-Host "Add-PEELToWindowsTerminal: Profile existence check completed."
        if ($peelProfileExists) {
            Write-Host "PEEL profile already exists in Windows Terminal."
            Write-Host "Add-PEELToWindowsTerminal: Registration skipped, profile already exists."
            return
        }
        

        catch {
            Write-Error "Error reading Windows Terminal settings file: $($_.Exception.Message)"
            throw $_
        }

       
        # Create the PEEL profile configuration
        Write-Host "Add-PEELToWindowsTerminal: Creating PEEL profile configuration..."
        $newProfile = @{
            "name" = "PEEL"
            "guid" = (New-Guid).Guid.ToString()
            "commandline" = "powershell.exe"
            "startingDirectory" = "%USERPROFILE%"
            "icon" = "$(Join-Path -Path $moduleRoot -ChildPath "../../../img/favicon.ico")"           
        }
        Write-Host "Add-PEELToWindowsTerminal: PEEL profile configuration created."
        # Add the new profile to the settings
        Write-Host "Add-PEELToWindowsTerminal: Adding PEEL profile to settings list..."
        $settingsContent.profiles.list += $newProfile
        Write-Host "Add-PEELToWindowsTerminal: PEEL profile added to settings list."
        # Write the updated configuration back to the settings file
        Write-Host "Add-PEELToWindowsTerminal: Writing updated settings back to file..."
        $settingsContent | ConvertTo-Json -Depth 10 | Set-Content -Path $settingsPath
        Write-Host "Add-PEELToWindowsTerminal: Settings file updated successfully."
    }
     catch {
        Write-Error "Error configuring PEEL in Windows Terminal: $($_.Exception.Message)"
        throw $_
    }
    Write-Host "Add-PEELToWindowsTerminal: Registration completed."

}

# Main script
{
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
    $installResult = Install-PEELModule -moduleRoot $moduleRoot -destinationPath $destinationPath
    Write-Host "Main script: Install-PEELModule finished with result: $installResult"

    if ($installResult) {
        Write-Host "Main script: Install was successfull, calling Add-PEELToWindowsTerminal"

        $addResult = Add-PEELToWindowsTerminal -moduleRoot $moduleRoot
        Write-Host "Main script: Add-PEELToWindowsTerminal finished with result: $addResult"
        if($addResult) {
            # Notify the user of success
            Write-Host "PEEL module and Windows Terminal profile installed successfully!"
            Write-Host "The PEEL profile will be available in Windows Terminal after the next restart."
        } else {
            Write-Error "PEEL module install, but an error occur when adding the profile to windows terminal"
        }
    } else {
        Write-Error "An error occurred while installing PEEL Module"
    }

    Write-Host "Main script: Script execution completed."
}
