Write-Host "Entering script..."


# Source the peel.psm1 module to make its functions available
Write-Host "Main script: Sourcing peel.psm1 module..."
Import-Module peel -Force
Write-Host "Main script: peel.psm1 module imported."
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

function Add-PEELToWindowsTerminal {
    param(
        [string]$moduleRoot
    )
    Write-Host "Add-PEELToWindowsTerminal: Starting registration..."
    try {
        Write-Host "Registering PEEL in Windows Terminal..."
        $settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
        Write-Host "Add-PEELToWindowsTerminal: Settings path: $settingsPath"
        if (!(Test-Path -Path $settingsPath)) {
            Write-Host "Add-PEELToWindowsTerminal: Settings file does not exist."
            Write-Error "Windows Terminal settings file not found: $settingsPath"
            throw "Windows Terminal settings file not found: $settingsPath"
        }
        Write-Host "Add-PEELToWindowsTerminal: Settings file found. Reading settings content..."
        $settingsContent = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        Write-Host "Add-PEELToWindowsTerminal: Settings content read."
        # Remove any existing PEEL profile
        Write-Host "Add-PEELToWindowsTerminal: Removing any existing PEEL profile..."
        $settingsContent.profiles.list = $settingsContent.profiles.list | Where-Object { $_.name -ne "PEEL" }
        # Create the PEEL profile configuration with correct absolute icon path and auto-import
        Write-Host "Add-PEELToWindowsTerminal: Creating PEEL profile configuration..."
        $workspaceRoot = (Resolve-Path -Path (Join-Path -Path $moduleRoot -ChildPath "..\..\..\.." )).Path
        $iconPath = Join-Path -Path $workspaceRoot -ChildPath "img\favicon.ico"
        $guid = "{" + ([guid]::NewGuid().ToString()) + "}"
        $newProfile = [PSCustomObject]@{
            name = "PEEL"
            guid = $guid
            # Import the PEEL module automatically on shell start
            commandline = 'powershell.exe -NoExit -Command "Import-Module peel"'
            startingDirectory = "%USERPROFILE%"
            icon = $iconPath
        }
        Write-Host "Add-PEELToWindowsTerminal: PEEL profile configuration created."
        $settingsContent.profiles.list += $newProfile
        Write-Host "Add-PEELToWindowsTerminal: PEEL profile added to settings list."
        Write-Host "Add-PEELToWindowsTerminal: Writing updated settings back to file..."
        $json = $settingsContent | ConvertTo-Json -Depth 10
        $json | Set-Content -Path $settingsPath
        Write-Host "Add-PEELToWindowsTerminal: Settings file updated successfully."
        return $true
    } catch {
        Write-Error "Error configuring PEEL in Windows Terminal: $($_.Exception.Message)"
        return $false
    }
    Write-Host "Add-PEELToWindowsTerminal: Registration completed."
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
    if ($installResult -eq $true) {
        Write-Host "Main script: Install was successful, calling Add-PEELToWindowsTerminal"
        $addResult = Add-PEELToWindowsTerminal -moduleRoot $moduleRoot
        Write-Host "Main script: Add-PEELToWindowsTerminal finished with result: $addResult"
        if ($addResult) {
            Write-Host "PEEL module and Windows Terminal profile installed successfully!"
            Write-Host "The PEEL profile will be available in Windows Terminal after the next restart."
        } else {
            Write-Error "PEEL module installed, but an error occurred when adding the profile to Windows Terminal."
        }
    } else {
        Write-Error "An error occurred while installing PEEL Module."
    }

    Write-Host "Main script: Script execution completed."

