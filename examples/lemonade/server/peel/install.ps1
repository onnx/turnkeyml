powershell
function Install-PEELModule {
    param(
        [string]$moduleRoot
    )
    $destinationPath = Join-Path -Path ([Environment]::GetFolderPath("MyDocuments")) -ChildPath "PowerShell\Modules\peel"
    try {
        if (Test-Path -Path $destinationPath) {
            Write-Host "PEEL module is already installed."
        } else {
            # Copy the module files to the destination directory
            Write-Host "Copying module files to: $destinationPath"
            Copy-Item -Path (Join-Path -Path $moduleRoot -ChildPath "*") -Destination $destinationPath -Recurse -Force
        }
    }
    catch{
        Write-Error "Error copying PEEL module files: $($_.Exception.Message)"
        throw $_
    }
    
}

function Add-PEELToWindowsTerminal {
    param(
        [string]$moduleRoot
    )
    # Register PEEL in Windows Terminal profiles
    try {
        Write-Host "Registering PEEL in Windows Terminal..."
        $settingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

        #Check if the file exist
        if (!(Test-Path -Path $settingsPath)) {
            Write-Error "Windows Terminal settings file not found: $settingsPath"
            throw "Windows Terminal settings file not found: $settingsPath"
        }

        $settingsContent = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        
        # Check if the PEEL profile already exists
        $peelProfileExists = $settingsContent.profiles.list | Where-Object { $_.name -eq "PEEL" }

        if ($peelProfileExists) {
            Write-Host "PEEL profile already exists in Windows Terminal."
            return
        }
        catch {
            Write-Error "Error reading Windows Terminal settings file: $($_.Exception.Message)"
            throw $_
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
    }
     catch {
        Write-Error "Error configuring PEEL in Windows Terminal: $($_.Exception.Message)"
        throw $_
    }
    

}

# Main script
{
    $VerbosePreference = "Continue"
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
    
    $installResult = Install-PEELModule -moduleRoot $moduleRoot -destinationPath $destinationPath
    
    if ($installResult) {
        $addResult = Add-PEELToWindowsTerminal -moduleRoot $moduleRoot
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

}
