powershell
@{
    ModuleVersion = '0.1.0'
    RootModule = 'peel.psm1'
    Description = 'PEEL: Powershell Enhanced with Enhanced Lemonade (LLM) Functionality'
    Author = 'TurnkeyML'
    CompanyName = 'TurnkeyML'
    Copyright = '(c) 2024 TurnkeyML'
    FunctionsToExport = @(
        'Lemonade-Install'
        'Get-Aid'
        'Get-More-Aid'
        'Get-Maximum-Aid'
    )
    GUID = '7920f222-2a30-4b09-ba8b-62911134a49e'
    PowerShellVersion = '5.1'
    PowerShellHostName = 'CoreHost'

}