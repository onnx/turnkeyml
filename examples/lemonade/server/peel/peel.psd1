@{
    ModuleVersion = '0.1.0'
    RootModule = 'peel.psm1'
    Description = 'PEEL: PowerShell Enhanced by Embedded Lemonade (LLM) Functionality'
    Author = 'TurnkeyML'
    CompanyName = 'TurnkeyML'
    Copyright = '(c) 2024 TurnkeyML'
    FunctionsToExport = @(
        'Install-Lemonade'
        'Get-Aid'
        'Get-MoreAid'
        'Get-MaximumAid'
    )
    GUID = '7920f222-2a30-4b09-ba8b-62911134a49e'
    PowerShellVersion = '5.1'
    PowerShellHostName = 'CoreHost'
}