<#
.SYNOPSIS
    Select the QAIRT/QNN 2.47.0.260601 environment for this PowerShell scope.

.DESCRIPTION
    Dot-source this file before QNN AOT commands:

        . .\scripts\use_qairt_2_47_0_260601.ps1

    The script does not alter the installed SDK and does not touch the WSL
    CUDA environment.  P0 scale experiments themselves do not require QNN;
    this scope is for the later AOT/SM8750 gate.
#>

$QairtRoot = 'D:\llm_exp\models\qualcomm-sdk\qairt\2.47.0.260601'
if (-not (Test-Path -LiteralPath $QairtRoot -PathType Container)) {
    throw "QAIRT root does not exist: $QairtRoot"
}

$env:QAIRT_ROOT = $QairtRoot
$env:QNN_SDK_ROOT = $QairtRoot
$env:QNN_SDK_ROOT_WIN = $QairtRoot
$env:PATH = "$QairtRoot\bin;$QairtRoot\lib;$env:PATH"

Write-Host "QAIRT_ROOT=$env:QAIRT_ROOT"
Write-Host "QNN_SDK_ROOT=$env:QNN_SDK_ROOT"
Write-Host "QNN AOT tools are now resolved from QAIRT 2.47.0.260601 when present."
