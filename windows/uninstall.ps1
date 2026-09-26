# Auto-Tune Studio — Windows uninstall entry point (F1.2-D).
#
# Default behaviour removes the program and the private runtime and *keeps* the
# user data. Deleting the data requires the explicit pair --remove-data --confirm
# and is refused for any target that is not the controlled data directory.

[CmdletBinding()]
param(
    [string]$LocalAppData = $env:LOCALAPPDATA,
    [switch]$RemoveData,
    [switch]$Confirm,
    [string]$InstallRoot = ''
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

$exitCode = 1
try {
    # Only the installation this entry point belongs to is ever removed.
    $root = Resolve-DeliveryInstallRoot -InstallRoot $InstallRoot -EntryRoot $PSScriptRoot `
        -LocalAppData $LocalAppData
    if ([string]::IsNullOrWhiteSpace($root)) {
        $layout = Get-DeliveryLayout -LocalAppData $LocalAppData
    } else {
        $layout = Get-DeliveryLayout -InstallRoot $root -LocalAppData $LocalAppData
    }

    Write-Host ""
    Write-Host "Auto-Tune Studio 卸载程序"
    Write-Host ("安装目录：{0}" -f $layout.Root)
    if ($RemoveData) {
        Write-Host "模式：删除程序、私有运行环境以及用户数据（--remove-data --confirm）" -ForegroundColor Yellow
    } else {
        Write-Host "模式：删除程序与私有运行环境，保留用户数据" -ForegroundColor Yellow
    }
    Write-Host ""

    $result = Invoke-Uninstall -Layout $layout -RemoveData:$RemoveData -Confirm:$Confirm

    if ($result.Ok) {
        Write-Host "卸载完成。" -ForegroundColor Green
        Write-Host $result.Message
        if (-not $result.DataRemoved) {
            Write-Host ("如需一并删除用户数据，请运行：uninstall.bat --remove-data --confirm")
        }
        Write-Host ("保留说明：{0}" -f $result.DataNoteFile)
        $exitCode = 0
    } else {
        Write-Host ""
        Write-Host ("卸载未完成：{0}" -f $result.ErrorCode) -ForegroundColor Red
        Write-Host $result.Message
        $exitCode = 1
    }
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("卸载未完成：{0}" -f $info.Code) -ForegroundColor Red
    Write-Host $info.Message
    $exitCode = 1
}

exit $exitCode
