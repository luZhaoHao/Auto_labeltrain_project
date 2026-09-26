# Auto-Tune Studio — Windows launcher entry point (F1.2-D).
#
# The start gates live in lib\AutoTuneDelivery.psm1. This file only resolves the
# layout (so the launcher works from any directory) and reports the outcome.
# There is deliberately no switch that skips the GPU preflight or that would let
# the service run on the CPU.

[CmdletBinding()]
param(
    [string]$LocalAppData = $env:LOCALAPPDATA,
    [string]$Port = '',
    [string]$InstallRoot = ''
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

$exitCode = 1
try {
    # The installed launcher lives inside the installation it starts, so its own
    # directory wins; the record the install left behind covers the other entry
    # points. Nothing here asks the operator a question a second time.
    $root = Resolve-DeliveryInstallRoot -InstallRoot $InstallRoot -EntryRoot $PSScriptRoot `
        -LocalAppData $LocalAppData
    if ([string]::IsNullOrWhiteSpace($root)) {
        $layout = Get-DeliveryLayout -LocalAppData $LocalAppData
    } else {
        $layout = Get-DeliveryLayout -InstallRoot $root -LocalAppData $LocalAppData
    }
    $result = Start-Studio -Layout $layout -EnvPort $Port

    if ($result.Ok) {
        if ($result.AlreadyRunning) {
            Write-Host "Auto Tune Studio 已经在运行，已为您打开浏览器。"
        } else {
            Write-Host ("Auto Tune Studio 已启动：http://127.0.0.1:{0}/" -f $result.Port) -ForegroundColor Green
        }
        Write-Host ("运行日志：{0}" -f (Join-Path $layout.Logs 'start.log'))
        $exitCode = 0
    } else {
        # A process that never answers the health check is a failed start, not a
        # slow one: the exit code says so, and nothing was left running.
        Write-Host ""
        Write-Host ("启动失败：{0}" -f $result.ErrorCode) -ForegroundColor Red
        Write-Host $result.Message
        Write-Host ("运行日志（已脱敏）：{0}" -f (Join-Path $layout.Logs 'start.log'))
        $exitCode = 1
    }
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("启动失败：{0}" -f $info.Code) -ForegroundColor Red
    Write-Host $info.Message
    $exitCode = 1
}

exit $exitCode
