# Auto-Tune Studio — Windows upgrade entry point (F1.2-D).
#
# Run this from a *new* package folder: the program files are replaced by the
# ones in this folder, the private runtime is only touched when the dependency
# lock changed, and every user file (configuration, history, SQLite index,
# weights, training results, installer cache) is left exactly as it was.

[CmdletBinding()]
param(
    [string]$LocalAppData = $env:LOCALAPPDATA,
    [string]$InstallRoot = ''
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

$exitCode = 1
try {
    # This file runs from the *new* package, so $PSScriptRoot names the package,
    # not the installation: the directory to upgrade is the one the installation
    # recorded when it was installed. It is never chosen again here.
    $root = Resolve-DeliveryInstallRoot -InstallRoot $InstallRoot -LocalAppData $LocalAppData
    if ([string]::IsNullOrWhiteSpace($root)) {
        $layout = Get-DeliveryLayout -LocalAppData $LocalAppData
    } else {
        $layout = Get-DeliveryLayout -InstallRoot $root -LocalAppData $LocalAppData
    }
    $manifest = Get-PackageManifest -PackageRoot $PSScriptRoot

    Write-Host ""
    Write-Host ("Auto-Tune Studio 升级程序：将升级到 {0}" -f $manifest.version)
    Write-Host ("安装目录：{0}" -f $layout.Root)
    Write-Host ""

    $result = Invoke-Upgrade -Layout $layout -PackageRoot $PSScriptRoot

    if ($result.Ok) {
        Write-Host ("升级完成，当前版本 {0}。" -f $result.Version) -ForegroundColor Green
        if ($result.RuntimeUpdated) {
            Write-Host "依赖清单有变化，私有运行环境已按新清单更新。"
        } else {
            Write-Host "依赖清单未变化，私有运行环境保持不变。"
        }
        Write-Host ("用户数据未改动：{0}" -f $layout.Data)
        $exitCode = 0
    } else {
        Write-Host ""
        Write-Host ("升级失败：{0}" -f $result.ErrorCode) -ForegroundColor Red
        Write-Host $result.Message
        Write-Host "上一个版本仍然可用；无需重新安装，修复问题后可再次运行 upgrade.bat。"
        Write-Host ("升级日志（已脱敏）：{0}" -f (Join-Path $layout.Logs 'upgrade.log'))
        $exitCode = 1
    }
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("升级失败：{0}" -f $info.Code) -ForegroundColor Red
    Write-Host $info.Message
    Write-Host "请确认安装包完整，并且已经通过 install.bat 完成过一次安装。"
    $exitCode = 1
}

exit $exitCode
