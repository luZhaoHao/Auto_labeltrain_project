# Auto-Tune Studio — Windows installer entry point (F1.2-D).
#
# Everything that decides *what* happens lives in lib\AutoTuneDelivery.psm1; this
# file resolves the installation directory, calls the install, prints the
# operator-facing result and turns the outcome into an exit code. It runs as the
# ordinary user: no administrator rights, no registry entry, no PATH change.
#
# The destination is chosen once, on the first installation. Later runs — and
# start, upgrade and uninstall — read the directory the installation recorded,
# so a second copy is never created somewhere else by accident.
#
# This is the only script that asks the operator a question, so it is the only
# one whose entry point (install.bat) must launch PowerShell without
# -NonInteractive: that switch turns Read-Host into a terminating error.

[CmdletBinding()]
param(
    [string]$LocalAppData = $env:LOCALAPPDATA,
    [int]$Port = 0,
    [string]$InstallRoot = '',
    [switch]$AcceptRecommended
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

$exitCode = 1
try {
    $manifest = Get-PackageManifest -PackageRoot $PSScriptRoot
    $requestedPort = $Port
    if ($requestedPort -lt 1 -or $requestedPort -gt 65535) {
        $requestedPort = [int]$manifest.install.default_port
    }
    if ($requestedPort -lt 1 -or $requestedPort -gt 65535) { $requestedPort = 8000 }

    $plan = Get-InstallPromptPlan -Requested $InstallRoot -LocalAppData $LocalAppData `
        -AcceptRecommended ([bool]$AcceptRecommended) `
        -NonInteractiveVariable $env:AUTO_TUNE_NONINTERACTIVE `
        -InputRedirected ([Console]::IsInputRedirected)
    $destination = [string]$plan.Destination
    if ([string]::IsNullOrWhiteSpace($destination)) {
        # First installation: propose a roomy disk that is not the system one.
        $recommended = Get-RecommendedInstallRoot
        if ($plan.Ask) {
            Write-Host ""
            Write-Host "Auto-Tune Studio 安装程序"
            Write-Host "首次安装，请选择安装目录。"
            Write-Host ("建议目录：{0}" -f $recommended)
            Write-Host "直接回车使用建议目录，或输入其它本地绝对路径（例如 E:\AutoTuneStudio）。"
            $answer = Read-Host "安装目录"
            if ([string]::IsNullOrWhiteSpace($answer)) { $answer = $recommended }
            $destination = $answer
        } else {
            $destination = $recommended
        }
    }

    $destination = Assert-InstallRootAllowed -Root $destination
    $advice = Get-InstallDestinationAdvice -Root $destination `
        -RequiredFreeBytes (Get-RequiredFreeBytes -Manifest $manifest)
    $layout = Get-DeliveryLayout -InstallRoot $destination -LocalAppData $LocalAppData `
        -Port $requestedPort

    Write-Host ""
    Write-Host "Auto-Tune Studio 安装程序"
    Write-Host ("安装目录：{0}" -f $layout.Root)
    Write-Host ("用户数据：{0}" -f $layout.Data)
    if (-not [string]::IsNullOrWhiteSpace([string]$advice.Message)) {
        Write-Host $advice.Message -ForegroundColor Yellow
    }
    Write-Host ""

    $result = Invoke-Install -Layout $layout -PackageRoot $PSScriptRoot

    if ($result.Ok) {
        Write-Host ""
        Write-Host "安装完成。" -ForegroundColor Green
        Write-Host ("  程序：      {0}" -f $layout.App)
        Write-Host ("  运行环境：  {0}" -f $layout.Interpreter)
        Write-Host ("  用户数据：  {0}" -f $layout.Data)
        Write-Host ("  服务地址：  http://127.0.0.1:{0}/" -f $layout.Port)
        Write-Host ""
        Write-Host "本次解压的安装包目录可以删除，安装目录已经独立可用："
        Write-Host ("  启动：      {0}" -f $layout.StartBat)
        Write-Host ("  卸载：      {0}" -f $layout.UninstallBat)
        if ($result.ShortcutCreated) {
            Write-Host ("  桌面快捷方式：{0}" -f $result.Shortcut)
        }
        Write-Host ""
        Write-Host "请双击安装目录下的 start.bat（或桌面快捷方式）启动 Auto Tune Studio。"
        $exitCode = 0
    } else {
        Write-Host ""
        Write-Host ("安装失败：{0}" -f $result.ErrorCode) -ForegroundColor Red
        Write-Host $result.Message
        Write-Host ""
        Write-Host ("详细日志（已脱敏）：{0}" -f (Join-Path $layout.Logs 'install.log'))
        Write-Host "安装可以重复执行：修复问题后再次双击 install.bat 会从断点继续。"
        $exitCode = 1
    }
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("安装失败：{0}" -f $info.Code) -ForegroundColor Red
    Write-Host $info.Message
    Write-Host ""
    Write-Host "请确认安装包完整：package-manifest.json 与 package-manifest.lock.json 必须与本文件在同一目录。"
    $exitCode = 1
}

exit $exitCode
