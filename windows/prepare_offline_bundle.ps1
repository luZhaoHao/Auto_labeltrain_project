# Auto-Tune Studio — offline dependency bundle builder (F1.2-D).
#
# Build machine only. It turns the three files the operator downloaded once
# (Miniconda, the CUDA torch wheel, the CUDA torchvision wheel) plus every
# ordinary runtime wheel into the ``offline\`` bundle the delivery ZIP ships:
#
#   offline\miniconda\<miniconda>.exe
#   offline\wheelhouse\*.whl
#   offline\wheelhouse\offline-lock.json
#
# Everything that verifies, downloads, names a missing package or writes the
# lock lives in lib\AutoTuneDelivery.psm1, so this file only locates the source
# folder, checks the interpreter it is told to use and reports the outcome.
#
# The download is the one step that needs a network, and it is deliberately
# narrow: Windows x64, CPython 3.10, binary wheels only (--only-binary=:all:).
# A failure never removes a file that already verified — re-running the script
# continues where it stopped.

[CmdletBinding()]
param(
    [string]$RepoRoot = '',
    [string]$DependencySource = '',
    [string]$OutputDir = '',
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

if ([string]::IsNullOrWhiteSpace($RepoRoot)) { $RepoRoot = Split-Path -Path $PSScriptRoot -Parent }
if ([string]::IsNullOrWhiteSpace($DependencySource)) { $DependencySource = Join-Path $RepoRoot '依赖' }
if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $RepoRoot 'offline_cache' }

$exitCode = 1
try {
    $interpreter = $Python
    if ([string]::IsNullOrWhiteSpace($interpreter)) {
        $command = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $command) {
            Write-Host ""
            Write-Host "请用 -Python 指定用于下载依赖的 Python 3.10 解释器，例如：" -ForegroundColor Red
            Write-Host '  powershell -File prepare_offline_bundle.ps1 -Python "D:\anaconda3\envs\py310\python.exe"'
            exit 1
        }
        $interpreter = [string]$command.Source
        Write-Host ("未指定 -Python，使用 PATH 中的解释器：{0}" -f $interpreter)
    }

    $version = ''
    try {
        $version = ((& $interpreter -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>&1 |
                Out-String) -replace "`r?`n", '').Trim()
    } catch {
        $version = ''
    }
    if ($version -ne '3.10') {
        throw (New-DeliveryFailure -Code 'OFFLINE_WHEELHOUSE_INCOMPLETE' `
                -Message ("下载依赖必须使用 Python 3.10 解释器（当前：{0}）。请用 -Python 指定。" -f `
                    $(if ([string]::IsNullOrWhiteSpace($version)) { '无法运行' } else { $version })))
    }

    Write-Host ""
    Write-Host "Auto-Tune Studio 离线依赖准备"
    Write-Host ("源码目录：  {0}" -f $RepoRoot)
    Write-Host ("离线源目录：{0}" -f $DependencySource)
    Write-Host ("输出目录：  {0}" -f $OutputDir)
    Write-Host ("解释器：    {0}" -f $interpreter)
    Write-Host ""

    $result = Prepare-OfflineBundle -RepoRoot $RepoRoot -DependencySource $DependencySource `
        -OutputDir $OutputDir -Python $interpreter

    Write-Host ("离线依赖准备完成，共 {0} 个文件。" -f @($result.Files).Count) -ForegroundColor Green
    Write-Host ("离线锁：{0}" -f (Join-Path $OutputDir 'offline-lock.json'))
    foreach ($file in @($result.Files)) { Write-Host ("  {0}" -f $file) }
    $exitCode = 0
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("离线依赖准备失败：{0}" -f $info.Code) -ForegroundColor Red
    Write-Host $info.Message
    if ($null -ne $info.Detail) {
        $missing = @()
        if ($null -ne $info.Detail.Missing) { $missing = @($info.Detail.Missing) }
        if (@($missing).Count -gt 0) {
            Write-Host ""
            Write-Host "缺少的依赖（请从 PyPI 下载对应 wheel，不要修改锁定版本）："
            foreach ($entry in $missing) {
                Write-Host ("  {0}=={1}" -f $entry.Name, $entry.Version)
                foreach ($wheel in @($entry.Wheels)) { Write-Host ("      {0}" -f $wheel) }
                Write-Host ("      {0}" -f $entry.Url)
            }
        }
        if ($null -ne $info.Detail.MissingFromPip) {
            $fromPip = @($info.Detail.MissingFromPip)
            if (@($fromPip).Count -gt 0) {
                Write-Host ""
                Write-Host "pip 报告缺少的依赖（传递依赖也算，请从 PyPI 获取对应 wheel，不要修改锁定版本）："
                foreach ($spec in $fromPip) { Write-Host ("  {0}" -f $spec) }
            }
        }
        if ($null -ne $info.Detail.PreservedFiles) {
            Write-Host ("已保留已校验的文件 {0} 个。" -f @($info.Detail.PreservedFiles).Count)
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$info.Detail.Hint)) {
            Write-Host ([string]$info.Detail.Hint)
        }
    }
    $exitCode = 1
}

exit $exitCode
