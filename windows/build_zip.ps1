# Auto-Tune Studio — Windows delivery package builder (F1.2-D).
#
# Produces the ZIP the operator unzips: the delivery scripts flattened to the
# archive root (install.bat must be the first thing they see), the sanitized
# program under payload\, the *complete* offline bundle under offline\ and an
# integrity lock the installer verifies before it copies a single file.
#
# Nothing is built from an incomplete bundle. The pinned Miniconda installer, the
# CUDA PyTorch wheels, the ordinary wheels and the bundle lock are all verified
# against windows\package-manifest.json and the bundle's own offline-lock.json
# *before* the archive is created, and the archive is checked afterwards to prove
# every offline file really is inside it.
#
# What ships is decided twice and enforced twice: the manifest declares the
# whitelist, and windows\lib\AutoTuneDelivery.psm1 additionally refuses real
# configurations, credentials, weights, datasets, logs, databases and caches
# whatever the manifest says. The build is repeatable and writes only inside its
# output directory, which therefore can never end up inside the archive.

[CmdletBinding()]
param(
    [string]$RepoRoot = '',
    [string]$OutputDir = '',
    [string]$Version = '',
    [string]$OfflineDir = ''
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\AutoTuneDelivery.psm1') -Force -DisableNameChecking

if ([string]::IsNullOrWhiteSpace($RepoRoot)) { $RepoRoot = Split-Path -Path $PSScriptRoot -Parent }
# The one spelling of the source tree this build works in, whatever spelling it
# was given (an 8.3 short name is the same directory as its long form).
$RepoRoot = Get-CanonicalPath -Path $RepoRoot

$staging = $null
try {
    $windowsSource = Join-Path $RepoRoot 'windows'
    if (-not (Test-Path (Join-Path $windowsSource 'package-manifest.json'))) {
        throw (New-DeliveryFailure -Code 'PACKAGE_ROOT_INVALID' `
                -Message ("{0} 下没有 windows\package-manifest.json，不是有效的交付源码目录。" -f $RepoRoot))
    }
    $manifest = Get-PackageManifest -PackageRoot $windowsSource
    if ([string]::IsNullOrWhiteSpace($Version)) { $Version = [string]$manifest.version }
    if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $RepoRoot 'build_output' }
    $OutputDir = Get-CanonicalPath -Path $OutputDir
    if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }
    if ([string]::IsNullOrWhiteSpace($OfflineDir)) { $OfflineDir = Join-Path $RepoRoot 'offline_cache' }

    # 1. the offline bundle is verified first: a package that cannot install
    #    offline must not be produced at all.
    $requirementsPath = Join-Path $windowsSource ([string]$manifest.runtime.pip_requirements)
    Assert-OfflineBundle -Manifest $manifest -OfflineRoot $OfflineDir `
        -RequirementsPath $requirementsPath | Out-Null
    $offlineLock = Get-OfflineLock -OfflineRoot (Get-OfflineLayout -Manifest $manifest `
            -OfflineRoot $OfflineDir).Root

    $zipPath = Join-Path $OutputDir ("AutoTuneStudio-Setup-{0}.zip" -f $Version)
    if (Test-Path $zipPath) { Remove-Item -Path $zipPath -Force }

    $staging = Join-Path $OutputDir ('.staging-' + [guid]::NewGuid().ToString('n'))
    New-Item -ItemType Directory -Force -Path $staging | Out-Null

    $scriptCount = 0
    foreach ($file in Get-ChildItem -Path $windowsSource -Recurse -File -Force) {
        $relative = Get-RelativePathUnderRoot -Root $windowsSource -Path $file.FullName
        if ([string]::IsNullOrEmpty($relative)) { continue }
        if (-not (Test-ScriptPath -RelativePath $relative -Manifest $manifest)) { continue }
        $target = Join-Path $staging $relative
        $parent = Split-Path -Path $target -Parent
        if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        Copy-Item -Path $file.FullName -Destination $target -Force
        $scriptCount = $scriptCount + 1
    }

    $payloadSource = Join-Path $RepoRoot 'auto_tune'
    $payloadCount = 0
    if (Test-Path $payloadSource) {
        foreach ($file in Get-ChildItem -Path $payloadSource -Recurse -File -Force) {
            $relative = Get-RelativePathUnderRoot -Root $RepoRoot -Path $file.FullName
            if ([string]::IsNullOrEmpty($relative)) { continue }
            if (-not (Test-PayloadPath -RelativePath $relative -Manifest $manifest)) { continue }
            $target = Join-Path (Join-Path $staging 'payload') $relative
            $parent = Split-Path -Path $target -Parent
            if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
            Copy-Item -Path $file.FullName -Destination $target -Force
            $payloadCount = $payloadCount + 1
        }
    }
    if ($payloadCount -eq 0) {
        throw (New-DeliveryFailure -Code 'PACKAGE_ROOT_INVALID' -Message '没有收集到任何程序文件，构建已终止。')
    }

    # 2. the verified offline bundle travels with the package, under the stable
    #    English directory name and never under the local source folder's name.
    $offlineTarget = Join-Path $staging ([string]$manifest.runtime.offline.directory)
    Copy-Item -Path $OfflineDir -Destination $offlineTarget -Recurse -Force

    $lock = New-PackageLock -PackageRoot $staging -Version $Version -Manifest $manifest
    Write-JsonFile -Path (Join-Path $staging 'package-manifest.lock.json') -Object $lock -Depth 8 | Out-Null

    # 3. archive it without ever holding the archive in memory, then prove the
    #    offline bundle is inside it.
    #
    #    The entries are written one at a time with a name that always uses "/":
    #    ZipFile.CreateFromDirectory derives entry names from the file system and
    #    on Windows can emit "offline\miniconda\x.exe", which is not a ZIP path —
    #    another tool would then unpack a file literally called
    #    "offline\miniconda\x.exe". The already-compressed bundle files are stored
    #    without deflating them again.
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::Open($zipPath,
        [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($item in @(Get-ChildItem -Path $staging -Recurse -File -Force |
                Sort-Object -Property FullName)) {
            $relative = Get-RelativePathUnderRoot -Root $staging -Path $item.FullName
            if ([string]::IsNullOrEmpty($relative)) { continue }
            $entryName = $relative.Replace('\', '/')
            $level = [System.IO.Compression.CompressionLevel]::Optimal
            if (@('.whl', '.exe', '.zip', '.gz') -contains $item.Extension.ToLower()) {
                $level = [System.IO.Compression.CompressionLevel]::NoCompression
            }
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $item.FullName, $entryName, $level)
        }
    } finally {
        $archive.Dispose()
    }

    $expectedOffline = @($offlineLock.files | ForEach-Object {
            ([string]$manifest.runtime.offline.directory + '/' + [string]$_.path).Replace('\', '/')
        })
    $missing = New-Object System.Collections.ArrayList
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $entryNames = @{}
        foreach ($entry in $archive.Entries) { $entryNames[$entry.FullName] = $true }
        foreach ($name in $expectedOffline) {
            if (-not $entryNames.ContainsKey($name)) { [void]$missing.Add($name) }
        }
        foreach ($required in @('install.bat', 'start.bat', 'upgrade.bat', 'uninstall.bat')) {
            if (-not $entryNames.ContainsKey($required)) { [void]$missing.Add($required) }
        }
    } finally {
        $archive.Dispose()
    }
    if (@($missing).Count -gt 0) {
        Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
        throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                -Message ("生成的安装包缺少以下文件，已删除：{0}" -f (@($missing) -join '、')))
    }

    Write-Host ("[package] 版本 {0}" -f $Version)
    Write-Host ("[package] 交付脚本 {0} 个，程序文件 {1} 个，离线依赖 {2} 个文件" -f `
        $scriptCount, $payloadCount, @($offlineLock.files).Count)
    Write-Host ("[package] 已生成 {0}" -f $zipPath)
    Write-Host "[package] 解压后请双击 install.bat（安装全程离线，无需联网）。"
    exit 0
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    Write-Host ""
    Write-Host ("[package] ERROR {0}: {1}" -f $info.Code, $info.Message)
    exit 1
} finally {
    if ($staging -and (Test-Path $staging)) {
        Remove-Item -Path $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}
