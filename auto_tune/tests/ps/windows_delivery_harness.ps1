# F1.2-D: drives the real Windows delivery module with injected seams.
#
# The delivery scripts cannot be exercised on a laptop: installing them for real
# would run a Python distribution installer, write into a chosen install root and
# open a browser. This harness imports the *real* module and replaces only the
# boundaries the operator's machine provides — the process runner and the probes
# (disk, driver, port, PID, health) — so the logic under test (layout, offline
# bundle verification, staging, state, safety) is the same code the operator
# runs, minus the hardware.
#
# The offline installation is *fully offline*: the Miniconda installer, the CUDA
# PyTorch wheels and every ordinary wheel travel inside the package under
# ``offline\`` and are installed with ``pip --no-index --find-links``. The
# harness therefore has no downloader seam at all — instead it installs a global
# ``Invoke-HttpsDownload`` trap that records and refuses any call, so a delivery
# that tried to reach the network would be caught and counted.
#
# Output: one line ``##RESULT## {json}``. Everything else the scripts print is
# ignored by the caller. No scenario touches the network, the registry, a real
# conda installation or a real browser.

param(
    [Parameter(Mandatory = $true)][string]$Scenario,
    [Parameter(Mandatory = $true)][string]$ModulePath,
    [Parameter(Mandatory = $true)][string]$WorkRoot,
    [string]$DangerTarget = '',
    [int]$Port = 0,
    [string]$EnvPort = '',
    [string]$Desktop = '',
    [string]$Variant = '',
    [string]$InstallRoot = '',
    [string]$DependencySource = '',
    [string]$TargetRoot = ''
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

# Each scenario gets its own tree: scenarios must never inherit another
# scenario's half-installed state.
$WorkRoot = Join-Path $WorkRoot $Scenario
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

# Two traps when working on the scenarios below (both cost real debugging time):
#
#   * Pass a *fresh* -WorkRoot on every run. A scenario that starts from a clean
#     installation reads the tree it is given, so re-running one against an
#     earlier run's leftovers silently tests the wrong starting point (a runtime
#     install that no longer happens, a state that is already complete). The
#     pytest suites get this for free from tmp_path_factory; an ad-hoc run does
#     not.
#   * Build fixtures through the LONG path and reach them through the SHORT one.
#     New-Item -ItemType Directory -Force on a multi-level path *below* an 8.3
#     root is refused by PowerShell (".NET: the directory specified, 'x', is not
#     a subdirectory of 'y'"), so create the tree normally and ask
#     Get-ShortDirectoryPath for the spelling to test with.

if ([string]::IsNullOrWhiteSpace($Desktop)) { $Desktop = Join-Path $WorkRoot 'desktop' }
New-Item -ItemType Directory -Force -Path $Desktop | Out-Null

# ``windows\`` next to the module: the synthetic package ships the *real*
# launcher scripts, so the installed entry points can be executed for real.
$script:WindowsSource = Split-Path -Path (Split-Path -Path (Resolve-Path $ModulePath).Path -Parent) -Parent

# The harness hashes with the CLR directly. It must never call Get-FileHash:
# the cmdlet is hidden below to reproduce a machine where PowerShell 7's
# Microsoft.PowerShell.Utility wins over the 5.1 one, and a test driver that
# still used it would fail for the wrong reason.
function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { $bytes = $sha.ComputeHash($stream) } finally { $sha.Dispose() }
    } finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return '' }
    return (Get-Sha256Hex -Path $Path)
}

# PowerShell 7 installs a Microsoft.PowerShell.Utility that a 5.1 host can load
# by mistake, and then Get-FileHash simply does not exist. The delivery must not
# depend on it, so the harness hides it for every scenario.
function global:Get-FileHash {
    param([string]$Path, [string]$Algorithm)
    throw "The term 'Get-FileHash' is not recognized as the name of a cmdlet (simulated PowerShell 7 module conflict)."
}

# The installation is offline: nothing may download anything. This trap records
# every attempt to fetch and refuses it, so ``networkCalls`` is an observation
# of the real code rather than a claim about it.
$script:NetworkCalls = New-Object System.Collections.ArrayList
function global:Invoke-HttpsDownload {
    param([string]$Url, [string]$Destination)
    [void]$script:NetworkCalls.Add([string]$Url)
    throw "the offline installation must never download anything (simulated disconnected machine)"
}

try {
    Import-Module $ModulePath -Force -DisableNameChecking
} catch {
    Write-Output ('##RESULT## ' + (@{
                scenario = $Scenario; ok = $false; error_code = 'HARNESS_ERROR'
                error_message = "the delivery module could not be imported: $($_.Exception.Message)"
                facts = @{}; calls = @()
            } | ConvertTo-Json -Depth 12 -Compress))
    exit 0
}

$script:Calls = New-Object System.Collections.ArrayList
$script:EnvMode = 'ok'
$script:PreflightMode = 'ok'
$script:HealthMode = 'ok'
$script:ProcessAlive = $true
$script:PortInUse = $false
$script:RuntimeInstallCalls = 0
$script:BrowserOpens = New-Object System.Collections.ArrayList
# What the delivery tree looked like at the moment a check ran. The installer
# asks the private runtime to prove itself *before* the program payload, the
# configuration and the persistent directories exist, so these are the facts
# that catch a runtime check which needs any of them.
$script:ObservableLayout = $null
$script:RuntimeCheckFacts = $null
$script:StartupCheckFacts = $null
# The offline core files may be written large to show the hashing streams; the
# default fixture keeps them tiny.
$script:OfflineCoreSize = 4096
# What the stand-in pip does during the bundle preparation.
$script:PipMode = 'ok'
$script:PypiDir = Join-Path $WorkRoot 'pypi'
# A credential and a local directory that only ever appear in pip's own output:
# no report may carry them.
$script:PrepareToken = 'sk-live-abcdef1234567890'
$script:PrepareSecrets = @($script:PrepareToken, 'build-agent', 'proxy.internal')

function Add-Call([string]$Name, $Data) {
    [void]$script:Calls.Add(@{ name = $Name; data = $Data })
}

function Get-CallNames() {
    return @($script:Calls | ForEach-Object { $_.name })
}

function Get-CallData([string]$Name) {
    return @($script:Calls | Where-Object { $_.name -eq $Name } | ForEach-Object { $_.data })
}

function Get-CheckLayoutFacts {
    # Whether the delivered tree already carried the program payload and the
    # operator's configuration when a delivery check ran.
    param($Layout)
    return @{
        config_present  = (Test-Path -PathType Leaf $Layout.ConfigPath)
        payload_present = (Test-Path -PathType Leaf (Join-Path $Layout.App 'auto_tune\delivery\preflight.py'))
    }
}

# The three pinned core files and the two ordinary dependency wheels the
# synthetic bundle carries. The names are the real ones so the wheel-format
# rules are exercised on filenames that could really occur.
$script:OfflineMiniconda = 'Miniconda3-test-Windows-x86_64.exe'
$script:OfflineTorch = 'torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
$script:OfflineTorchvision = 'torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl'
$script:OfflineDependencies = @('fastapi-0.139.2-py3-none-any.whl', 'uvicorn-0.51.0-py3-none-any.whl')

function Write-FillerFile {
    # A file of a known size with a content that depends on the seed, so two
    # rewritten files of the same size still differ.
    param([Parameter(Mandatory = $true)][string]$Path, [long]$Size = 4096, [int]$Seed = 1)
    $directory = Split-Path -Path $Path -Parent
    if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    $bytes = New-Object byte[] ([int]$Size)
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        $bytes[$i] = [byte](($i + $Seed) % 251)
    }
    [System.IO.File]::WriteAllBytes($Path, $bytes)
    return $Path
}

function Get-TestOfflinePaths {
    param(
        [string]$PackageRoot = '',
        [string]$OfflineRoot = ''
    )
    if ([string]::IsNullOrWhiteSpace($OfflineRoot)) { $OfflineRoot = Join-Path $PackageRoot 'offline' }
    $wheelhouse = Join-Path $OfflineRoot 'wheelhouse'
    return @{
        Root        = $OfflineRoot
        Miniconda   = Join-Path (Join-Path $OfflineRoot 'miniconda') $script:OfflineMiniconda
        Wheelhouse  = $wheelhouse
        Lock        = Join-Path $OfflineRoot 'offline-lock.json'
        Torch       = Join-Path $wheelhouse $script:OfflineTorch
        Torchvision = Join-Path $wheelhouse $script:OfflineTorchvision
    }
}

function Get-TestDependencyWheels {
    # The ordinary wheels of the bundle: everything except the three core files
    # the operator downloaded once. Callers wrap the call in @(...).
    param([Parameter(Mandatory = $true)][string]$OfflineRoot)
    $paths = Get-TestOfflinePaths -OfflineRoot $OfflineRoot
    return @(Get-ChildItem -Path $paths.Wheelhouse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $script:OfflineTorch -and $_.Name -ne $script:OfflineTorchvision } |
        Sort-Object -Property Name | ForEach-Object { $_.FullName })
}

function Write-TestOfflineLock {
    # The lock is produced by the shipping generator, so the fixture cannot
    # drift from the format the installer verifies.
    param(
        [Parameter(Mandatory = $true)][string]$ManifestRoot,
        [Parameter(Mandatory = $true)][string]$OfflineRoot,
        [string]$RequirementsPath = ''
    )
    $manifest = Get-PackageManifest -PackageRoot $ManifestRoot
    if ([string]::IsNullOrWhiteSpace($RequirementsPath)) {
        $RequirementsPath = Join-Path $ManifestRoot ([string]$manifest.runtime.pip_requirements)
    }
    $lock = New-OfflineLock -OfflineRoot $OfflineRoot -Manifest $manifest `
        -RequirementsPath $RequirementsPath
    Write-JsonFile -Path (Join-Path $OfflineRoot 'offline-lock.json') -Object $lock | Out-Null
}

function New-TestOfflineBundle {
    # The three core files plus one stand-in wheel for every pinned dependency
    # (torch and torchvision are the core files, not wheelhouse dependencies).
    #
    # ``-DependencySeed`` changes the *bytes* of the ordinary wheels without
    # touching their names or the requirements file: the bundle a package ships
    # really changes while its dependency lock stays identical.
    param(
        [Parameter(Mandatory = $true)][string]$OfflineRoot,
        [Parameter(Mandatory = $true)][string]$RequirementsPath,
        [long]$CoreSize = 4096,
        [int]$DependencySeed = 31
    )
    $paths = Get-TestOfflinePaths -OfflineRoot $OfflineRoot
    [void](Write-FillerFile -Path $paths.Miniconda -Size $CoreSize -Seed 11)
    [void](Write-FillerFile -Path $paths.Torch -Size $CoreSize -Seed 17)
    [void](Write-FillerFile -Path $paths.Torchvision -Size ([math]::Max(64, [int]($CoreSize / 8))) -Seed 23)
    $seed = $DependencySeed
    if (Test-Path -PathType Leaf $RequirementsPath) {
        foreach ($line in @(Get-Content -Path $RequirementsPath -Encoding UTF8)) {
            $entry = ([string]$line).Trim()
            if ([string]::IsNullOrWhiteSpace($entry) -or $entry.StartsWith('#')) { continue }
            $match = [regex]::Match($entry, '^([A-Za-z0-9._-]+)==([^\s;]+)$')
            if (-not $match.Success) { continue }
            $name = (Get-NormalizedPackageName -Name $match.Groups[1].Value)
            if ($name -eq 'torch' -or $name -eq 'torchvision') { continue }
            $wheel = Join-Path $paths.Wheelhouse ("{0}-{1}-py3-none-any.whl" -f $name, $match.Groups[2].Value)
            [void](Write-FillerFile -Path $wheel -Size 1024 -Seed $seed)
            $seed = $seed + 7
        }
    }
}

function Copy-LockWithFiles {
    # The same lock with a hand-edited file list, so one defect can be planted at
    # a time without changing the rest of the record.
    param([Parameter(Mandatory = $true)]$Lock, [Parameter(Mandatory = $true)]$Files)
    return [ordered]@{
        schema_version      = [string]$Lock.schema_version
        product             = [string]$Lock.product
        python_version      = [string]$Lock.python_version
        platform            = [string]$Lock.platform
        requirements_file   = [string]$Lock.requirements_file
        requirements_sha256 = [string]$Lock.requirements_sha256
        files               = @($Files)
    }
}

function New-PrepareFixture {
    # A build machine: ``windows\`` with the accepted manifest and dependency
    # lock, the local ``依赖\`` folder holding the three downloaded core files,
    # and a stand-in index directory the injected pip runner copies from.
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$Variant = ''
    )
    $repo = Join-Path $Root 'repo'
    $windows = Join-Path $repo 'windows'
    $deps = Join-Path $repo '依赖'
    $pypi = Join-Path $Root 'pypi'
    foreach ($directory in @($windows, $deps, $pypi)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    $minicondaPath = Write-FillerFile -Path (Join-Path $deps $script:OfflineMiniconda) -Size 4096 -Seed 11
    $torchPath = Write-FillerFile -Path (Join-Path $deps $script:OfflineTorch) -Size 8192 -Seed 17
    $torchvisionPath = Write-FillerFile -Path (Join-Path $deps $script:OfflineTorchvision) -Size 2048 -Seed 23
    $minicondaSha = Get-Sha256Hex -Path $minicondaPath
    $torchSha = Get-Sha256Hex -Path $torchPath
    $torchvisionSha = Get-Sha256Hex -Path $torchvisionPath

    # the stand-in index: the ordinary wheels pip would download
    $seed = 31
    foreach ($name in $script:OfflineDependencies) {
        [void](Write-FillerFile -Path (Join-Path $pypi $name) -Size 1024 -Seed $seed)
        $seed = $seed + 7
    }

    if ($Variant -eq 'miniconda-missing') { Remove-Item -Path $minicondaPath -Force }
    if ($Variant -eq 'source-hash-mismatch') {
        [void](Write-FillerFile -Path $torchPath -Size 8192 -Seed 71)
    }

    Set-Content -Path (Join-Path $windows 'requirements-windows.lock.txt') -Encoding UTF8 `
        -Value "fastapi==0.139.2`nuvicorn==0.51.0`n"
    $lockSha = Get-Sha256Hex -Path (Join-Path $windows 'requirements-windows.lock.txt')
    $manifest = New-TestManifest -MinicondaSha $minicondaSha -MinicondaSize 4096 `
        -TorchSha $torchSha -TorchSize 8192 `
        -TorchvisionSha $torchvisionSha -TorchvisionSize 2048 -LockSha $lockSha
    Write-JsonFile -Path (Join-Path $windows 'package-manifest.json') -Object $manifest

    $script:PypiDir = $pypi
    return @{ RepoRoot = $repo; DependencySource = $deps; PypiDir = $pypi }
}

function New-FakePipRunner {
    # Models ``pip download``: the wheels come from the stand-in index into
    # ``--dest`` unless the variant asks pip to fail.
    return {
        param($FilePath, $Arguments, $WorkingDirectory, $Environment, $LogFile)
        Add-Call 'run' @{ file = $FilePath; args = @($Arguments); cwd = $WorkingDirectory
            env = $Environment }
        $arguments = @($Arguments)
        $dest = ''
        for ($i = 0; $i -lt $arguments.Count; $i++) {
            if ($arguments[$i] -eq '--dest' -and ($i + 1) -lt $arguments.Count) {
                $dest = [string]$arguments[$i + 1]
            }
        }
        if ($dest) {
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            $available = @(Get-ChildItem -Path $script:PypiDir -File -ErrorAction SilentlyContinue)
            if ($script:PipMode -eq 'missing-wheel') {
                $available = @($available | Where-Object { $_.Name -notlike 'uvicorn-*' })
            }
            foreach ($item in $available) {
                Copy-Item -Path $item.FullName -Destination (Join-Path $dest $item.Name) -Force
            }
        }
        if ($script:PipMode -eq 'missing-wheel') {
            return @{ ExitCode = 1; StdOut = ''
                StdErr = 'ERROR: Could not find a version that satisfies the requirement uvicorn==0.51.0' }
        }
        if ($script:PipMode -eq 'transitive-missing') {
            # Every *pinned* wheel resolved; pip failed on a dependency the
            # requirements file never names. The rest of the output is what must
            # never reach the operator: a local build path and an index URL with
            # credentials.
            $stdErr = @(
                ('Looking in indexes: https://build:{0}@proxy.internal/simple' -f $script:PrepareToken)
                'Using cached wheel: D:\build-agent\pip-cache\http-v2\contourpy-1.3.2-cp310-cp310-win_amd64.whl'
                'ERROR: Could not find a version that satisfies the requirement numpy>=1.23.5 (from contourpy) (from versions: none)'
                'ERROR: No matching distribution found for numpy>=1.23.5'
                'ERROR: Could not find a version that satisfies the requirement contourpy==1.3.2 (from fastapi)'
                'WARNING: You are using pip version 23.0.1'
            ) -join "`n"
            return @{ ExitCode = 1; StdOut = ''; StdErr = $stdErr }
        }
        if ($script:PipMode -eq 'transitive-missing-many') {
            # More missing packages than the report is allowed to carry, plus one
            # line shaped like a requirement but carrying a path and a token.
            $lines = New-Object System.Collections.ArrayList
            foreach ($name in @('numpy', 'contourpy', 'pillow', 'scipy', 'pandas', 'pyarrow')) {
                [void]$lines.Add(('ERROR: No matching distribution found for {0}==1.0.0' -f $name))
            }
            [void]$lines.Add('ERROR: No matching distribution found for D:\build-agent\cache\sk-live-abcdef123456')
            return @{ ExitCode = 1; StdOut = ''; StdErr = ($lines -join "`n") }
        }
        if ($script:PipMode -eq 'transient') {
            return @{ ExitCode = 1; StdOut = ''; StdErr = 'ERROR: connection reset by peer' }
        }
        return @{ ExitCode = 0; StdOut = ''; StdErr = '' }
    }
}

function New-TestManifest {
    # One shape for every synthetic package, so the fixture and the delivery
    # agreement tests cannot drift apart.
    param(
        [string]$Version = '1.0.0',
        [long]$RequiredFreeBytes = 1073741824,
        [string]$MinicondaSha = '',
        [long]$MinicondaSize = 0,
        [string]$TorchSha = '',
        [long]$TorchSize = 0,
        [string]$TorchvisionSha = '',
        [long]$TorchvisionSize = 0,
        [string]$LockSha = ''
    )
    return [ordered]@{
        schema_version = '1.0'
        product        = 'auto-tune-studio'
        version        = $Version
        python_version = '3.10'
        package        = [ordered]@{
            scripts_include = @('*.bat', '*.ps1', 'lib/**', 'package-manifest.json',
                'requirements-windows.lock.txt')
            scripts_exclude = @('build_zip.ps1', 'prepare_offline_bundle.ps1')
        }
        payload        = [ordered]@{
            include = @('auto_tune/')
            exclude = @('auto_tune/tests/', 'auto_tune/scripts/', 'auto_tune/evaluation/',
                'auto_tune/docs/', 'auto_tune/requirements.txt', '**/*.md')
        }
        install        = [ordered]@{
            required_free_bytes = $RequiredFreeBytes
            default_port        = 8000
        }
        runtime        = [ordered]@{
            conda_installer = [ordered]@{
                url       = "https://repo.anaconda.com/miniconda/$script:OfflineMiniconda"
                file_name = $script:OfflineMiniconda
                sha256    = $MinicondaSha
                size      = $MinicondaSize
            }
            torch                = [ordered]@{
                packages  = @('torch==2.5.1', 'torchvision==0.20.1')
                index_url = 'https://download.pytorch.org/whl/cu121'
                wheels    = @(
                    [ordered]@{
                        file_name = $script:OfflineTorch
                        version   = '2.5.1+cu121'
                        sha256    = $TorchSha
                        size      = $TorchSize
                        purpose   = 'cuda-torch'
                    },
                    [ordered]@{
                        file_name = $script:OfflineTorchvision
                        version   = '0.20.1+cu121'
                        sha256    = $TorchvisionSha
                        size      = $TorchvisionSize
                        purpose   = 'cuda-torchvision'
                    }
                )
            }
            offline              = [ordered]@{
                directory            = 'offline'
                miniconda_directory  = 'miniconda'
                wheelhouse_directory = 'wheelhouse'
                lock_file            = 'offline-lock.json'
            }
            pip_requirements        = 'requirements-windows.lock.txt'
            pip_requirements_sha256 = $LockSha
        }
    }
}

function New-TestPackage {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [string]$Version = '1.0.0',
        [string]$PayloadMarker = 'v1',
        [string]$LockContent = "fastapi==0.139.2`nuvicorn==0.51.0`n",
        [long]$RequiredFreeBytes = 1073741824,
        [long]$CoreSize = 4096,
        [int]$DependencySeed = 31,
        [switch]$BreakPayloadHash
    )
    $payload = Join-Path $PackageRoot 'payload'
    New-Item -ItemType Directory -Force -Path (Join-Path $payload 'auto_tune\delivery') | Out-Null

    Set-Content -Path (Join-Path $payload 'auto_tune\main.py') -Encoding UTF8 -Value "MARKER = '$PayloadMarker'"
    Set-Content -Path (Join-Path $payload 'auto_tune\config.template.yaml') -Encoding UTF8 `
        -Value "project:`n  name: 示例项目`nlocal_index:`n  database_path: log/auto_tune.db`n"
    Set-Content -Path (Join-Path $payload 'auto_tune\delivery\preflight.py') -Encoding UTF8 -Value '# payload preflight'
    # files the whitelist must drop even though they sit inside the payload tree
    New-Item -ItemType Directory -Force -Path (Join-Path $payload 'auto_tune\tests') | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $payload 'auto_tune\__pycache__') | Out-Null
    Set-Content -Path (Join-Path $payload 'auto_tune\config.yaml') -Encoding UTF8 -Value 'project: 真实项目'
    Set-Content -Path (Join-Path $payload 'auto_tune\tests\test_leak.py') -Encoding UTF8 -Value '# never shipped'
    Set-Content -Path (Join-Path $payload 'auto_tune\__pycache__\main.pyc') -Encoding UTF8 -Value 'cache'
    Set-Content -Path (Join-Path $payload 'auto_tune\weights.pt') -Encoding UTF8 -Value 'weights'
    Set-Content -Path (Join-Path $PackageRoot 'requirements-windows.lock.txt') -Encoding UTF8 -Value $LockContent

    # The package carries the real launcher scripts, so the installed entry
    # points are executable and the permanent start/uninstall entries are the
    # production ones rather than a stub.
    foreach ($name in @('start.bat', 'start.ps1', 'uninstall.bat', 'uninstall.ps1')) {
        Copy-Item -Path (Join-Path $script:WindowsSource $name) -Destination (Join-Path $PackageRoot $name) -Force
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot 'lib') | Out-Null
    Copy-Item -Path $ModulePath -Destination (Join-Path $PackageRoot 'lib\AutoTuneDelivery.psm1') -Force

    # Everything the runtime needs travels in the package: the pinned Miniconda
    # installer, the CUDA PyTorch wheels and the ordinary dependency wheels.
    $paths = Get-TestOfflinePaths -PackageRoot $PackageRoot
    New-TestOfflineBundle -OfflineRoot $paths.Root `
        -RequirementsPath (Join-Path $PackageRoot 'requirements-windows.lock.txt') -CoreSize $CoreSize `
        -DependencySeed $DependencySeed
    $torchHash = Get-Sha256Hex -Path $paths.Torch
    $torchvisionHash = Get-Sha256Hex -Path $paths.Torchvision
    $installerHash = Get-Sha256Hex -Path $paths.Miniconda
    $lockHash = Get-Sha256Hex -Path (Join-Path $PackageRoot 'requirements-windows.lock.txt')

    $manifest = New-TestManifest -Version $Version -RequiredFreeBytes $RequiredFreeBytes `
        -MinicondaSha $installerHash -MinicondaSize (Get-Item -LiteralPath $paths.Miniconda).Length `
        -TorchSha $torchHash -TorchSize (Get-Item -LiteralPath $paths.Torch).Length `
        -TorchvisionSha $torchvisionHash -TorchvisionSize (Get-Item -LiteralPath $paths.Torchvision).Length `
        -LockSha $lockHash
    Write-JsonFile -Path (Join-Path $PackageRoot 'package-manifest.json') -Object $manifest | Out-Null
    Write-TestOfflineLock -ManifestRoot $PackageRoot -OfflineRoot $paths.Root | Out-Null

    $lock = New-PackageLock -PackageRoot $PackageRoot -Version $Version
    Write-JsonFile -Path (Join-Path $PackageRoot 'package-manifest.lock.json') -Object $lock | Out-Null

    if ($BreakPayloadHash) {
        Set-Content -Path (Join-Path $payload 'auto_tune\main.py') -Encoding UTF8 -Value "MARKER = 'tampered'"
    }
    # Returns nothing on purpose: a scenario calls this bare, and a leaked value
    # would turn the scenario's facts into an array.
}

function New-Layout {
    param([string]$LocalAppData)
    if ($PSBoundParameters.ContainsKey('LocalAppData')) {
        return Get-DeliveryLayout -LocalAppData $LocalAppData -Desktop $Desktop
    }
    return Get-DeliveryLayout -LocalAppData $WorkRoot -Desktop $Desktop
}

function New-FakeRuntimeInstaller {
    return {
        param($Layout, $Manifest, $Runner, $LogFile, $PackageRoot, [switch]$RefreshOnly)
        $script:RuntimeInstallCalls = $script:RuntimeInstallCalls + 1
        Add-Call 'install-runtime' @{ interpreter = $Layout.Interpreter;
            package = $PackageRoot; refresh_only = [bool]$RefreshOnly }
        if ($script:EnvMode -eq 'fail') {
            # the same failure the real layer raises when the private runtime
            # cannot be built from the package's offline bundle
            throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                    -Message 'pip install failed: api_key: sk-live-abcdef123456 at D:\Program Files\anaconda3\envs\auto_tune\python.exe')
        }
        $runtimeDir = Split-Path $Layout.Interpreter -Parent
        New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
        # each build writes a distinguishable interpreter, so "the live runtime
        # was not touched" is a claim that can actually be observed
        Set-Content -Path $Layout.Interpreter -Encoding UTF8 `
            -Value ("# private interpreter stub build={0}" -f $script:RuntimeInstallCalls)
        if ($script:EnvMode -eq 'fail-after-interpreter') {
            # the environment exists but the dependencies are incomplete and no
            # runtime stamp was written: the next run must finish the job
            throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                    -Message 'ERROR: pip failed while installing onnx: api_key: sk-live-abcdef123456')
        }
        return $Layout.Interpreter
    }
}

function New-FakeRunner {
    # The boundary the real Install-PrivateRuntime talks to. It records every
    # command and models the two effects the private runtime depends on: the
    # Miniconda installer creates the private interpreter, and pip/preflight
    # runs leave it in place.
    return {
        param($FilePath, $Arguments, $WorkingDirectory, $Environment, $LogFile)
        Add-Call 'run' @{ file = $FilePath; args = @($Arguments); cwd = $WorkingDirectory
            env = $Environment }
        if ($null -ne $script:ObservableLayout -and $null -eq $script:RuntimeCheckFacts -and
                @($Arguments) -contains '--require-offline-runtime') {
            $script:RuntimeCheckFacts = Get-CheckLayoutFacts -Layout $script:ObservableLayout
        }
        if ([System.IO.Path]::GetFileName($FilePath) -like 'Miniconda3-*') {
            $target = ''
            foreach ($argument in @($Arguments)) {
                if ([string]$argument -like '/D=*') { $target = ([string]$argument).Substring(3) }
            }
            if ($target) {
                New-Item -ItemType Directory -Force -Path $target | Out-Null
                Set-Content -Path (Join-Path $target 'python.exe') -Encoding UTF8 `
                    -Value '# private interpreter stub'
            }
        }
        if ($script:EnvMode -eq 'runtime-mismatch' -and @($Arguments) -contains '--require-offline-runtime') {
            # what the private interpreter answers when torch is not the pinned
            # CUDA build
            return @{ ExitCode = 1; StdOut = '' `
                    ; StdErr = '[delivery] ERROR DELIVERY_RUNTIME_MISMATCH: 私有运行环境与交付锁定的版本不一致。' }
        }
        return @{ ExitCode = 0; StdOut = ''; StdErr = '' }
    }
}

function New-FakePreflight {
    return {
        param($Python, $Arguments, $WorkingDirectory, $Environment, $LogFile)
        # ``liveInterpreter`` is the content of the *installed* interpreter at the
        # moment of the call: an upgrade that stages a new runtime must not have
        # touched it yet when the staged runtime is preflighted.
        $live = ''
        if (-not [string]::IsNullOrWhiteSpace($script:LiveInterpreter)) {
            $live = Get-TextSha256 -Path $script:LiveInterpreter
        }
        Add-Call 'preflight' @{ python = $Python; args = @($Arguments); cwd = $WorkingDirectory;
            env = $Environment; liveInterpreter = $live }
        if ($null -ne $script:ObservableLayout -and $null -eq $script:StartupCheckFacts -and
                @($Arguments) -contains '--require-gpu') {
            $script:StartupCheckFacts = Get-CheckLayoutFacts -Layout $script:ObservableLayout
        }
        $offlineCheck = (@($Arguments) -contains '--require-offline-runtime')
        if ($script:PreflightMode -eq 'no-gpu') {
            return @{ ExitCode = 1; StdOut = ''; StdErr = '[delivery] ERROR DELIVERY_GPU_UNAVAILABLE: 未检测到可用的 NVIDIA GPU。' }
        }
        if ($offlineCheck -and $script:PreflightMode -eq 'runtime-mismatch') {
            # what the private runtime answers when the installed torch is not
            # the pinned CUDA build (a CPU wheel would land here)
            return @{ ExitCode = 1; StdOut = ''; StdErr = '[delivery] ERROR DELIVERY_RUNTIME_MISMATCH: 私有运行环境与交付锁定的版本不一致。' }
        }
        return @{ ExitCode = 0; StdOut = '[delivery] 持久化目录就绪'; StdErr = '' }
    }
}

function New-FakeLauncher {
    return {
        param($FilePath, $Arguments, $WorkingDirectory, $Environment, $LogFile)
        Add-Call 'launch-service' @{ file = $FilePath; args = @($Arguments); cwd = $WorkingDirectory }
        return @{ Pid = 4242; StartedAt = (Get-Date).ToUniversalTime().ToString('o') }
    }
}

function New-FakeHealthProbe {
    return {
        param($Port, $TimeoutMs)
        Add-Call 'healthz' @{ port = $Port }
        return ($script:HealthMode -eq 'ok')
    }
}

function New-FakeBrowser {
    return {
        param($Url)
        Add-Call 'browser' @{ url = $Url }
        [void]$script:BrowserOpens.Add($Url)
        return $true
    }
}

function New-FakeDeferrer {
    return {
        param($Paths, $DelaySeconds)
        Add-Call 'defer-delete' @{ paths = @($Paths); delay = $DelaySeconds }
        return @($Paths)
    }
}

function New-FakeLinker {
    return {
        param($Path, $Target, $WorkingDirectory)
        Add-Call 'shortcut' @{ path = $Path; target = $Target; cwd = $WorkingDirectory }
        return $true
    }
}

function New-RealLauncher {
    param([Parameter(Mandatory = $true)]$Process)
    $processId = $Process.Id
    $started = $Process.StartTime.ToUniversalTime().ToString('o')
    return {
        param($FilePath, $Arguments, $WorkingDirectory, $Environment, $LogFile)
        Add-Call 'launch-service' @{ file = $FilePath; args = @($Arguments); cwd = $WorkingDirectory }
        return @{ Pid = $processId; StartedAt = $started }
    }
}

function Start-RealIdleProcess {
    # A real, harmless process that stays alive long enough to be observed and
    # killed. It is never a Python interpreter: the private runtime of a
    # synthetic installation is a stub.
    return Start-Process -FilePath $env:ComSpec `
        -ArgumentList @('/c', 'ping -n 120 127.0.0.1 > nul') -WindowStyle Hidden -PassThru
}

function Test-ProcessStillAlive {
    param([int]$ProcessId)
    return ($null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue))
}

function Get-ShortcutTarget {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -Path $Path -PathType Leaf)) { return '' }
    try {
        $shell = New-Object -ComObject WScript.Shell
        return [string]$shell.CreateShortcut($Path).TargetPath
    } catch {
        return ''
    }
}

function New-FakeProbe {
    return @{
        Is64BitOS    = { $true }
        FreeBytes    = { param($Path) 107374182400 }
        DriverPresent = { $true }
        IsWritable   = { param($Path) $true }
        ProcessAlive = { param($ProcessId, $StartedAt) $script:ProcessAlive }
        PortInUse    = { param($Port) $script:PortInUse }
    }
}

function Get-ShortDirectoryPath {
    # The spelling the file system has for a directory, which on a volume with
    # 8.3 names enabled is the short one ("C:\Users\ADMINI~1.DES"). On a volume
    # without them this is the long path again and the caller can see that the
    # short form is not available.
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        $fso = New-Object -ComObject Scripting.FileSystemObject
        $short = [string]$fso.GetFolder($Path).ShortPath
        if (-not [string]::IsNullOrWhiteSpace($short) -and (Test-Path -LiteralPath $short)) {
            return $short
        }
    } catch { }
    return $Path
}

function Get-LauncherLeftovers {
    # ``.new`` files are staged copies and ``.bak`` files are the copies kept
    # while an entry point is swapped in: neither may survive a run, successful
    # or not. Names are returned as one array, empty included, so the fact reads
    # the same in every scenario.
    param([Parameter(Mandatory = $true)]$Layout)
    $names = New-Object System.Collections.ArrayList
    foreach ($item in @(Get-ChildItem -Path $Layout.Root -Recurse -File -Force -ErrorAction SilentlyContinue)) {
        if ($item.Name -like '*.new' -or $item.Name -like '*.bak') { [void]$names.Add([string]$item.Name) }
    }
    return , @($names)
}

function New-ScarceProbe {
    # The machine Codex reproduced: the package asks for a full installation
    # budget (12 GiB) and only 1 GiB is free.
    param([long]$FreeBytes = 1073741824)
    $probe = New-FakeProbe
    $probe['FreeBytes'] = { param($Path) $FreeBytes }.GetNewClosure()
    return $probe
}

function Invoke-FullInstall {
    param($Layout, [string]$PackageRoot, [string]$Version = '1.0.0', [string]$PayloadMarker = 'v1',
        [switch]$BreakPayloadHash)
    $null = New-TestPackage -PackageRoot $PackageRoot -Version $Version `
        -PayloadMarker $PayloadMarker -BreakPayloadHash:$BreakPayloadHash
    return Invoke-Install -Layout $Layout -PackageRoot $PackageRoot `
        -Probe (New-FakeProbe) `
        -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
        -PreflightRunner (New-FakePreflight)
}

function Invoke-OfflineInstall {
    # The real private-runtime installer, driven by a recording runner: this is
    # the code path the operator's machine executes.
    param($Layout, [string]$PackageRoot, $Probe = $null, $RuntimeInstaller = $null)
    if ($null -eq $Probe) { $Probe = New-FakeProbe }
    $arguments = @{
        Layout = $Layout; PackageRoot = $PackageRoot; Probe = $Probe
        Runner = (New-FakeRunner); PreflightRunner = (New-FakePreflight)
    }
    if ($null -ne $RuntimeInstaller) { $arguments['RuntimeInstaller'] = $RuntimeInstaller }
    return Invoke-Install @arguments
}

function New-LayoutWithRoot {
    param([string]$Root)
    if ($PSBoundParameters.ContainsKey('Root') -and -not [string]::IsNullOrWhiteSpace($Root)) {
        return Get-DeliveryLayout -InstallRoot $Root -Desktop $Desktop
    }
    return New-Layout
}

function Invoke-Capture {
    param([scriptblock]$Body)
    try {
        $r = & $Body
        if ($null -eq $r) { return @{ Ok = $true; ErrorCode = $null; Message = $null; Result = $null; Detail = $null } }
        $carriesVerdict = ($r -is [System.Collections.IDictionary] -and $r.Contains('Ok')) -or
            ($r.PSObject.Properties.Name -contains 'Ok')
        if (-not $carriesVerdict) {
            return @{ Ok = $true; ErrorCode = $null; Message = $null; Result = $r; Detail = $null }
        }
        return @{ Ok = [bool]$r.Ok; ErrorCode = $r.ErrorCode; Message = $r.Message; Result = $r; Detail = $null }
    } catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        return @{ Ok = $false; ErrorCode = $info.Code; Message = $info.Message; Result = $null; Detail = $info.Detail }
    }
}

function Get-OfflineCallFacts {
    # What the recorded commands say about the offline promise: every pip
    # invocation must be unable to reach an index, and nothing may have gone
    # through the network trap.
    $runs = @(Get-CallData 'run')
    $pipCommands = @($runs | Where-Object {
            $argsList = @($_.args)
            ($argsList.Count -ge 2) -and ($argsList[0] -eq '-m') -and ($argsList[1] -eq 'pip')
        })
    $pipFacts = @($pipCommands | ForEach-Object { @($_.args) -join ' ' })
    $badPip = @($pipFacts | Where-Object {
            $_ -notlike '*--no-index*' -or $_ -notlike '*--find-links*'
        })
    $networkish = New-Object System.Collections.ArrayList
    foreach ($run in $runs) {
        $text = ([string]$run.file) + ' ' + (@($run.args) -join ' ')
        if ($text -match '(?i)https?://' -and $text -notmatch '(?i)--no-index') {
            [void]$networkish.Add($text)
        }
        if ([string]$run.file -match '(?i)conda\.exe') { [void]$networkish.Add($text) }
        if (@($run.args) -contains 'create') { [void]$networkish.Add($text) }
    }
    return [ordered]@{
        pipCommands = $pipFacts
        badPip      = $badPip
        networkish  = @($networkish)
        fetchCalls  = @($script:NetworkCalls).Count
        runCalls    = @($runs).Count
    }
}

# ── scenarios ────────────────────────────────────────────────────────────────

function Invoke-Scenario([string]$Name) {
    switch ($Name) {
        'layout-default' {
            $layout = New-Layout
            return [ordered]@{
                root         = $layout.Root
                app          = $layout.App
                runtime      = $layout.Runtime
                interpreter  = $layout.Interpreter
                data         = $layout.Data
                configPath   = $layout.ConfigPath
                cache        = $layout.Cache
                logs         = $layout.Logs
                stateFile    = $layout.StateFile
                port         = $layout.Port
                underRoot    = (@($layout.App, $layout.Runtime, $layout.Data, $layout.Cache,
                        $layout.Logs, $layout.StateFile) | Where-Object { $_ -notlike "$($layout.Root)*" }).Count -eq 0
                separateData = ($layout.Data -ne $layout.App) -and ($layout.Data -ne $layout.Runtime)
            }
        }
        'layout-unicode-space' {
            $base = Join-Path $WorkRoot '自动 调优 测试 目录'
            New-Item -ItemType Directory -Force -Path $base | Out-Null
            $layout = Get-DeliveryLayout -LocalAppData $base
            $created = Initialize-DeliveryLayout -Layout $layout
            $probe = New-FakeProbe
            $null = Assert-InstallPreconditions -Layout $layout -Probe $probe
            return [ordered]@{
                root      = $layout.Root
                data      = $layout.Data
                unicodeKept = $layout.Root.Contains('自动 调优 测试 目录')
                createdCount = @($created).Count
                dataExists = (Test-Path $layout.Data)
                stateWritable = (Test-Path (Split-Path $layout.StateFile -Parent))
            }
        }
        'layout-blank-localappdata' {
            try {
                $null = Get-DeliveryLayout -LocalAppData ''
                return [ordered]@{ rejected = $false }
            } catch {
                $info = Get-DeliveryErrorInfo -ErrorRecord $_
                return [ordered]@{ rejected = $true; code = $info.Code }
            }
        }
        'offline-bundle' {
            # One deliberate mutation (-Variant) models a broken delivery; the
            # shipping verifier must name the file and refuse it.
            #
            # Every variant starts from a *clean* package: the variants add files,
            # and a leftover from the previous run (this directory is shared by
            # every variant of the scenario) would be reported instead of — or in
            # addition to — the defect under test.
            $package = Join-Path $WorkRoot 'package'
            if (Test-Path $package) { Remove-Item -Path $package -Recurse -Force }
            $null = New-TestPackage -PackageRoot $package -CoreSize $script:OfflineCoreSize
            $paths = Get-TestOfflinePaths -PackageRoot $package
            $manifest = Get-PackageManifest -PackageRoot $package
            $mutation = 'none'
            switch ($Variant) {
                'valid' { }
                'no-bundle' { Remove-Item -Path $paths.Root -Recurse -Force; $mutation = 'no-bundle' }
                'no-lock' { Remove-Item -Path $paths.Lock -Force; $mutation = 'no-lock' }
                'miniconda-missing' {
                    Remove-Item -Path $paths.Miniconda -Force; $mutation = 'miniconda-missing'
                }
                'torch-missing' { Remove-Item -Path $paths.Torch -Force; $mutation = 'torch-missing' }
                'torchvision-missing' {
                    Remove-Item -Path $paths.Torchvision -Force; $mutation = 'torchvision-missing'
                }
                'core-hash-mismatch' {
                    [void](Write-FillerFile -Path $paths.Miniconda -Size $script:OfflineCoreSize -Seed 99)
                    $mutation = 'core-hash-mismatch'
                }
                'torch-hash-mismatch' {
                    [void](Write-FillerFile -Path $paths.Torch -Size $script:OfflineCoreSize -Seed 97)
                    $mutation = 'torch-hash-mismatch'
                }
                'wheel-missing' {
                    $dependencies = @(Get-TestDependencyWheels -OfflineRoot $paths.Root)
                    Remove-Item -Path $dependencies[0] -Force
                    $mutation = 'wheel-missing'
                }
                'wheel-hash-mismatch' {
                    $dependencies = @(Get-TestDependencyWheels -OfflineRoot $paths.Root)
                    [void](Write-FillerFile -Path $dependencies[-1] -Size 1024 -Seed 77)
                    $mutation = 'wheel-hash-mismatch'
                }
                'unlocked-wheel' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse 'extra-1.0.0-py3-none-any.whl') `
                            -Size 512 -Seed 5)
                    $mutation = 'unlocked-wheel'
                }
                'cpu-torch' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse 'torch-2.5.1-cp310-cp310-win_amd64.whl') `
                            -Size 2048 -Seed 3)
                    $mutation = 'cpu-torch'
                }
                'linux-wheel' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse `
                                'fastapi-0.139.2-py3-none-manylinux1_x86_64.whl') -Size 512 -Seed 4)
                    $mutation = 'linux-wheel'
                }
                'wrong-abi' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse 'uvicorn-0.51.0-cp39-cp39-win_amd64.whl') `
                            -Size 512 -Seed 6)
                    $mutation = 'wrong-abi'
                }
                'sdist' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse 'uvicorn-0.51.0.tar.gz') `
                            -Size 512 -Seed 7)
                    $mutation = 'sdist'
                }
                'abi3-wheel' {
                    # Not a defect: an abi3 wheel whose minimum CPython is 3.10 or
                    # older installs into the private runtime (opencv and psutil
                    # really ship cp37-abi3 builds). The build machine accepts
                    # them, so the bundle re-verification must accept them too
                    # instead of calling the wheelhouse incomplete. The lock is
                    # regenerated because these wheels really belong to the
                    # bundle, exactly as the shipping generator would record them.
                    foreach ($wheel in @('opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl',
                                         'psutil-7.0.0-cp37-abi3-win_amd64.whl')) {
                        [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse $wheel) `
                                -Size 1024 -Seed 21)
                    }
                    Write-TestOfflineLock -ManifestRoot $package -OfflineRoot $paths.Root | Out-Null
                    $mutation = 'abi3-wheel'
                }
                'requirements-changed' {
                    Add-Content -Path (Join-Path $package 'requirements-windows.lock.txt') -Value 'h11==0.16.0'
                    $mutation = 'requirements-changed'
                }
                # Files the lock does not name may not travel in the bundle, at
                # any depth: the exact file set is part of the offline promise.
                'miniconda-extra-file' {
                    [void](Write-FillerFile -Path (Join-Path (Split-Path $paths.Miniconda -Parent) `
                                'extra-cuda.dll') -Size 256 -Seed 13)
                    $mutation = 'miniconda-extra-file'
                }
                'miniconda-nested-extra-file' {
                    [void](Write-FillerFile -Path (Join-Path (Split-Path $paths.Miniconda -Parent) `
                                'Library\bin\extra-nested.dll') -Size 256 -Seed 14)
                    $mutation = 'miniconda-nested-extra-file'
                }
                'wheelhouse-nested-extra-file' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse `
                                'sub\extra-nested-1.0.0-py3-none-any.whl') -Size 256 -Seed 15)
                    $mutation = 'wheelhouse-nested-extra-file'
                }
                'offline-root-extra-file' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Root 'build-notes.txt') `
                            -Size 256 -Seed 16)
                    $mutation = 'offline-root-extra-file'
                }
                'lock-duplicate-path' {
                    $lock = Get-OfflineLock -OfflineRoot $paths.Root
                    $files = @($lock.files)
                    $files += @($files[0])
                    Write-JsonFile -Path $paths.Lock -Object (Copy-LockWithFiles -Lock $lock -Files $files) | Out-Null
                    $mutation = 'lock-duplicate-path'
                }
                'lock-traversal-path' {
                    $lock = Get-OfflineLock -OfflineRoot $paths.Root
                    $files = @($lock.files) + @([ordered]@{
                            path = '../outside-runtime.whl'; size = 8
                            sha256 = ('0' * 64); purpose = 'runtime-dependency' })
                    Write-JsonFile -Path $paths.Lock -Object (Copy-LockWithFiles -Lock $lock -Files $files) | Out-Null
                    $mutation = 'lock-traversal-path'
                }
                'lock-absolute-path' {
                    $lock = Get-OfflineLock -OfflineRoot $paths.Root
                    $files = @($lock.files) + @([ordered]@{
                            path = 'C:\Windows\system32\extra.dll'; size = 8
                            sha256 = ('0' * 64); purpose = 'runtime-dependency' })
                    Write-JsonFile -Path $paths.Lock -Object (Copy-LockWithFiles -Lock $lock -Files $files) | Out-Null
                    $mutation = 'lock-absolute-path'
                }
                'lock-size-mismatch' {
                    $lock = Get-OfflineLock -OfflineRoot $paths.Root
                    $files = @($lock.files)
                    $files[-1] = [ordered]@{
                        path = [string]$files[-1].path
                        size = [long]$files[-1].size + 1
                        sha256 = [string]$files[-1].sha256
                        purpose = [string]$files[-1].purpose
                    }
                    Write-JsonFile -Path $paths.Lock -Object (Copy-LockWithFiles -Lock $lock -Files $files) | Out-Null
                    $mutation = 'lock-size-mismatch'
                }
                default { $mutation = "unknown variant: $Variant" }
            }
            $tested = Test-OfflineBundle -PackageRoot $package -Manifest $manifest
            $captured = Invoke-Capture { Assert-OfflineBundle -PackageRoot $package -Manifest $manifest }
            $layout = Get-OfflineLayout -PackageRoot $package -Manifest $manifest
            # The plain .NET string, never `Get-Content -Raw`: a provider-decorated
            # string in a fact makes `ConvertTo-Json` hang (see the module header
            # trap note). This text is parsed by the Python suite.
            $lockText = if (Test-Path $layout.Lock) {
                [System.IO.File]::ReadAllText($layout.Lock, [System.Text.Encoding]::UTF8)
            } else {
                ''
            }
            $expectations = @(Get-OfflineCoreExpectations -Manifest $manifest)
            $offenders = @($tested.Offenders)
            $offenderText = ($offenders -join ' ')
            $lock = $null
            if (-not [string]::IsNullOrWhiteSpace($lockText)) {
                try { $lock = $lockText | ConvertFrom-Json } catch { $lock = $null }
            }
            $lockedFiles = @()
            if ($null -ne $lock) {
                $lockedFiles = @($lock.files | ForEach-Object { ([string]$_.path).Replace('\', '/') })
            }
            return [ordered]@{
                variant        = $Variant
                mutation       = $mutation
                testedOk       = $tested.Ok
                testedCode     = [string]$tested.Code
                accepted       = $captured.Ok
                code           = $captured.ErrorCode
                message        = $captured.Message
                offenders      = $offenders
                # what the operator is shown must never name a local absolute path
                offendersHaveAbsolutePath = ($offenderText -match '[A-Za-z]:\\')
                messageHasAbsolutePath = ([string]$captured.Message -match '[A-Za-z]:\\')
                minicondaInside = ($layout.Miniconda -like (Join-Path $layout.Root 'miniconda\*'))
                wheelhouseInside = ($layout.Wheelhouse -like (Join-Path $layout.Root 'wheelhouse*'))
                lockInside     = ($layout.Lock -eq (Join-Path $layout.Root 'offline-lock.json'))
                coreNames      = @($expectations | ForEach-Object { $_.Name })
                corePaths      = @($expectations | ForEach-Object { $_.RelativePath })
                corePurposes   = @($expectations | ForEach-Object { $_.Purpose })
                coreSizes      = @($expectations | ForEach-Object { [long]$_.Size })
                lockText       = $lockText
                lockHasAbsolutePath = ($lockText -match '[A-Za-z]:\\')
                # the exact file set, both sides: what the tree holds (the lock
                # itself excluded) and what the lock names
                onDiskFiles    = @(Get-ChildItem -Path $layout.Root -Recurse -File -Force -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -ne [System.IO.Path]::GetFileName($layout.Lock) } |
                    ForEach-Object { (Get-RelativePathUnderRoot -Root $layout.Root -Path $_.FullName).Replace('\', '/') } |
                    Sort-Object)
                lockedFiles    = @($lockedFiles | Sort-Object)
            }
        }
        'offline-install' {
            # The real private-runtime installer, on the recording runner: the
            # Miniconda installer, pip and the offline precheck are the shipping
            # commands, and no download trap may ever fire.
            $layout = New-LayoutWithRoot -Root $InstallRoot
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package -CoreSize $script:OfflineCoreSize
            $paths = Get-TestOfflinePaths -PackageRoot $package
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $result = Invoke-OfflineInstall -Layout $layout -PackageRoot $package
            $first = Get-OfflineCallFacts
            $runs = @(Get-CallData 'run')
            $installerRuns = @($runs | Where-Object { [System.IO.Path]::GetFileName([string]$_.file) -like 'Miniconda3-*' })
            $installerDir = ''
            if (@($installerRuns).Count -gt 0) {
                foreach ($argument in @($installerRuns[0].args)) {
                    if ([string]$argument -like '/D=*') { $installerDir = ([string]$argument).Substring(3) }
                }
            }
            $pipInstalls = @($runs | Where-Object {
                    $argsList = @($_.args)
                    ($argsList.Count -ge 3) -and ($argsList[0] -eq '-m') -and ($argsList[1] -eq 'pip') -and
                    ($argsList[2] -eq 'install')
                })
            $torchInstallArgs = @()
            foreach ($pip in $pipInstalls) {
                foreach ($argument in @($pip.args)) {
                    if ([string]$argument -like '*torch*') { $torchInstallArgs += [string]$argument }
                }
            }
            $torchOutsideWheelhouse = @($torchInstallArgs | Where-Object {
                    -not $_.StartsWith($paths.Wheelhouse, [System.StringComparison]::OrdinalIgnoreCase)
                })
            $stamp = Read-RuntimeStamp -Layout $layout
            $state = Read-InstallState -StateFile $layout.StateFile
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $second = Invoke-OfflineInstall -Layout $layout -PackageRoot $package
            $secondFacts = Get-OfflineCallFacts
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                message             = $result.Message
                status              = $state.status
                interpreter         = $layout.Interpreter
                interpreterInside   = $layout.Interpreter -like "$($layout.Root)*"
                interpreterExists   = (Test-Path -PathType Leaf $layout.Interpreter)
                interpreterPrivate  = ($layout.Interpreter -eq (Join-Path $layout.Root 'runtime\py310\python.exe'))
                installerDir        = $installerDir
                installerInside     = ($installerDir -eq (Split-Path -Path $layout.Interpreter -Parent))
                installerArgs       = if (@($installerRuns).Count -gt 0) { @($installerRuns[0].args) } else { @() }
                stampPresent        = (Test-Path -PathType Leaf (Get-RuntimeStampPath -Layout $layout))
                stampKey            = [string]$stamp.runtime_key
                pipCommands         = $first.pipCommands
                badPip              = $first.badPip
                networkish          = $first.networkish
                fetchCalls          = $first.fetchCalls
                runCount            = $first.runCalls
                torchInstallArgs    = @($torchInstallArgs)
                torchOutsideWheelhouse = @($torchOutsideWheelhouse)
                offlineCheckArgs    = @(@($runs | Where-Object {
                            @($_.args) -contains '--require-offline-runtime'
                        }) | ForEach-Object { @($_.args) -join ' ' })
                secondOk            = $second.Ok
                secondMode          = [string]$second.Mode
                secondRuns          = $secondFacts.runCalls
                secondPip           = $secondFacts.pipCommands
                secondFetch         = $secondFacts.fetchCalls
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
            }
        }
        'offline-install-broken' {
            # A bundle that is not exactly the verified one must stop the
            # installation before any runtime work happens.
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package -CoreSize $script:OfflineCoreSize
            $paths = Get-TestOfflinePaths -PackageRoot $package
            switch ($Variant) {
                'miniconda-missing' { Remove-Item -Path $paths.Miniconda -Force }
                'torch-hash-mismatch' {
                    [void](Write-FillerFile -Path $paths.Torch -Size $script:OfflineCoreSize -Seed 97)
                }
                'wheel-missing' {
                    $dependencies = @(Get-TestDependencyWheels -OfflineRoot $paths.Root)
                    Remove-Item -Path $dependencies[0] -Force
                }
                'cpu-torch' {
                    [void](Write-FillerFile -Path (Join-Path $paths.Wheelhouse 'torch-2.5.1-cp310-cp310-win_amd64.whl') `
                            -Size 2048 -Seed 3)
                }
                'no-lock' { Remove-Item -Path $paths.Lock -Force }
                default { }
            }
            $layout = New-Layout
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $result = Invoke-OfflineInstall -Layout $layout -PackageRoot $package
            $facts = Get-OfflineCallFacts
            return [ordered]@{
                variant      = $Variant
                ok           = $result.Ok
                errorCode    = $result.ErrorCode
                message      = $result.Message
                runCount     = $facts.runCalls
                pipCommands  = $facts.pipCommands
                networkish   = $facts.networkish
                fetchCalls   = $facts.fetchCalls
                appCopied    = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                stateExists  = (Test-Path -PathType Leaf $layout.StateFile)
                interpreterExists = (Test-Path -PathType Leaf $layout.Interpreter)
                stampPresent = (Test-Path -PathType Leaf (Get-RuntimeStampPath -Layout $layout))
                leftovers    = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'offline-install-broken-existing' {
            # The same refusal on a machine that already runs the product: the
            # complete installation must come out byte for byte as it went in.
            $package = Join-Path $WorkRoot 'package-good'
            $broken = Join-Path $WorkRoot 'package-broken'
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $weightsPath = Join-Path $layout.Data 'models\weights\best.pt'
            Set-Content -Path $weightsPath -Encoding UTF8 -Value 'weights'
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $appHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $stampHash = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbHash = Get-Sha256Hex -Path $dbPath
            $weightsHash = Get-Sha256Hex -Path $weightsPath
            $null = New-TestPackage -PackageRoot $broken -Version '1.0.1' -PayloadMarker 'v2'
            $brokenPaths = Get-TestOfflinePaths -PackageRoot $broken
            switch ($Variant) {
                'torch-hash-mismatch' {
                    [void](Write-FillerFile -Path $brokenPaths.Torch -Size $script:OfflineCoreSize -Seed 97)
                }
                'wheel-missing' {
                    $dependencies = @(Get-TestDependencyWheels -OfflineRoot $brokenPaths.Root)
                    Remove-Item -Path $dependencies[-1] -Force
                }
                default { }
            }
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $result = Invoke-OfflineInstall -Layout $layout -PackageRoot $broken
            $facts = Get-OfflineCallFacts
            return [ordered]@{
                variant           = $Variant
                ok                = $result.Ok
                errorCode         = $result.ErrorCode
                message           = $result.Message
                stateUnchanged    = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                stateStatus       = (Read-InstallState -StateFile $layout.StateFile).status
                stateVersion      = (Read-InstallState -StateFile $layout.StateFile).version
                appUnchanged      = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $appHash)
                appRunnable       = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                interpreterPresent = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged    = ((Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)) -eq $stampHash)
                configKept        = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept        = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                weightsKept       = ((Get-Sha256Hex -Path $weightsPath) -eq $weightsHash)
                runCount          = $facts.runCalls
                fetchCalls        = $facts.fetchCalls
                networkish        = $facts.networkish
                leftovers         = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'offline-install-runtime-mismatch' {
            # The private runtime answered the offline precheck with a mismatch:
            # the installation must not claim to be complete.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package -CoreSize $script:OfflineCoreSize
            $script:EnvMode = 'runtime-mismatch'
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $result = Invoke-OfflineInstall -Layout $layout -PackageRoot $package
            $script:EnvMode = 'ok'
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok            = $result.Ok
                errorCode     = $result.ErrorCode
                message       = $result.Message
                stateStatus   = if ($null -ne $state) { [string]$state.status } else { '' }
                stampPresent  = (Test-Path -PathType Leaf (Get-RuntimeStampPath -Layout $layout))
                offlineChecks = @(Get-CallData 'run' | Where-Object { @($_.args) -contains '--require-offline-runtime' }).Count
                fetchCalls    = @($script:NetworkCalls).Count
            }
        }
        'install-fresh' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $result = Invoke-FullInstall -Layout $layout -PackageRoot $package
            $state = Read-InstallState -StateFile $layout.StateFile
            $configFirst = Get-Content $layout.ConfigPath -Raw -Encoding UTF8
            # the operator edits the configuration, then re-runs install.bat
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $edited = Get-Content $layout.ConfigPath -Raw -Encoding UTF8
            $second = Invoke-FullInstall -Layout $layout -PackageRoot $package
            $stateAfter = Read-InstallState -StateFile $layout.StateFile
            $installCalls = @(Get-CallData 'install-runtime').Count
            $interpreterInside = $layout.Interpreter -like "$($layout.Runtime)*"
            $commands = @(Get-CallData 'run') + @(Get-CallData 'preflight') + @(Get-CallData 'launch-service')
            # every executable must be the installation's own, and none may point
            # at a developer interpreter
            $outsideInstall = @($commands | Where-Object {
                    $file = [string]$_.file
                    if (-not $file) { $file = [string]$_.python }
                    -not $file.StartsWith($layout.Root, [System.StringComparison]::OrdinalIgnoreCase)
                }).Count
            $devInterpreter = @($commands | Where-Object {
                    $file = [string]$_.file
                    if (-not $file) { $file = [string]$_.python }
                    $file -like '*anaconda3*' -or $file -like '*miniconda3*' -or $file -like '*\AppData\Local\Programs\Python*'
                }).Count
            return [ordered]@{
                ok              = $result.Ok
                errorCode       = $result.ErrorCode
                status          = $state.status
                schemaVersion   = $state.schema_version
                product         = $state.product
                version         = $state.version
                interpreter     = $layout.Interpreter
                interpreterInside = $interpreterInside
                interpreterUsed = @(Get-CallData 'preflight')[0].python
                appCopied       = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                payloadTrimmed  = -not (Test-Path (Join-Path $layout.App 'auto_tune\tests'))
                noRealConfigCopied = -not (Test-Path (Join-Path $layout.App 'auto_tune\config.yaml'))
                noWeightCopied  = -not (Test-Path (Join-Path $layout.App 'auto_tune\weights.pt'))
                noCacheCopied   = -not (Test-Path (Join-Path $layout.App 'auto_tune\__pycache__'))
                copiedFiles     = @(Get-ChildItem -Path $layout.App -Recurse -File).Count
                configCreated   = ($configFirst -like '*示例项目*')
                configKept      = ((Get-Content $layout.ConfigPath -Raw -Encoding UTF8) -eq $edited)
                dataDirs        = @('log', 'detect', 'runs', 'datasets') | Where-Object { Test-Path (Join-Path $layout.Data $_) }
                weightsDir      = (Test-Path (Join-Path $layout.Data 'models\weights'))
                secondRunOk     = $second.Ok
                secondRunError  = $second.ErrorCode
                runtimeInstallCalls = $installCalls
                stateAfterSecond = $stateAfter.status
                order           = Get-CallNames
                downloadCalls   = @(Get-CallData 'download').Count
                outsideInstallCalls = $outsideInstall
                devInterpreterCalls = $devInterpreter
                allCommandsAbsolute = (@($commands | Where-Object {
                            $file = [string]$_.file
                            if (-not $file) { $file = [string]$_.python }
                            -not [System.IO.Path]::IsPathRooted($file)
                        }).Count -eq 0)
                preflightArgs   = @(Get-CallData 'preflight')[0].args
                preflightCwd    = @(Get-CallData 'preflight')[0].cwd
                preflightEnv    = @(Get-CallData 'preflight')[0].env
                stateRuntimeLock = $state.runtime.lock_sha256
                configPathInState = $state.layout.config_path
            }
        }
        'install-runtime-check-order' {
            # Codex's real finding: a first installation asks the private
            # runtime to prove itself the moment it has been built — while the
            # program payload, the configuration and the persistent directories
            # do not exist yet — and only afterwards runs the formal GPU
            # preflight, once all of them are in place. The real
            # ``Install-PrivateRuntime`` runs here, so the offline precheck is
            # the command the operator's machine executes.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package -CoreSize $script:OfflineCoreSize
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $script:RuntimeCheckFacts = $null
            $script:StartupCheckFacts = $null
            $script:ObservableLayout = $layout
            try {
                $result = Invoke-OfflineInstall -Layout $layout -PackageRoot $package
            } finally {
                $script:ObservableLayout = $null
            }
            $state = Read-InstallState -StateFile $layout.StateFile
            $runtimeChecks = @(Get-CallData 'run' | Where-Object {
                    @($_.args) -contains '--require-offline-runtime' })
            $startupChecks = @(Get-CallData 'preflight' | Where-Object {
                    @($_.args) -contains '--require-gpu' })
            $runtimeFacts = $script:RuntimeCheckFacts
            $startupFacts = $script:StartupCheckFacts
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                message             = $result.Message
                status              = if ($null -ne $state) { [string]$state.status } else { '' }
                interpreter         = $layout.Interpreter
                runtimeCheckRan     = ($runtimeChecks.Count -gt 0)
                runtimeCheckArgs    = if ($runtimeChecks.Count -gt 0) { @($runtimeChecks[0].args) } else { @() }
                runtimeCheckFile    = if ($runtimeChecks.Count -gt 0) { [string]$runtimeChecks[0].file } else { '' }
                runtimeCheckEnv     = if ($runtimeChecks.Count -gt 0) { $runtimeChecks[0].env } else { @{} }
                runtimeCheckConfig  = if ($null -ne $runtimeFacts) { [bool]$runtimeFacts.config_present } else { $null }
                runtimeCheckPayload = if ($null -ne $runtimeFacts) { [bool]$runtimeFacts.payload_present } else { $null }
                startupCheckRan     = ($startupChecks.Count -gt 0)
                startupCheckArgs    = if ($startupChecks.Count -gt 0) { @($startupChecks[0].args) } else { @() }
                startupCheckPython  = if ($startupChecks.Count -gt 0) { [string]$startupChecks[0].python } else { '' }
                startupCheckConfig  = if ($null -ne $startupFacts) { [bool]$startupFacts.config_present } else { $null }
                startupCheckPayload = if ($null -ne $startupFacts) { [bool]$startupFacts.payload_present } else { $null }
                order               = Get-CallNames
                fetchCalls          = @($script:NetworkCalls).Count
            }
        }
        'install-preserves-existing-data' {
            $layout = New-Layout
            $null = Initialize-DeliveryLayout -Layout $layout
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            Set-Content -Path (Join-Path $layout.Data 'log\auto_tune.db') -Encoding UTF8 -Value 'sqlite-bytes'
            Set-Content -Path (Join-Path $layout.Data 'models\weights\best.pt') -Encoding UTF8 -Value 'weights'
            $before = @{}
            foreach ($relative in @('config\config.yaml', 'log\auto_tune.db', 'models\weights\best.pt')) {
                $before[$relative] = Get-Sha256Hex -Path (Join-Path $layout.Data $relative)
            }
            $result = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $after = @{}
            foreach ($relative in @('config\config.yaml', 'log\auto_tune.db', 'models\weights\best.pt')) {
                $after[$relative] = Get-Sha256Hex -Path (Join-Path $layout.Data $relative)
            }
            return [ordered]@{
                ok        = $result.Ok
                preserved = (@($before.Keys | Where-Object { $before[$_] -ne $after[$_] }).Count -eq 0)
                files     = @($before.Keys)
            }
        }
        'install-resume-after-failure' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package
            $script:PreflightMode = 'no-gpu'
            $failed = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $stateAfterFailure = Read-InstallState -StateFile $layout.StateFile
            $appAfterFailure = Test-Path (Join-Path $layout.App 'auto_tune\main.py')
            $configAfterFailure = Test-Path $layout.ConfigPath
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $script:PreflightMode = 'ok'
            $script:EnvMode = 'ok'
            $resumed = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                firstFailed   = (-not $failed.Ok)
                firstError    = $failed.ErrorCode
                appAfterFailure = $appAfterFailure
                configAfterFailure = $configAfterFailure
                partialStateStatus = $stateAfterFailure.status
                resumedOk     = $resumed.Ok
                resumedError  = $resumed.ErrorCode
                finalStatus   = $state.status
                configKept    = ((Get-Content $layout.ConfigPath -Raw -Encoding UTF8) -like '*客户项目*')
            }
        }
        'install-resumes-partial-runtime' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package
            $script:EnvMode = 'fail-after-interpreter'
            $failed = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $interpreterCreated = Test-Path -Path $layout.Interpreter -PathType Leaf
            $stampAbsent = -not (Test-Path -Path (Get-RuntimeStampPath -Layout $layout))
            $partialStateStatus = (Read-InstallState -StateFile $layout.StateFile).status
            $script:EnvMode = 'ok'
            $script:Calls.Clear()
            $resumed = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            $installs = @(Get-CallData 'install-runtime')
            $refreshOnly = $null
            if (@($installs).Count -gt 0) { $refreshOnly = [bool]$installs[0].refresh_only }
            return [ordered]@{
                firstOk            = $failed.Ok
                firstError         = $failed.ErrorCode
                interpreterCreated = $interpreterCreated
                stampAbsent        = $stampAbsent
                partialStateStatus = $partialStateStatus
                resumedOk          = $resumed.Ok
                resumedError       = $resumed.ErrorCode
                finalStatus        = $state.status
                refreshOnly        = $refreshOnly
                installerCalls     = @($installs).Count
            }
        }
        'install-insufficient-disk' {
            $layout = New-Layout
            $probe = New-FakeProbe
            $probe['FreeBytes'] = { param($Path) 1024 }
            $null = Initialize-DeliveryLayout -Layout $layout
            $captured = Invoke-Capture { Assert-SufficientFreeBytes -Layout $layout -Probe $probe `
                    -RequiredFreeBytes 1073741824 }
            return [ordered]@{
                rejected = (-not $captured.Ok)
                code     = $captured.ErrorCode
                message  = $captured.Message
            }
        }
        'install-fresh-insufficient-disk' {
            # A first installation on a machine with 1 GiB free where 12 GiB is
            # needed: it must stop before the download, the runtime and the
            # program files, not halfway through any of them.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package `
                -RequiredFreeBytes 12884901888
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                ok                  = $result.Ok
                mode                = [string]$result.Mode
                errorCode           = $result.ErrorCode
                message             = $result.Message
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                preflightCalls      = @(Get-CallData 'preflight').Count
                appCopied           = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                stateExists         = (Test-Path -PathType Leaf $layout.StateFile)
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'install-incomplete-insufficient-disk' {
            # The state file claims a finished installation but the runtime is
            # gone: this is not a repaired installation, it needs the full budget
            # and must not report itself as already installed.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package `
                -RequiredFreeBytes 12884901888
            $first = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            Remove-Item -Path $layout.Interpreter -Force
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                firstOk             = $first.Ok
                ok                  = $result.Ok
                mode                = [string]$result.Mode
                errorCode           = $result.ErrorCode
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                interpreterRestored = (Test-Path -PathType Leaf $layout.Interpreter)
            }
        }
        'install-repairs-launcher-with-scarce-disk' {
            # Codex's real finding: an installation of the same version that is
            # complete except for the permanent entry points, on a machine with
            # 1 GiB free where a fresh installation would need 12 GiB. The
            # launchers must be filled in and nothing else may be touched.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package `
                -RequiredFreeBytes 12884901888
            $null = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            # …the installation predates the permanent entry points
            foreach ($relative in Get-LauncherRelativePaths) {
                Remove-Item -Path (Join-Path $layout.Root $relative) -Force
            }
            Remove-Item -Path $layout.Shortcut -Force
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            $appHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $cacheFiles = @(Get-ChildItem -Path $layout.Cache -File -Recurse).Count
            $launchersBefore = (Test-LauncherInstalled -Layout $layout)
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                ok                  = $result.Ok
                mode                = [string]$result.Mode
                errorCode           = $result.ErrorCode
                launchersBefore     = $launchersBefore
                launchersAfter      = (Test-LauncherInstalled -Layout $layout)
                # hashed tolerantly: a run that fails to restore the entry points
                # must produce facts, not a harness error
                startBatMatchesPackage = ((Get-TextSha256 -Path $layout.StartBat) -eq
                    (Get-TextSha256 -Path (Join-Path $package 'start.bat')))
                moduleMatchesPackage = ((Get-TextSha256 -Path $layout.LibModule) -eq
                    (Get-TextSha256 -Path (Join-Path $package 'lib\AutoTuneDelivery.psm1')))
                shortcutExists      = (Test-Path -PathType Leaf $layout.Shortcut)
                shortcutTarget      = Get-ShortcutTarget -Path $layout.Shortcut
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                preflightCalls      = @(Get-CallData 'preflight').Count
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $appHash)
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                cacheKept           = (@(Get-ChildItem -Path $layout.Cache -File -Recurse).Count -ge $cacheFiles)
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'install-from-short-package-path' {
            # The package is reached through the short spelling of the tree
            # ("...\ADMINI~1.DES\...") while Get-ChildItem reports long paths:
            # the package lock built from it and the installation made from it
            # must both see every file.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package
            $shortPackage = Get-ShortDirectoryPath -Path $package
            # The lock the operator's package carries is built from the spelling
            # the build was given, so build it from the short one here.
            $lockOk = $true
            $lockError = ''
            $lockPayload = 0
            try {
                $lock = New-PackageLock -PackageRoot $shortPackage
                $lockPayload = @(@($lock.files) | Where-Object {
                        ([string]$_.path).StartsWith('payload/')
                    }).Count
                $lock | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 `
                    (Join-Path $package 'package-manifest.lock.json')
            } catch {
                $lockOk = $false
                $lockError = (Get-DeliveryErrorInfo -ErrorRecord $_).Code
            }
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $shortPackage `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                shortAvailable   = ($shortPackage -ne $package)
                shortPath        = $shortPackage
                longPath         = $package
                shortIsShorter   = ($shortPackage.Length -lt $package.Length)
                lockOk           = $lockOk
                lockError        = $lockError
                lockPayloadFiles = $lockPayload
                ok               = $result.Ok
                errorCode        = $result.ErrorCode
                copiedFiles      = @(Get-ChildItem -Path $layout.App -Recurse -File -ErrorAction SilentlyContinue).Count
                appCopied        = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                mainMarker       = (Test-Path -PathType Leaf (Join-Path $layout.App 'auto_tune\main.py')) -and
                    ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -like '*v1*')
                launchersAfter   = (Test-LauncherInstalled -Layout $layout)
                shortcutExists   = (Test-Path -PathType Leaf $layout.Shortcut)
                stateStatus      = (Read-InstallState -StateFile $layout.StateFile).status
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
            }
        }
        'install-failure-keeps-complete-state' {
            # A complete installation plus a package that cannot be installed in
            # the space available: nothing about the installation may change.
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbHash = Get-Sha256Hex -Path $dbPath
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $appHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -RequiredFreeBytes 12884901888
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $second `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                stateStatus         = $state.status
                stateVersion        = $state.version
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $appHash)
                appRunnable         = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
            }
        }
        'install-repair-launcher-failure-keeps-everything' {
            # Repairing the entry points must be all-or-nothing: one of the
            # launcher files is held open so it cannot be replaced, and the run
            # has to put back whatever it had already swapped.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbHash = Get-Sha256Hex -Path $dbPath
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $appHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $before = [ordered]@{}
            foreach ($path in @($layout.StartBat, $layout.StartScript, $layout.UninstallBat,
                    $layout.UninstallScript, $layout.LibModule)) {
                $before[$path] = Get-TextSha256 -Path $path
            }
            $locked = $layout.StartScript
            $handle = [System.IO.File]::Open($locked, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
            try {
                $script:Calls.Clear()
                $result = Invoke-Install -Layout $layout -PackageRoot $package `
                    -Probe (New-FakeProbe) `
                    -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                    -PreflightRunner (New-FakePreflight)
            } finally {
                $handle.Dispose()
            }
            $unchanged = $true
            foreach ($path in @($before.Keys)) {
                if ((Get-TextSha256 -Path $path) -ne $before[$path]) { $unchanged = $false }
            }
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                stateStatus         = (Read-InstallState -StateFile $layout.StateFile).status
                launchersUnchanged  = $unchanged
                launchersPresent    = (Test-LauncherInstalled -Layout $layout)
                leftovers           = Get-LauncherLeftovers -Layout $layout
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $appHash)
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
            }
        }
        'install-repair-shortcut-failure-keeps-everything' {
            # The desktop cannot be written to: the installation itself is fine
            # and must stay fine, and the shortcut is reported as not created.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbHash = Get-Sha256Hex -Path $dbPath
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $appHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $brokenLinker = {
                param($Path, $Target, $WorkingDirectory)
                throw 'the shell could not create the shortcut'
            }
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight) -Linker $brokenLinker
            $logFile = Join-Path $layout.Logs 'install.log'
            $logText = if (Test-Path $logFile) { Get-Content $logFile -Raw -Encoding UTF8 } else { '' }
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                shortcutCreated     = [bool]$result.ShortcutCreated
                shortcutWarned      = ($logText -like '*快捷方式*')
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                stateStatus         = (Read-InstallState -StateFile $layout.StateFile).status
                launchersPresent    = (Test-LauncherInstalled -Layout $layout)
                leftovers           = Get-LauncherLeftovers -Layout $layout
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $appHash)
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
            }
        }
        'install-already-installed-with-scarce-disk' {
            # Nothing is missing at all: running install.bat again on a machine
            # with almost no free space is simply a no-op that succeeds.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package `
                -RequiredFreeBytes 12884901888
            $null = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $script:Calls.Clear()
            $result = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                ok                  = $result.Ok
                mode                = [string]$result.Mode
                errorCode           = $result.ErrorCode
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                preflightCalls      = @(Get-CallData 'preflight').Count
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
            }
        }
        'install-no-nvidia-driver' {
            $layout = New-Layout
            $probe = New-FakeProbe
            $probe['DriverPresent'] = { $false }
            $null = New-TestPackage -PackageRoot (Join-Path $WorkRoot 'package')
            $captured = Invoke-Capture {
                Invoke-Install -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package') `
                    -Probe $probe `
                    -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                    -PreflightRunner (New-FakePreflight)
            }
            return [ordered]@{
                rejected           = (-not $captured.Ok)
                code               = $captured.ErrorCode
                message            = $captured.Message
                downloadCalls      = @(Get-CallData 'download').Count
                runtimeInstallCalls = $script:RuntimeInstallCalls
                preflightCalls     = @(Get-CallData 'preflight').Count
                appCopied          = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                stateExists        = (Test-Path -PathType Leaf $layout.StateFile)
            }
        }
        'install-log-redaction' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = New-TestPackage -PackageRoot $package
            $script:EnvMode = 'fail'
            $failed = Invoke-Install -Layout $layout -PackageRoot $package `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $logFile = Join-Path $layout.Logs 'install.log'
            $text = if (Test-Path $logFile) { Get-Content $logFile -Raw -Encoding UTF8 } else { '' }
            $forbidden = @()
            foreach ($needle in @('sk-live-abcdef123456', 'sk-should-never-appear', 'api_key:',
                    'D:\Program Files\anaconda3', 'at System.', 'End of inner exception')) {
                if ($text.Contains($needle)) { $forbidden += $needle }
            }
            return [ordered]@{
                failed     = (-not $failed.Ok)
                errorCode  = $failed.ErrorCode
                logExists  = (Test-Path $logFile)
                logFile    = $logFile
                forbidden  = $forbidden
                hasRedaction = ($text -like '*REDACTED*')
                mentionsCode = ($text -like "*$($failed.ErrorCode)*")
            }
        }
        'start-incomplete-state' {
            $layout = New-Layout
            $null = Initialize-DeliveryLayout -Layout $layout
            $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser)
            return [ordered]@{
                ok               = $result.Ok
                errorCode        = $result.ErrorCode
                launches         = @(Get-CallData 'launch-service').Count
                preflightCalls   = @(Get-CallData 'preflight').Count
                browserOpens     = @($script:BrowserOpens).Count
            }
        }
        'start-happy-path' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $script:Calls.Clear()
            $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            $preflight = @(Get-CallData 'preflight')[0]
            $launch = @(Get-CallData 'launch-service')[0]
            return [ordered]@{
                ok           = $result.Ok
                errorCode    = $result.ErrorCode
                order        = Get-CallNames
                preflightArgs = $preflight.args
                preflightPython = $preflight.python
                launchFile   = $launch.file
                launchCwd    = $launch.cwd
                launchArgs   = $launch.args
                browserOpens = @($script:BrowserOpens).Count
                browserUrl   = if (@($script:BrowserOpens).Count -gt 0) { $script:BrowserOpens[0] } else { '' }
                pid          = $result.Pid
                instanceFile = $result.InstanceFile
                instanceExists = (Test-Path $result.InstanceFile)
            }
        }
        'start-no-gpu' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $script:Calls.Clear()
            $script:PreflightMode = 'no-gpu'
            $script:BrowserOpens.Clear()
            $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            return [ordered]@{
                ok            = $result.Ok
                errorCode     = $result.ErrorCode
                message       = $result.Message
                launches      = @(Get-CallData 'launch-service').Count
                healthProbes  = @(Get-CallData 'healthz').Count
                browserOpens  = @($script:BrowserOpens).Count
            }
        }
        'start-port-invalid' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $script:Calls.Clear()
            $script:BrowserOpens.Clear()
            $result = Start-Studio -Layout $layout -EnvPort $EnvPort -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            return [ordered]@{
                ok           = $result.Ok
                errorCode    = $result.ErrorCode
                launches     = @(Get-CallData 'launch-service').Count
                preflightCalls = @(Get-CallData 'preflight').Count
            }
        }
        'start-port-from-state' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $state = Read-InstallState -StateFile $layout.StateFile
            $state.port = 8123
            $state | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $layout.StateFile
            $fromState = Get-StudioPort -Layout $layout -State (Read-InstallState -StateFile $layout.StateFile) -EnvPort ''
            $fromEnv = Get-StudioPort -Layout $layout -State (Read-InstallState -StateFile $layout.StateFile) -EnvPort '9000'
            return [ordered]@{ fromState = $fromState; fromEnv = $fromEnv }
        }
        'start-port-in-use' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $script:Calls.Clear()
            $script:PortInUse = $true
            $script:ProcessAlive = $false
            $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            return [ordered]@{
                ok            = $result.Ok
                errorCode     = $result.ErrorCode
                launches      = @(Get-CallData 'launch-service').Count
                preflightCalls = @(Get-CallData 'preflight').Count
            }
        }
        'start-second-instance' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $null = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            $script:Calls.Clear()
            $script:BrowserOpens.Clear()
            $script:ProcessAlive = $true
            $script:PortInUse = $true
            $second = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                -SleepMs 1 -TimeoutSeconds 5
            return [ordered]@{
                ok            = $second.Ok
                alreadyRunning = $second.AlreadyRunning
                errorCode     = $second.ErrorCode
                launches      = @(Get-CallData 'launch-service').Count
                preflightCalls = @(Get-CallData 'preflight').Count
                browserOpens  = @($script:BrowserOpens).Count
            }
        }
        'upgrade-preserves-data' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            Set-Content -Path (Join-Path $layout.Data 'log\auto_tune.db') -Encoding UTF8 -Value 'sqlite-bytes'
            Set-Content -Path (Join-Path $layout.Data 'log\history.jsonl') -Encoding UTF8 -Value '{"run":"train63"}'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbHash = Get-Sha256Hex -Path (Join-Path $layout.Data 'log\auto_tune.db')
            $cacheFiles = @(Get-ChildItem -Path $layout.Cache -File -Recurse).Count
            $stateBefore = Read-InstallState -StateFile $layout.StateFile
            $null = New-TestPackage -PackageRoot $second `
                -Version '1.0.1' -PayloadMarker 'v2'
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $stateAfter = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok              = $result.Ok
                errorCode       = $result.ErrorCode
                versionBefore   = $stateBefore.version
                versionAfter    = $stateAfter.version
                appMarkerChanged = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -like "*v2*")
                configKept      = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept      = ((Get-Sha256Hex -Path (Join-Path $layout.Data 'log\auto_tune.db')) -eq $dbHash)
                historyKept     = (Test-Path (Join-Path $layout.Data 'log\history.jsonl'))
                cacheKept       = (@(Get-ChildItem -Path $layout.Cache -File -Recurse).Count -ge $cacheFiles)
                runtimeUntouched = ($stateAfter.runtime.lock_sha256 -eq $stateBefore.runtime.lock_sha256)
                runtimeInstallCalls = $script:RuntimeInstallCalls
                stateHistory    = @($stateAfter.upgrades).Count
            }
        }
        'upgrade-runtime-lock-change' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -LockContent "fastapi==0.139.2`nuvicorn==0.51.0`ntorch==2.5.1`n"
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok                = $result.Ok
                errorCode         = $result.ErrorCode
                runtimeUpdated    = $result.RuntimeUpdated
                runtimeInstalls   = @(Get-CallData 'install-runtime').Count
                lockChanged       = ($state.runtime.lock_sha256 -ne $null)
                version           = $state.version
            }
        }
        'upgrade-lock-change-insufficient-disk' {
            # A changed dependency lock means a complete second runtime has to be
            # built beside the live one — the one upgrade that needs the full
            # budget. It must be refused before anything is staged or replaced,
            # so the installation the operator still has keeps working.
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $stampHash = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $markerHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -RequiredFreeBytes 12884901888 `
                -LockContent "fastapi==1.0.0`nuvicorn==2.0.0`ntorch==2.5.1`n"
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                stateVersion        = (Read-InstallState -StateFile $layout.StateFile).version
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $markerHash)
                appRunnable         = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged      = ((Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)) -eq $stampHash)
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                preflightCalls      = @(Get-CallData 'preflight').Count
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'upgrade-code-only-insufficient-disk' {
            # The same package lock: no runtime is built, so the program-only
            # upgrade must go through on a machine that could never host a
            # fresh installation.
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -RequiredFreeBytes 12884901888
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-ScarceProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                runtimeUpdated      = $result.RuntimeUpdated
                version             = $state.version
                appMarker           = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -like '*v2*')
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                downloadCalls       = @(Get-CallData 'download').Count
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'upgrade-verify-failure' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $markerBefore = Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -BreakPayloadHash
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $state = Read-InstallState -StateFile $layout.StateFile
            return [ordered]@{
                ok            = $result.Ok
                errorCode     = $result.ErrorCode
                appUnchanged  = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -eq $markerBefore)
                stateVersion  = $state.version
                appRunnable   = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
            }
        }
        'uninstall-keeps-data' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            Set-Content -Path (Join-Path $layout.Data 'log\history.jsonl') -Encoding UTF8 -Value '{"run":"train63"}'
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $result = Invoke-Uninstall -Layout $layout -Probe (New-FakeProbe) -Deferrer (New-FakeDeferrer)
            $notePath = $result.DataNoteFile
            $noteText = if (Test-Path $notePath) { Get-Content $notePath -Raw -Encoding UTF8 } else { '' }
            return [ordered]@{
                ok          = $result.Ok
                appRemoved  = -not (Test-Path $layout.App)
                runtimeRemoved = -not (Test-Path $layout.Runtime)
                dataKept    = (Test-Path $layout.Data)
                dataFiles   = @(Get-ChildItem -Path $layout.Data -File -Recurse).Count
                configKept  = ((Get-Content $layout.ConfigPath -Raw -Encoding UTF8) -like '*客户项目*')
                noteFile    = $notePath
                noteMentionsData = $noteText.Contains($layout.Data)
                stateRemoved = -not (Test-Path $layout.StateFile)
                message     = $result.Message
            }
        }
        'uninstall-remove-data-requires-confirm' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $blocked = Invoke-Uninstall -Layout $layout -Probe (New-FakeProbe) -Deferrer (New-FakeDeferrer) -RemoveData
            $dataAfterBlock = Test-Path $layout.Data
            $confirmed = Invoke-Uninstall -Layout $layout -Probe (New-FakeProbe) -Deferrer (New-FakeDeferrer) -RemoveData -Confirm
            return [ordered]@{
                firstOk      = $blocked.Ok
                firstError   = $blocked.ErrorCode
                dataAfterBlock = $dataAfterBlock
                secondOk     = $confirmed.Ok
                dataAfterConfirm = (Test-Path $layout.Data)
                dataRemoved  = $confirmed.DataRemoved
            }
        }
        'uninstall-safe-targets' {
            $layout = New-Layout
            $null = Initialize-DeliveryLayout -Layout $layout
            $candidates = @(
                @{ name = 'empty'; target = '' },
                @{ name = 'drive-root'; target = 'C:\' },
                @{ name = 'user-profile'; target = $env:USERPROFILE },
                @{ name = 'localappdata-root'; target = $env:LOCALAPPDATA },
                @{ name = 'install-root'; target = $layout.Root },
                @{ name = 'outside-install'; target = (Join-Path (Split-Path $layout.Root -Parent) 'Windows') },
                @{ name = 'repo-root'; target = (Split-Path $script:ModulePath -Parent) }
            )
            $rejected = New-Object System.Collections.ArrayList
            foreach ($candidate in $candidates) {
                try {
                    $null = Assert-SafeRemovalTarget -Layout $layout -Target $candidate.target
                    [void]$rejected.Add("$($candidate.name):ACCEPTED")
                } catch {
                    $info = Get-DeliveryErrorInfo -ErrorRecord $_
                    [void]$rejected.Add("$($candidate.name):$($info.Code)")
                }
            }
            $accepted = $false
            try {
                $null = Assert-SafeRemovalTarget -Layout $layout -Target $layout.Data
                $accepted = $true
            } catch { $accepted = $false }
            return [ordered]@{ results = @($rejected); dataAccepted = $accepted }
        }
        'install-deploys-launcher' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $result = Invoke-FullInstall -Layout $layout -PackageRoot $package
            $shortcut = $layout.Shortcut
            $launcher = @($layout.StartBat, $layout.StartScript, $layout.UninstallBat,
                $layout.UninstallScript, $layout.LibModule)
            return [ordered]@{
                ok                 = $result.Ok
                errorCode          = $result.ErrorCode
                startBat           = $layout.StartBat
                startScript        = $layout.StartScript
                uninstallBat       = $layout.UninstallBat
                uninstallScript    = $layout.UninstallScript
                modulePath         = $layout.LibModule
                startBatExists     = (Test-Path -PathType Leaf $layout.StartBat)
                startScriptExists  = (Test-Path -PathType Leaf $layout.StartScript)
                uninstallBatExists = (Test-Path -PathType Leaf $layout.UninstallBat)
                uninstallScriptExists = (Test-Path -PathType Leaf $layout.UninstallScript)
                moduleExists       = (Test-Path -PathType Leaf $layout.LibModule)
                insideInstallRoot  = (@($launcher | Where-Object {
                            -not $_.StartsWith($layout.Root, [System.StringComparison]::OrdinalIgnoreCase)
                        }).Count -eq 0)
                startBatMatchesPackage = ((Get-Sha256Hex -Path $layout.StartBat) -eq
                    (Get-Sha256Hex -Path (Join-Path $package 'start.bat')))
                moduleMatchesPackage = ((Get-Sha256Hex -Path $layout.LibModule) -eq
                    (Get-Sha256Hex -Path (Join-Path $package 'lib\AutoTuneDelivery.psm1')))
                shortcut           = $shortcut
                shortcutExists     = (Test-Path -PathType Leaf $shortcut)
                shortcutTarget     = Get-ShortcutTarget -Path $shortcut
                desktop            = $Desktop
                shortcutOnFakeDesktop = ($shortcut.StartsWith($Desktop, [System.StringComparison]::OrdinalIgnoreCase))
                shortcutCreated    = [bool]$result.ShortcutCreated
                stateStatus        = (Read-InstallState -StateFile $layout.StateFile).status
                configUntouched    = ((Get-Content $layout.ConfigPath -Raw -Encoding UTF8) -like '*示例项目*')
            }
        }
        'launcher-survives-package-removal' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            Set-Content -Path (Join-Path $layout.Data 'log\history.jsonl') -Encoding UTF8 -Value '{"run":"train63"}'
            Remove-Item -Path $package -Recurse -Force
            $packageGone = -not (Test-Path $package)

            $env:AUTO_TUNE_NO_PAUSE = '1'
            try {
                # The port is held by the caller, so the real start gates answer
                # PORT_IN_USE: the entry point ran, the module loaded and the
                # product decision was reached without the extracted package.
                $command = '"{0}" -LocalAppData "{1}" -Port {2}' -f $layout.StartBat, $WorkRoot, $Port
                $output = & $env:ComSpec /c $command 2>&1 | Out-String
                $exitCode = $LASTEXITCODE
            } finally {
                Remove-Item Env:AUTO_TUNE_NO_PAUSE -ErrorAction SilentlyContinue
            }
            return [ordered]@{
                packageGone      = $packageGone
                exitCode         = [int]$exitCode
                reachedStartGate = ($output -like '*PORT_IN_USE*')
                missingEntry     = ($output -like '*not found*')
                output           = $output.Trim()
                dataKept         = (Test-Path (Join-Path $layout.Data 'log\history.jsonl'))
            }
        }
        'upgrade-updates-launcher' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $package = Invoke-FullInstall -Layout $layout -PackageRoot $first
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            # an older or damaged launcher is replaced by the new package
            Set-Content -Path $layout.StartBat -Encoding UTF8 -Value 'rem stale launcher'
            Set-Content -Path $layout.UninstallScript -Encoding UTF8 -Value '# stale'
            New-TestPackage -PackageRoot $second `
                -Version '1.0.1' -PayloadMarker 'v2'
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            return [ordered]@{
                ok               = $result.Ok
                errorCode        = $result.ErrorCode
                startBatRestored = ((Get-Sha256Hex -Path $layout.StartBat) -eq
                    (Get-Sha256Hex -Path (Join-Path $second 'start.bat')))
                uninstallRestored = ((Get-Sha256Hex -Path $layout.UninstallScript) -eq
                    (Get-Sha256Hex -Path (Join-Path $second 'uninstall.ps1')))
                modulePresent    = (Test-Path -PathType Leaf $layout.LibModule)
                shortcutExists   = (Test-Path -PathType Leaf $layout.Shortcut)
                shortcutTarget   = Get-ShortcutTarget -Path $layout.Shortcut
                configKept       = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept       = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                version          = (Read-InstallState -StateFile $layout.StateFile).version
            }
        }
        'upgrade-runtime-staged-success' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterBefore = Get-TextSha256 -Path $layout.Interpreter
            $stampHashBefore = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
            $stateHashBefore = Get-TextSha256 -Path $layout.StateFile
            $runtimeKeyBefore = [string](Read-RuntimeStamp -Layout $layout).runtime_key
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -LockContent "fastapi==1.0.0`nuvicorn==2.0.0`ntorch==2.5.1`n"
            $script:LiveInterpreter = $layout.Interpreter
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $script:LiveInterpreter = ''
            $state = Read-InstallState -StateFile $layout.StateFile
            $stamp = Read-RuntimeStamp -Layout $layout
            $preflights = @(Get-CallData 'preflight')
            $stagedPreflight = $preflights[0]
            $finalPreflight = $preflights[-1]
            # Facts are hashes, never whole files: ``Get-Content`` decorates its
            # output with the FileSystem provider's note properties (PSPath,
            # Length, ...), and ConvertTo-Json then serializes those as well.
            return [ordered]@{
                ok                 = $result.Ok
                errorCode          = $result.ErrorCode
                runtimeUpdated     = $result.RuntimeUpdated
                version            = $state.version
                interpreterPath    = $layout.Interpreter
                interpreterPresent = (Test-Path -PathType Leaf $layout.Interpreter)
                runtimeKeyBefore   = $runtimeKeyBefore
                runtimeKeyAfter    = [string]$stamp.runtime_key
                stampHashBefore    = $stampHashBefore
                stampHashAfter     = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
                stateHashBefore    = $stateHashBefore
                stateHashAfter     = Get-TextSha256 -Path $layout.StateFile
                preflightCount     = @($preflights).Count
                stagedPreflightPython = [string]$stagedPreflight.python
                stagedPreflightOutsideLiveRuntime = (-not ([string]$stagedPreflight.python).Equals(
                        $layout.Interpreter, [System.StringComparison]::OrdinalIgnoreCase))
                stagedPreflightEnvPythonPath = [string]$stagedPreflight.env.PYTHONPATH
                stagedPreflightLiveInterpreterHash = [string]$stagedPreflight.liveInterpreter
                interpreterHashBefore = $interpreterBefore
                finalPreflightPython = [string]$finalPreflight.python
                leftovers          = @(Get-ChildItem -Path $layout.Root -Force -Directory |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
                oldRuntimeLeftover = @(Get-ChildItem -Path $layout.Root -Force -Directory |
                    Where-Object { $_.Name -like '.runtime-*' } | ForEach-Object { $_.Name })
                configKept         = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept         = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                appMarker          = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -like '*v2*')
            }
        }
        'upgrade-offline-wheel-refresh' {
            # The requirements file does not change, but the wheel bytes inside
            # the offline bundle do: the offline lock and the package lock are
            # both rebuilt for the new bytes. The runtime identity must follow
            # the verified offline bundle, so the upgrade builds the new runtime
            # beside the live one instead of reusing it — and a failed build
            # leaves the installed version exactly as it was.
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterBefore = Get-TextSha256 -Path $layout.Interpreter
            $stampHashBefore = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
            $stateHashBefore = Get-TextSha256 -Path $layout.StateFile
            $markerHashBefore = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            $runtimeKeyBefore = [string](Read-RuntimeStamp -Layout $layout).runtime_key
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            $requirementsBefore = Get-Sha256Hex -Path (Join-Path $first 'requirements-windows.lock.txt')
            $offlineLockBefore = Get-Sha256Hex -Path (Join-Path $first 'offline\offline-lock.json')
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -DependencySeed 131
            $requirementsAfter = Get-Sha256Hex -Path (Join-Path $second 'requirements-windows.lock.txt')
            $offlineLockAfter = Get-Sha256Hex -Path (Join-Path $second 'offline\offline-lock.json')
            if ($Variant -eq 'fail') { $script:EnvMode = 'fail-after-interpreter' }
            $script:LiveInterpreter = $layout.Interpreter
            $script:Calls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $script:LiveInterpreter = ''
            $script:EnvMode = 'ok'
            $state = Read-InstallState -StateFile $layout.StateFile
            $stamp = Read-RuntimeStamp -Layout $layout
            $runtimeBuilds = @(Get-CallData 'install-runtime')
            $preflights = @(Get-CallData 'preflight')
            $stagedPath = ''
            if (@($runtimeBuilds).Count -gt 0) { $stagedPath = [string]$runtimeBuilds[0].interpreter }
            return [ordered]@{
                variant             = $Variant
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                runtimeUpdated      = $result.RuntimeUpdated
                runtimeInstalls     = @($runtimeBuilds).Count
                runtimeKeyBefore    = $runtimeKeyBefore
                runtimeKeyAfter     = [string]$stamp.runtime_key
                requirementsUnchanged = ($requirementsBefore -eq $requirementsAfter)
                offlineLockChanged  = ($offlineLockBefore -ne $offlineLockAfter)
                version             = $state.version
                appMarker           = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -like '*v2*')
                appUnchanged        = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $markerHashBefore)
                appRunnable         = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterBefore   = $interpreterBefore
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterBefore)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged      = ((Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)) -eq $stampHashBefore)
                stateUnchanged      = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHashBefore)
                stateVersion        = $state.version
                stagedInterpreterPath = $stagedPath
                stagedOutsideLiveRuntime = (-not [string]::IsNullOrWhiteSpace($stagedPath) -and
                    -not $stagedPath.Equals($layout.Interpreter, [System.StringComparison]::OrdinalIgnoreCase))
                stagedPreflightLiveInterpreterHash = if (@($preflights).Count -gt 0) {
                    [string]$preflights[0].liveInterpreter
                } else { '' }
                preflightCount      = @($preflights).Count
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept          = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                fetchCalls          = @($script:NetworkCalls).Count
                offlineChecks       = @(Get-CallData 'run' | Where-Object {
                        @($_.args) -contains '--require-offline-runtime' }).Count
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'runtime-stamp-legacy' {
            # A stamp written by an older delivery carries no offline-bundle
            # identity in its key; an even older one may carry no key at all.
            # Neither may raise: an unrecognised stamp is simply not a match, so
            # the dependencies are installed into the existing interpreter again
            # and a current stamp is written.
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            $stampPath = Get-RuntimeStampPath -Layout $layout
            $legacy = [ordered]@{
                schema_version = '1.0'
                runtime_key    = 'aaaaaaaaaaaaaaaa|torch==2.5.1,torchvision==0.20.1'
                python_version = ''
                interpreter    = $layout.Interpreter
                installed_at   = (Get-Date).ToUniversalTime().ToString('o')
            }
            if ($Variant -eq 'missing-key') {
                $legacy = [ordered]@{
                    schema_version = '1.0'
                    interpreter    = $layout.Interpreter
                }
            }
            if ($Variant -eq 'unknown-schema') {
                $legacy = [ordered]@{
                    schema_version = '9.9'
                    runtime_key    = 'aaaaaaaaaaaaaaaa|torch==2.5.1,torchvision==0.20.1'
                    interpreter    = $layout.Interpreter
                }
            }
            Write-InstallState -StateFile $stampPath -State $legacy | Out-Null
            $legacyKey = [string]$legacy.runtime_key
            $script:Calls.Clear()
            $captured = Invoke-Capture {
                Ensure-PrivateRuntime -Layout $layout `
                    -Manifest (Get-PackageManifest -PackageRoot $package) `
                    -PackageRoot $package -Runner (New-FakeRunner) `
                    -RuntimeInstaller (New-FakeRuntimeInstaller)
            }
            $stampAfter = Read-RuntimeStamp -Layout $layout
            $builds = @(Get-CallData 'install-runtime')
            return [ordered]@{
                variant            = $Variant
                ok                 = $captured.Ok
                errorCode          = $captured.ErrorCode
                errorMessage       = $captured.Message
                transferMode       = [string]$captured.Result.TransferMode
                runtimeInstalls    = @($builds).Count
                refreshOnlyCount   = @($builds | Where-Object { $_.refresh_only }).Count
                legacyKey          = $legacyKey
                runtimeKeyAfter    = [string]$stampAfter.runtime_key
                keyChanged         = ([string]$stampAfter.runtime_key -ne $legacyKey)
                stampHasOfflineIdentity = ([string]$stampAfter.runtime_key -like 'v2|*')
                stampParses        = ($null -ne $stampAfter)
                interpreterPresent = (Test-Path -PathType Leaf $layout.Interpreter)
            }
        }
        'upgrade-runtime-build-failure' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterBefore = Get-TextSha256 -Path $layout.Interpreter
            $stampBefore = Get-Content (Get-RuntimeStampPath -Layout $layout) -Raw -Encoding UTF8
            $stateBefore = Get-Content $layout.StateFile -Raw -Encoding UTF8
            $markerBefore = Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -LockContent "fastapi==1.0.0`nuvicorn==2.0.0`ntorch==2.5.1`n"
            $script:EnvMode = 'fail-after-interpreter'
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $script:EnvMode = 'ok'
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                appUnchanged        = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -eq $markerBefore)
                appRunnable         = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterBefore)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged      = ((Get-Content (Get-RuntimeStampPath -Layout $layout) -Raw -Encoding UTF8) -eq $stampBefore)
                stateUnchanged      = ((Get-Content $layout.StateFile -Raw -Encoding UTF8) -eq $stateBefore)
                stateVersion        = (Read-InstallState -StateFile $layout.StateFile).version
                dataKept            = (((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash) -and
                    ((Get-Sha256Hex -Path $dbPath) -eq $dbHash))
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
                runtimeInstallCalls = @(Get-CallData 'install-runtime').Count
            }
        }
        'upgrade-runtime-preflight-failure' {
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterBefore = Get-TextSha256 -Path $layout.Interpreter
            $stampBefore = Get-Content (Get-RuntimeStampPath -Layout $layout) -Raw -Encoding UTF8
            $stateBefore = Get-Content $layout.StateFile -Raw -Encoding UTF8
            $markerBefore = Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' `
                -PayloadMarker 'v2' -LockContent "fastapi==1.0.0`nuvicorn==2.0.0`ntorch==2.5.1`n"
            $script:PreflightMode = 'no-gpu'
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) `
                -RuntimeInstaller (New-FakeRuntimeInstaller) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $script:PreflightMode = 'ok'
            return [ordered]@{
                ok                  = $result.Ok
                errorCode           = $result.ErrorCode
                appUnchanged        = ((Get-Content (Join-Path $layout.App 'auto_tune\main.py') -Raw -Encoding UTF8) -eq $markerBefore)
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterBefore)
                interpreterPresent  = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged      = ((Get-Content (Get-RuntimeStampPath -Layout $layout) -Raw -Encoding UTF8) -eq $stampBefore)
                stateUnchanged      = ((Get-Content $layout.StateFile -Raw -Encoding UTF8) -eq $stateBefore)
                stateVersion        = (Read-InstallState -StateFile $layout.StateFile).version
                configKept          = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                leftovers           = @(Get-ChildItem -Path $layout.Root -Force -Directory |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'upgrade-offline-precheck-failure' {
            # A changed dependency lock builds a second runtime beside the live
            # one. The private interpreter of that staged runtime answers the
            # offline precheck with a mismatch, so the upgrade must roll back the
            # program, the runtime and the state.
            $layout = New-Layout
            $first = Join-Path $WorkRoot 'package-1'
            $second = Join-Path $WorkRoot 'package-2'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $first
            $interpreterHash = Get-TextSha256 -Path $layout.Interpreter
            $stampHash = Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)
            $stateHash = Get-TextSha256 -Path $layout.StateFile
            $markerHash = Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')
            Set-Content -Path $layout.ConfigPath -Encoding UTF8 -Value "project:`n  name: 客户项目`n"
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $dbPath = Join-Path $layout.Data 'log\auto_tune.db'
            Set-Content -Path $dbPath -Encoding UTF8 -Value 'sqlite-bytes'
            $dbHash = Get-Sha256Hex -Path $dbPath
            New-TestPackage -PackageRoot $second -Version '1.0.1' -PayloadMarker 'v2' `
                -LockContent "fastapi==1.0.0`nuvicorn==2.0.0`ntorch==2.5.1`n"
            $script:EnvMode = 'runtime-mismatch'
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $result = Invoke-Upgrade -Layout $layout -PackageRoot $second `
                -Probe (New-FakeProbe) -Runner (New-FakeRunner) `
                -PreflightRunner (New-FakePreflight)
            $script:EnvMode = 'ok'
            return [ordered]@{
                ok                   = $result.Ok
                errorCode            = $result.ErrorCode
                stateVersion         = (Read-InstallState -StateFile $layout.StateFile).version
                appUnchanged         = ((Get-TextSha256 -Path (Join-Path $layout.App 'auto_tune\main.py')) -eq $markerHash)
                appRunnable          = (Test-Path (Join-Path $layout.App 'auto_tune\main.py'))
                interpreterUnchanged = ((Get-TextSha256 -Path $layout.Interpreter) -eq $interpreterHash)
                interpreterPresent   = (Test-Path -PathType Leaf $layout.Interpreter)
                stampUnchanged       = ((Get-TextSha256 -Path (Get-RuntimeStampPath -Layout $layout)) -eq $stampHash)
                stateUnchanged       = ((Get-TextSha256 -Path $layout.StateFile) -eq $stateHash)
                configKept           = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                sqliteKept           = ((Get-Sha256Hex -Path $dbPath) -eq $dbHash)
                fetchCalls           = @($script:NetworkCalls).Count
                pipCommands          = @(Get-CallData 'run' | Where-Object {
                        $argsList = @($_.args)
                        ($argsList.Count -ge 2) -and ($argsList[0] -eq '-m') -and ($argsList[1] -eq 'pip')
                    } | ForEach-Object { @($_.args) -join ' ' })
                offlineChecks        = @(Get-CallData 'run' | Where-Object {
                        @($_.args) -contains '--require-offline-runtime'
                    }).Count
                leftovers            = @(Get-ChildItem -Path $layout.Root -Force -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like '.staging-*' -or $_.Name -like '.rollback-*' } |
                    ForEach-Object { $_.Name })
            }
        }
        'start-health-timeout' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $script:Calls.Clear()
            $script:BrowserOpens.Clear()
            $script:HealthMode = 'fail'
            $process = Start-RealIdleProcess
            $processId = $process.Id
            $processStartedAt = $process.StartTime.ToUniversalTime().ToString('o')
            try {
                $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                    -PreflightRunner (New-FakePreflight) -Launcher (New-RealLauncher -Process $process) `
                    -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                    -SleepMs 200 -TimeoutSeconds 1
                Start-Sleep -Milliseconds 500
                $aliveAfter = Test-ProcessStillAlive -ProcessId $processId
                $survivor = Get-Process -Id $processId -ErrorAction SilentlyContinue
                $survivorName = ''
                if ($null -ne $survivor) { $survivorName = $survivor.ProcessName }
                $identityMatched = Test-ProcessIdentity -ProcessId $processId -StartedAt $processStartedAt
            } finally {
                if (Test-ProcessStillAlive -ProcessId $process.Id) {
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
            }
            $instancePath = Join-Path $layout.Logs 'studio.json'
            $logPath = Join-Path $layout.Logs 'start.log'
            $logText = if (Test-Path $logPath) { Get-Content $logPath -Raw -Encoding UTF8 } else { '' }
            return [ordered]@{
                ok               = $result.Ok
                errorCode        = $result.ErrorCode
                healthy          = $result.Healthy
                message          = $result.Message
                browserOpens     = @($script:BrowserOpens).Count
                launches         = @(Get-CallData 'launch-service').Count
                healthProbes     = @(Get-CallData 'healthz').Count
                startedPid       = [int]$processId
                processAliveAfter = $aliveAfter
                survivorName     = $survivorName
                identityMatched  = $identityMatched
                instanceFile     = $instancePath
                instanceExists   = (Test-Path -PathType Leaf $instancePath)
                logMentionsCode  = ($logText -like '*HEALTH_CHECK_FAILED*')
                logMentionsVolatile = ($logText -like "*$processId*" -or $logText -like '*studio.err.log*')
                port             = [int]$result.Port
            }
        }
        'start-health-timeout-existing-instance' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            $existing = Start-RealIdleProcess
            try {
                [void](Write-InstallState -StateFile (Join-Path $layout.Logs 'studio.json') -State ([ordered]@{
                        pid        = [int]$existing.Id
                        started_at = $existing.StartTime.ToUniversalTime().ToString('o')
                        port       = $layout.Port
                    }))
                $script:Calls.Clear()
                $script:BrowserOpens.Clear()
                $script:ProcessAlive = $true
                $script:HealthMode = 'fail'
                $result = Start-Studio -Layout $layout -Probe (New-FakeProbe) `
                    -PreflightRunner (New-FakePreflight) -Launcher (New-FakeLauncher) `
                    -HealthProbe (New-FakeHealthProbe) -BrowserOpener (New-FakeBrowser) `
                    -SleepMs 200 -TimeoutSeconds 1
                Start-Sleep -Milliseconds 500
                $aliveAfter = Test-ProcessStillAlive -ProcessId $existing.Id
            } finally {
                if (Test-ProcessStillAlive -ProcessId $existing.Id) {
                    Stop-Process -Id $existing.Id -Force -ErrorAction SilentlyContinue
                }
            }
            return [ordered]@{
                ok              = $result.Ok
                alreadyRunning  = $result.AlreadyRunning
                browserOpens    = @($script:BrowserOpens).Count
                launches        = @(Get-CallData 'launch-service').Count
                healthProbes    = @(Get-CallData 'healthz').Count
                existingAlive   = $aliveAfter
                instanceKept    = (Test-Path -PathType Leaf (Join-Path $layout.Logs 'studio.json'))
            }
        }
        'uninstall-removes-launcher-and-shortcut' {
            $layout = New-Layout
            $null = Invoke-FullInstall -Layout $layout -PackageRoot (Join-Path $WorkRoot 'package')
            Set-Content -Path (Join-Path $layout.Data 'log\history.jsonl') -Encoding UTF8 -Value '{"run":"train63"}'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $shortcut = $layout.Shortcut
            $result = Invoke-Uninstall -Layout $layout -Probe (New-FakeProbe) -Deferrer (New-FakeDeferrer)
            $deferred = @(Get-CallData 'defer-delete')
            $deferredPaths = @()
            if (@($deferred).Count -gt 0) { $deferredPaths = @($deferred[0].paths) }
            return [ordered]@{
                ok               = $result.Ok
                appRemoved       = -not (Test-Path $layout.App)
                runtimeRemoved   = -not (Test-Path $layout.Runtime)
                stateRemoved     = -not (Test-Path $layout.StateFile)
                dataKept         = (Test-Path (Join-Path $layout.Data 'log\history.jsonl'))
                configKept       = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                cacheKept        = (Test-Path $layout.Cache)
                startBatRemoved  = -not (Test-Path $layout.StartBat)
                startScriptRemoved = -not (Test-Path $layout.StartScript)
                libRemoved       = -not (Test-Path $layout.Launcher)
                shortcutRemoved  = -not (Test-Path -PathType Leaf $shortcut)
                uninstallBatKept = (Test-Path -PathType Leaf $layout.UninstallBat)
                uninstallScriptKept = (Test-Path -PathType Leaf $layout.UninstallScript)
                deferredPaths    = @($deferredPaths)
                deferredCount    = @($deferred).Count
                noteExists       = (Test-Path -PathType Leaf $result.DataNoteFile)
                message          = $result.Message
            }
        }
        'uninstall-entry-runs-from-install-root' {
            $layout = New-Layout
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            Set-Content -Path (Join-Path $layout.Data 'log\history.jsonl') -Encoding UTF8 -Value '{"run":"train63"}'
            $configHash = Get-Sha256Hex -Path $layout.ConfigPath
            $fakeDesktopShortcut = $layout.Shortcut
            Remove-Item -Path $package -Recurse -Force
            $packageGone = -not (Test-Path $package)

            $env:AUTO_TUNE_NO_PAUSE = '1'
            try {
                $command = '"{0}" -LocalAppData "{1}"' -f $layout.UninstallBat, $WorkRoot
                $output = & $env:ComSpec /c $command 2>&1 | Out-String
                $exitCode = $LASTEXITCODE
            } finally {
                Remove-Item Env:AUTO_TUNE_NO_PAUSE -ErrorAction SilentlyContinue
            }

            # the entry point is deleted by a delayed helper, never while cmd is
            # still reading it
            $deadline = (Get-Date).AddSeconds(30)
            while ((Get-Date) -lt $deadline -and (Test-Path -PathType Leaf $layout.UninstallBat)) {
                Start-Sleep -Milliseconds 500
            }
            return [ordered]@{
                packageGone        = $packageGone
                exitCode           = [int]$exitCode
                output             = $output.Trim()
                appRemoved         = -not (Test-Path $layout.App)
                runtimeRemoved     = -not (Test-Path $layout.Runtime)
                startBatRemoved    = -not (Test-Path $layout.StartBat)
                startScriptRemoved = -not (Test-Path $layout.StartScript)
                libRemoved         = -not (Test-Path $layout.Launcher)
                uninstallBatRemoved = -not (Test-Path -PathType Leaf $layout.UninstallBat)
                uninstallScriptRemoved = -not (Test-Path -PathType Leaf $layout.UninstallScript)
                dataKept           = (Test-Path (Join-Path $layout.Data 'log\history.jsonl'))
                configKept         = ((Get-Sha256Hex -Path $layout.ConfigPath) -eq $configHash)
                cacheKept          = (Test-Path $layout.Cache)
                noteExists         = (Test-Path -PathType Leaf $layout.DataNote)
                unrelatedShortcutKept = (Test-Path -PathType Leaf $fakeDesktopShortcut)
            }
        }
        'install-root-layout' {
            # Where the product may be installed, and where it may not.
            switch ($Variant) {
                'accept' {
                    $layout = Get-DeliveryLayout -InstallRoot $InstallRoot -LocalAppData $WorkRoot -Desktop $Desktop
                    $created = Initialize-DeliveryLayout -Layout $layout
                    return [ordered]@{
                        accepted        = $true
                        root            = $layout.Root
                        canonicalRoot   = (Get-CanonicalPath -Path $InstallRoot)
                        sameAsAsked     = ((Get-CanonicalPath -Path $layout.Root) -eq
                            (Get-CanonicalPath -Path $InstallRoot))
                        app             = $layout.App
                        data            = $layout.Data
                        runtime         = $layout.Runtime
                        stateFile       = $layout.StateFile
                        interpreter     = $layout.Interpreter
                        interpreterInside = ($layout.Interpreter -like "$($layout.Root)*")
                        inheritedLocalAppData = $layout.LocationFile
                        locationOutsideRoot = (-not $layout.LocationFile.StartsWith(
                                $layout.Root, [System.StringComparison]::OrdinalIgnoreCase))
                        createdCount    = @($created).Count
                        dataExists      = (Test-Path $layout.Data)
                        separateData    = (($layout.Data -ne $layout.App) -and ($layout.Data -ne $layout.Runtime))
                    }
                }
                'accept-short' {
                    $target = Join-Path $WorkRoot '长目录 名称 测试\AutoTuneStudio'
                    New-Item -ItemType Directory -Force -Path $target | Out-Null
                    $short = Get-ShortDirectoryPath -Path $target
                    $layout = Get-DeliveryLayout -InstallRoot $short -LocalAppData $WorkRoot -Desktop $Desktop
                    return [ordered]@{
                        shortAvailable = ($short -ne $target)
                        shortPath      = $short
                        longPath       = $target
                        root           = $layout.Root
                        isLongForm     = ($layout.Root -eq (Get-CanonicalPath -Path $target))
                    }
                }
                'reject' {
                    $captured = Invoke-Capture { Assert-InstallRootAllowed -Root $InstallRoot }
                    return [ordered]@{
                        accepted = $captured.Ok
                        code     = $captured.ErrorCode
                        message  = $captured.Message
                    }
                }
                'recommended' {
                    $recommended = Get-RecommendedInstallRoot
                    return [ordered]@{
                        recommended     = $recommended
                        isAbsolute      = [System.IO.Path]::IsPathRooted($recommended)
                        notSystemDrive  = (-not $recommended.StartsWith($env:SystemDrive,
                                [System.StringComparison]::OrdinalIgnoreCase))
                        leaf            = Split-Path -Path $recommended -Leaf
                    }
                }
                'advice' {
                    # What the operator is told about the directory they chose:
                    # a system-drive destination is allowed but never silent.
                    $advice = Get-InstallDestinationAdvice -Root $InstallRoot -Probe (New-FakeProbe) `
                        -RequiredFreeBytes 12884901888
                    return [ordered]@{
                        root          = [string]$advice.Root
                        onSystemDrive = [bool]$advice.OnSystemDrive
                        message       = [string]$advice.Message
                        mentionsRoot  = ([string]$advice.Message).Contains($InstallRoot)
                    }
                }
                'inspect' {
                    # resolves the layout without creating anything, so a
                    # candidate on the operator's own disk stays untouched
                    $layout = Get-DeliveryLayout -InstallRoot $InstallRoot -LocalAppData $WorkRoot `
                        -Desktop $Desktop
                    return [ordered]@{
                        accepted        = $true
                        root            = $layout.Root
                        sameAsAsked     = ((Get-CanonicalPath -Path $layout.Root) -eq
                            (Get-CanonicalPath -Path $InstallRoot))
                        app             = $layout.App
                        data            = $layout.Data
                        runtime         = $layout.Runtime
                        interpreter     = $layout.Interpreter
                        interpreterInside = ($layout.Interpreter -like "$($layout.Root)*")
                        separateData    = (($layout.Data -ne $layout.App) -and ($layout.Data -ne $layout.Runtime))
                    }
                }
                default { }
            }
        }
        'install-destination-prompt' {
            # Whether a *first* installation may ask the operator for a directory.
            # A double-clicked install.bat owns a console and asks; every
            # automated run — an explicit directory, the switch, the
            # non-interactive variable or redirected input — must proceed without
            # waiting for a human, and then falls back to the recommended disk.
            $recordedRoot = ''
            $arguments = @{
                Requested              = ''
                LocalAppData           = $WorkRoot
                AcceptRecommended      = $false
                NonInteractiveVariable = ''
                InputRedirected        = $false
            }
            switch ($Variant) {
                'console' { }
                'accept-recommended' { $arguments.AcceptRecommended = $true }
                'noninteractive' { $arguments.NonInteractiveVariable = '1' }
                'input-redirected' { $arguments.InputRedirected = $true }
                'requested' { $arguments.Requested = $InstallRoot }
                'recorded' {
                    # A second run finds the directory the installation recorded
                    # and never asks again.
                    $layout = Get-DeliveryLayout -InstallRoot (Join-Path $WorkRoot 'chosen\AutoTuneStudio') `
                        -LocalAppData $WorkRoot -Desktop $Desktop
                    Initialize-DeliveryLayout -Layout $layout | Out-Null
                    [void](Write-InstallLocation -Layout $layout)
                    $recordedRoot = $layout.Root
                }
                default { throw "unknown variant: $Variant" }
            }
            $plan = Get-InstallPromptPlan -Requested $arguments.Requested `
                -LocalAppData $arguments.LocalAppData `
                -AcceptRecommended $arguments.AcceptRecommended `
                -NonInteractiveVariable $arguments.NonInteractiveVariable `
                -InputRedirected $arguments.InputRedirected
            $recommended = Get-RecommendedInstallRoot
            return [ordered]@{
                variant      = $Variant
                ask          = [bool]$plan.Ask
                reason       = [string]$plan.Reason
                destination  = [string]$plan.Destination
                recordedRoot = $recordedRoot
                recommended  = $recommended
                # what install.ps1 does with this answer: it asks only when
                # ``ask`` is true, otherwise it takes the destination or, when
                # there is none, the recommended disk
                waitsForInput = [bool]$plan.Ask
                usesRecommended = ((-not [bool]$plan.Ask) -and
                    [string]::IsNullOrWhiteSpace([string]$plan.Destination) -and
                    -not [string]::IsNullOrWhiteSpace($recommended))
            }
        }
        'install-root-binding' {
            # Install away from LOCALAPPDATA: the installed entry points and the
            # discovery record must name that same root, never a second one.
            $target = Join-Path $WorkRoot 'chosen-root\AutoTuneStudio'
            $layout = Get-DeliveryLayout -InstallRoot $target -LocalAppData $WorkRoot -Desktop $Desktop
            $package = Join-Path $WorkRoot 'package'
            $null = Invoke-FullInstall -Layout $layout -PackageRoot $package
            $other = Join-Path $WorkRoot 'some-other-localappdata'
            New-Item -ItemType Directory -Force -Path $other | Out-Null
            $fromEntry = Resolve-DeliveryInstallRoot -LocalAppData $other -EntryRoot $layout.Root
            $fromRecord = Resolve-DeliveryInstallRoot -LocalAppData $WorkRoot -EntryRoot ''
            $fromNothing = Resolve-DeliveryInstallRoot -LocalAppData $other -EntryRoot ''
            $locationFile = Get-InstallLocationPath -LocalAppData $WorkRoot
            $env:AUTO_TUNE_NO_PAUSE = '1'
            try {
                # a misleading -LocalAppData: the entry point runs from the
                # installation and must reach the real start gate there
                $command = '"{0}" -LocalAppData "{1}" -Port {2}' -f $layout.StartBat, $other, $Port
                $output = & $env:ComSpec /c $command 2>&1 | Out-String
                $exitCode = $LASTEXITCODE
            } finally {
                Remove-Item Env:AUTO_TUNE_NO_PAUSE -ErrorAction SilentlyContinue
            }
            return [ordered]@{
                root            = $layout.Root
                locationFile    = $locationFile
                locationExists  = (Test-Path -PathType Leaf $locationFile)
                recordedRoot    = [string]((Get-Content $locationFile -Raw -Encoding UTF8 |
                        ConvertFrom-Json).root)
                fromEntry       = $fromEntry
                fromRecord      = $fromRecord
                fromNothing     = $fromNothing
                entryResolved   = ($fromEntry -eq $layout.Root)
                recordResolved  = ($fromRecord -eq $layout.Root)
                nothingResolved = ($fromNothing -eq '')
                exitCode        = [int]$exitCode
                reachedStartGate = ($output -like '*PORT_IN_USE*')
                wrongRootRejected = ($output -notlike '*INSTALL_STATE_INCOMPLETE*')
            }
        }
        'offline-prepare' {
            # The build-time preparation of the offline bundle. Everything that
            # reaches the outside world goes through the injected pip runner.
            # Every variant gets its own source and output tree: a shared "prepare"
            # directory would let an earlier variant's verified files satisfy a
            # later one (a missing wheel would still be there from the run before).
            $suffix = 'default'
            if (-not [string]::IsNullOrWhiteSpace($Variant)) { $suffix = $Variant }
            $fixture = New-PrepareFixture -Root (Join-Path $WorkRoot ('prepare-' + $suffix)) -Variant $Variant
            if ($Variant -eq 'missing-wheel') { $script:PipMode = 'missing-wheel' }
            if ($Variant -eq 'transitive-missing-wheel') { $script:PipMode = 'transitive-missing' }
            if ($Variant -eq 'transitive-missing-many') { $script:PipMode = 'transitive-missing-many' }
            if ($Variant -eq 'transient-download-failure') { $script:PipMode = 'transient' }
            $output = Join-Path $WorkRoot ('offline-out-' + $suffix)
            $script:Calls.Clear()
            $script:NetworkCalls.Clear()
            $captured = Invoke-Capture {
                Prepare-OfflineBundle -RepoRoot $fixture.RepoRoot -DependencySource $fixture.DependencySource `
                    -OutputDir $output -Python 'C:\fake\python.exe' -Runner (New-FakePipRunner)
            }
            $lockPath = Join-Path $output 'offline-lock.json'
            $lockText = if (Test-Path $lockPath) { Get-Content $lockPath -Raw -Encoding UTF8 } else { '' }
            $lock = $null
            if (Test-Path $lockPath) { $lock = Get-Content $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json }
            $files = @()
            if ($null -ne $lock) { $files = @($lock.files) }
            $pipCalls = @(Get-CallData 'run')
            $pipArgs = @($pipCalls | ForEach-Object { @($_.args) -join ' ' })
            # a second run must produce the same lock bytes
            $second = Invoke-Capture {
                Prepare-OfflineBundle -RepoRoot $fixture.RepoRoot -DependencySource $fixture.DependencySource `
                    -OutputDir $output -Python 'C:\fake\python.exe' -Runner (New-FakePipRunner)
            }
            $lockTextAfter = if (Test-Path $lockPath) { Get-Content $lockPath -Raw -Encoding UTF8 } else { '' }
            $script:PipMode = 'ok'
            # What the operator is shown, checked for the secrets and local paths
            # that only pip's own output carries.
            $reportText = [string]$captured.Message
            $directMissing = @()
            $fromPip = @()
            if ($null -ne $captured.Detail) {
                if ($null -ne $captured.Detail.Missing) {
                    $directMissing = @($captured.Detail.Missing | ForEach-Object { [string]$_.Name })
                }
                if ($null -ne $captured.Detail.MissingFromPip) {
                    $fromPip = @($captured.Detail.MissingFromPip | ForEach-Object { [string]$_ })
                }
                $reportText = $reportText + ' ' + ($captured.Detail | ConvertTo-Json -Compress -Depth 8)
            }
            $leaked = @($script:PrepareSecrets | Where-Object { $reportText -like ('*' + $_ + '*') })
            return [ordered]@{
                variant        = $Variant
                ok             = $captured.Ok
                code           = $captured.ErrorCode
                message        = $captured.Message
                messageHasAbsolutePath = ($reportText -match '[A-Za-z]:\\')
                leakedSecrets  = $leaked
                directMissing  = $directMissing
                missingFromPip = $fromPip
                detail         = $captured.Detail
                secondOk       = $second.Ok
                deterministic  = ($lockText -eq $lockTextAfter -and $lockText -ne '')
                lockExists     = (Test-Path -PathType Leaf $lockPath)
                lockFiles      = @($files | ForEach-Object { [string]$_.path })
                lockPurposes   = @($files | ForEach-Object { [string]$_.purpose })
                lockHasAbsolutePath = ($lockText -match '[A-Za-z]:\\')
                requirementsSha = if ($null -ne $lock) { [string]$lock.requirements_sha256 } else { '' }
                expectedRequirementsSha = (Get-Sha256Hex -Path (Join-Path $fixture.RepoRoot 'windows\requirements-windows.lock.txt'))
                wheelhouseFiles = @(Get-ChildItem -Path (Join-Path $output 'wheelhouse') -File -ErrorAction SilentlyContinue |
                    ForEach-Object { $_.Name })
                minicondaCopied = (Test-Path -PathType Leaf (Join-Path $output ('miniconda\' + $script:OfflineMiniconda)))
                minicondaHash  = Get-TextSha256 -Path (Join-Path $output ('miniconda\' + $script:OfflineMiniconda))
                minicondaSourceHash = Get-TextSha256 -Path (Join-Path $fixture.DependencySource $script:OfflineMiniconda)
                pipArgs        = $pipArgs
                pipFile        = if (@($pipCalls).Count -gt 0) { [string]$pipCalls[0].file } else { '' }
                torchDownloads = @(Get-CallData 'run' | ForEach-Object {
                        $text = @($_.args) -join ' '
                        if ($text -like '*' + $script:OfflineTorch + '*') { $text }
                    })
                fetchCalls     = @($script:NetworkCalls).Count
            }
        }
        'wheel-name-rules' {
            # What may be installed into the private runtime. Each answer is the
            # shipping rule, so a foreign or CPU wheel is refused by name alone.
            $names = @(
                'fastapi-0.139.2-py3-none-any.whl',
                'pytz-2025.2-py2.py3-none-any.whl',
                'contourpy-1.3.2-cp310-cp310-win_amd64.whl',
                'greenlet-3.5.5-cp310-cp310-win_amd64.whl',
                'humanfriendly-10.0-py2.py3-none-any.whl',
                'torch-2.5.1+cu121-cp310-cp310-win_amd64.whl',
                'torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl',
                'torch-2.5.1-cp310-cp310-win_amd64.whl',
                'torchvision-0.20.1-cp310-cp310-win_amd64.whl',
                'fastapi-0.139.2-py3-none-manylinux1_x86_64.whl',
                'fastapi-0.139.2-py3-none-macosx_11_0_arm64.whl',
                'uvicorn-0.51.0-cp39-cp39-win_amd64.whl',
                'uvicorn-0.51.0-cp311-cp311-win_amd64.whl',
                'uvicorn-0.51.0-cp311-abi3-win_amd64.whl',
                # An abi3 tag names the *oldest* CPython that can load the wheel,
                # so a minimum at or below 3.10 installs into the private runtime:
                # opencv and psutil really ship cp37-abi3 builds, MarkupSafe
                # cp39-abi3. The cp310-abi3 row probes the accepted boundary.
                'opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl',
                'psutil-7.0.0-cp37-abi3-win_amd64.whl',
                'MarkupSafe-3.0.2-cp39-abi3-win_amd64.whl',
                'cryptography-44.0.1-cp310-abi3-win_amd64.whl',
                # Not abi3: a cp37-cp37 wheel needs CPython 3.7's ABI exactly.
                'uvicorn-0.51.0-cp37-cp37-win_amd64.whl',
                'numpy-2.2.6-cp310-cp310-win32.whl',
                'uvicorn-0.51.0.tar.gz',
                'fastapi-0.139.2.zip'
            )
            $results = New-Object System.Collections.ArrayList
            foreach ($name in $names) {
                $checked = Test-OfflineWheelFileName -FileName $name
                [void]$results.Add(@{
                        name   = $name
                        ok     = [bool]$checked.Ok
                        reason = ([string]$checked.Reason)
                        project = [string]$checked.Project
                    })
            }
            return [ordered]@{ results = @($results) }
        }
        'zip-fixture-source' {
            # Writes the synthetic delivery source the ZIP suite builds from:
            # the real delivery scripts and a complete offline bundle, described
            # by a manifest the shipping generators produced. The 2.4 GB torch
            # wheel is replaced by a stand-in of the same name, so the archive
            # can be built and inspected without shipping gigabytes in a test.
            $repo = $TargetRoot
            New-Item -ItemType Directory -Force -Path $repo | Out-Null
            $windows = Join-Path $repo 'windows'
            if (Test-Path $windows) { Remove-Item -Path $windows -Recurse -Force }
            Copy-Item -Path $script:WindowsSource -Destination $windows -Recurse -Force
            # the local dependency folder: a build input that must never be
            # archived
            [void](Write-FillerFile -Path (Join-Path (Join-Path $repo '依赖') $script:OfflineMiniconda) `
                    -Size 4096 -Seed 11)
            # a synthetic (short) dependency lock: the ordinary wheels are derived
            # from it, so the bundle is complete by construction
            $lockContent = "fastapi==0.139.2`nuvicorn==0.51.0`n"
            Set-Content -Path (Join-Path $windows 'requirements-windows.lock.txt') -Encoding UTF8 `
                -Value $lockContent
            $offline = Join-Path $repo 'offline_cache'
            $null = New-TestOfflineBundle -OfflineRoot $offline `
                -RequirementsPath (Join-Path $windows 'requirements-windows.lock.txt') -CoreSize 4096
            $paths = Get-TestOfflinePaths -OfflineRoot $offline
            $manifest = New-TestManifest `
                -MinicondaSha (Get-Sha256Hex -Path $paths.Miniconda) `
                -MinicondaSize (Get-Item -LiteralPath $paths.Miniconda).Length `
                -TorchSha (Get-Sha256Hex -Path $paths.Torch) `
                -TorchSize (Get-Item -LiteralPath $paths.Torch).Length `
                -TorchvisionSha (Get-Sha256Hex -Path $paths.Torchvision) `
                -TorchvisionSize (Get-Item -LiteralPath $paths.Torchvision).Length `
                -LockSha (Get-Sha256Hex -Path (Join-Path $windows 'requirements-windows.lock.txt'))
            Write-JsonFile -Path (Join-Path $windows 'package-manifest.json') -Object $manifest | Out-Null
            Write-TestOfflineLock -ManifestRoot $windows -OfflineRoot $offline | Out-Null
            return [ordered]@{
                repo        = $repo
                offline     = $offline
                windows     = $windows
                wheelhouse  = $paths.Wheelhouse
                lockFiles   = @(Get-ChildItem -Path $paths.Wheelhouse -File | ForEach-Object { $_.Name })
            }
        }
        default {
            throw "unknown scenario $Name"
        }
    }
}

try {
    $facts = Invoke-Scenario -Name $Scenario
    if ($facts -is [System.Array]) {
        # A bare helper call that emits a value makes the facts an array, and the
        # failure then surfaces far away (``$facts['scenario_ok']`` cannot index a
        # string into an array). Say so here instead.
        throw ("the scenario leaked {0} extra pipeline value(s); assign helper calls with `$null =" -f (@($facts).Count - 1))
    }
    if ($null -eq $facts) { $facts = [ordered]@{} }
    # ``scenario_ok`` marks "the scenario ran to completion"; the facts keep
    # whatever ``ok`` the operation itself reported.
    $facts['scenario_ok'] = $true
    $facts['scenario'] = $Scenario
    $result = [ordered]@{ scenario = $Scenario; ok = $true; error_code = $null; error_message = $null; facts = $facts; calls = Get-CallNames }
} catch {
    $info = Get-DeliveryErrorInfo -ErrorRecord $_
    $result = [ordered]@{ scenario = $Scenario; ok = $false; error_code = $info.Code;
        error_message = $info.Message; facts = [ordered]@{}; calls = Get-CallNames }
}

Write-Output ('##RESULT## ' + ($result | ConvertTo-Json -Depth 12 -Compress))
