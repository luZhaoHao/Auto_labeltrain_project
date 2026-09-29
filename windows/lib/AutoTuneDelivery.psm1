# Auto-Tune Studio — Windows delivery layer (F1.2-D).
#
# One installation, four operations: install, start, upgrade, uninstall. The
# module owns every rule that is worth testing — the layout, the verified
# download cache, the payload whitelist, the configuration bootstrap, the
# versioned state file, the start gates and the removal safety — so install.bat,
# start.bat, upgrade.bat and uninstall.bat stay thin and a test can drive the
# real logic with injected probes.
#
# Boundaries that belong to the operator's machine are injectable and never
# re-implemented here:
#   * who downloads   -> $Downloader   (default: HTTPS only, official hosts)
#   * what runs       -> $Runner / $Launcher (default: the private interpreter)
#   * what the machine answers -> $Probe (disk, driver, port, PID, health)
# The GPU question is *not* answered here: it is asked of the shared
# auto_tune.delivery.preflight that the container also uses.
#
# This file is UTF-8 **with BOM**: Windows PowerShell 5.1 reads a BOM-less
# script in the ANSI code page, which would break the Chinese messages.

$ErrorActionPreference = 'Stop'

$script:ErrorPrefix = 'AUTOTUNE_DELIVERY|'
$script:TrustedDownloadHosts = @('repo.anaconda.com', 'download.pytorch.org', 'pypi.org', 'files.pythonhosted.org')

# Files that may never ship inside the package, whatever the manifest says.
# ``credentials.json`` is the container's persisted key file (Linux file
# backend): it is operator data, and it must never travel in a package.
$script:PayloadDeniedNames = @('config.yaml', '.env', '.env.local', 'secrets.json',
    'credentials.json')
$script:PayloadDeniedSegments = @('__pycache__', '.git', '.pytest_cache', '.mypy_cache',
    'node_modules', 'docker-data', 'build_output', 'staging')
$script:PayloadDeniedExtensions = @('.pt', '.pth', '.onnx', '.engine', '.db', '.db-wal',
    '.db-shm', '.db-journal', '.pyc', '.pyo', '.zip', '.tar', '.gz', '.bak')

# ── errors ──────────────────────────────────────────────────────────────────
# PowerShell classes are not visible to a caller that merely imports a module,
# so the stable code travels inside the exception message as ``CODE|message``
# and Get-DeliveryErrorInfo is the only reader. Entry scripts and the test
# harness both use it, which is why the codes stay comparable everywhere.

function New-DeliveryFailure {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Message,
        $Detail = $null
    )
    $exception = New-Object System.Exception ("$script:ErrorPrefix$Code|$Message")
    if ($null -ne $Detail) { $exception.Data['detail'] = $Detail }
    return $exception
}

function Get-DeliveryErrorInfo {
    param($ErrorRecord)
    $exception = $null
    if ($ErrorRecord -is [System.Management.Automation.ErrorRecord]) {
        $exception = $ErrorRecord.Exception
    } elseif ($ErrorRecord -is [System.Exception]) {
        $exception = $ErrorRecord
    }

    $message = ''
    $detail = $null
    if ($null -ne $exception) {
        $message = [string]$exception.Message
        if ($exception.Data -and $exception.Data.Contains('detail')) {
            $detail = $exception.Data['detail']
        }
    }

    $code = 'HARNESS_ERROR'
    if ($message.StartsWith($script:ErrorPrefix)) {
        $rest = $message.Substring($script:ErrorPrefix.Length)
        $separator = $rest.IndexOf('|')
        if ($separator -ge 0) {
            $code = $rest.Substring(0, $separator)
            $message = $rest.Substring($separator + 1)
        }
    }
    return @{ Code = $code; Message = $message; Detail = $detail }
}

# ── logging ─────────────────────────────────────────────────────────────────
# Everything the delivery writes to a log goes through here: secrets, foreign
# absolute paths and stack traces are removed before the line reaches the disk.

function ConvertTo-SafeLogLine {
    param([string]$Message, [string]$InstallRoot = '')
    if ([string]::IsNullOrEmpty($Message)) { return '' }

    $line = $Message -replace "`r?`n", ' '
    $line = [regex]::Replace($line,
        '(?i)\b(api[_-]?key|apikey|authorization|token|secret|password|credential)\b\s*[:=]\s*\S+',
        '$1=REDACTED')
    $line = [regex]::Replace($line, 'sk-[A-Za-z0-9_\-]{6,}', 'sk-REDACTED')
    $line = [regex]::Replace($line, '(?i)-{2,}\s*End of inner exception.*$', '<stacktrace>')
    $line = [regex]::Replace($line, '\s+at\s+[A-Za-z_][\w\.]*[\.\(].*$', ' <stacktrace>')

    $restore = $false
    if (-not [string]::IsNullOrEmpty($InstallRoot)) {
        $escaped = [regex]::Escape($InstallRoot.TrimEnd('\'))
        $line = [regex]::Replace($line, $escaped, '@@INSTALLROOT@@', 'IgnoreCase')
        $restore = $true
    }
    $line = [regex]::Replace($line, '(?<![\w-])[A-Za-z]:\\[^\s"'',;\)]*', '<path>')
    $line = [regex]::Replace($line, '(?<![\w-])\\\\[^\s"'',;\)]*', '<path>')
    if ($restore) {
        $line = [regex]::Replace($line, '@@INSTALLROOT@@', $InstallRoot.TrimEnd('\'), 'IgnoreCase')
    }

    if ($line.Length -gt 400) { $line = $line.Substring(0, 400) + '...' }
    return $line.Trim()
}

function Write-DeliveryLog {
    param(
        [string]$LogFile,
        [string]$Message,
        [string]$Level = 'INFO',
        [string]$InstallRoot = ''
    )
    $line = "{0} [{1}] {2}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'),
        $Level, (ConvertTo-SafeLogLine -Message $Message -InstallRoot $InstallRoot)

    if (-not [string]::IsNullOrEmpty($LogFile)) {
        $directory = Split-Path -Path $LogFile -Parent
        if ($directory -and -not (Test-Path $directory)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
        try {
            [System.IO.File]::AppendAllText($LogFile, $line + "`r`n",
                (New-Object System.Text.UTF8Encoding($false)))
        } catch {
            Write-Host $line
            return
        }
    }
    Write-Host $line
}

# ── probes ──────────────────────────────────────────────────────────────────

function Invoke-DeliveryProbe {
    param($Probe, [string]$Name, [scriptblock]$Fallback, [object[]]$Arguments = @())
    if ($null -ne $Probe) {
        if ($Probe -is [System.Collections.IDictionary]) {
            if ($Probe.Contains($Name)) { return & $Probe[$Name] @Arguments }
        } else {
            $property = $Probe.PSObject.Properties[$Name]
            if ($null -ne $property) { return & $property.Value @Arguments }
        }
    }
    return & $Fallback @Arguments
}

function Get-CanonicalPath {
    # One spelling per path, whatever the caller wrote. The file system is asked
    # for the *long* form instead of the string being reshaped: an 8.3 name
    # (for example, a short user-profile path) and its long form name the same directory, and
    # the length of one has nothing to do with the other, so a relative path
    # derived from a length is only correct when both sides are spelled the
    # way the file system spells them. Trailing separators, "." and ".." and a
    # lower-case drive letter collapse here as well.
    param([Parameter(Mandatory = $true)][string]$Path)

    $full = [System.IO.Path]::GetFullPath($Path)
    $resolved = ''
    try {
        $item = Get-Item -LiteralPath $full -Force -ErrorAction Stop
        if ($null -ne $item) { $resolved = [string]$item.FullName }
    } catch {
        $resolved = ''
    }
    # A path that does not exist yet has no on-disk spelling to ask for.
    if ([string]::IsNullOrWhiteSpace($resolved)) { $resolved = $full }

    $root = [System.IO.Path]::GetPathRoot($resolved)
    if (-not [string]::IsNullOrEmpty($root) -and
        ($resolved.TrimEnd('\', '/') -eq $root.TrimEnd('\', '/'))) {
        # 'C:\' and '\\server\share\' are roots: trimming them would name
        # something else entirely.
        return $root
    }
    return $resolved.TrimEnd('\', '/')
}

function Get-RelativePathUnderRoot {
    # The path of $Path inside $Root with the separator of the platform, or
    # $null when $Path is not below $Root. Both sides are canonicalised first,
    # so the length used for the slice is the length of the spelling the file
    # system returned for both of them.
    param([Parameter(Mandatory = $true)][string]$Root, [Parameter(Mandatory = $true)][string]$Path)

    $canonicalRoot = Get-CanonicalPath -Path $Root
    $canonicalPath = Get-CanonicalPath -Path $Path
    $separator = [string][System.IO.Path]::DirectorySeparatorChar
    $prefix = $canonicalRoot
    if (-not $prefix.EndsWith($separator)) { $prefix = $prefix + $separator }
    if (-not $canonicalPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }
    return $canonicalPath.Substring($prefix.Length)
}

function Test-Is64BitWindows {
    return [bool][System.Environment]::Is64BitOperatingSystem
}

function Get-FreeDiskBytes {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($full)
    $drive = New-Object System.IO.DriveInfo($root)
    return [long]$drive.AvailableFreeSpace
}

function Test-NvidiaDriverPresent {
    $controllers = $null
    try {
        $controllers = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop)
    } catch {
        try { $controllers = @(Get-WmiObject -Class Win32_VideoController -ErrorAction Stop) }
        catch { return $false }
    }
    foreach ($controller in $controllers) {
        if ($controller.Name -and ([string]$controller.Name -like '*NVIDIA*')) { return $true }
    }
    return $false
}

function Test-ProcessIdentity {
    param([int]$ProcessId, [string]$StartedAt)
    try { $process = Get-Process -Id $ProcessId -ErrorAction Stop }
    catch { return $false }
    try {
        $started = $process.StartTime.ToUniversalTime().ToString('o')
        if (-not [string]::IsNullOrEmpty($StartedAt) -and $started -ne $StartedAt) { return $false }
    } catch {
        return $false
    }
    return $true
}

function Test-TcpPortInUse {
    param([int]$Port, [string]$ComputerName = '127.0.0.1')
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($ComputerName, $Port, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(1000, $false)
        if (-not $connected) { return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Test-HealthEndpoint {
    param([int]$Port, [int]$TimeoutMs = 2000)
    try {
        $response = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/healthz" -f $Port) `
            -UseBasicParsing -TimeoutSec ([math]::Max(1, [int]($TimeoutMs / 1000)))
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Open-DefaultBrowser {
    param([string]$Url)
    try {
        Start-Process $Url | Out-Null
    } catch {
        Write-DeliveryLog -Message "无法自动打开浏览器，请手动访问 $Url"
    }
    return $true
}

# ── layout ──────────────────────────────────────────────────────────────────
# Program, private runtime, installer cache, logs and user data live next to
# each other but never *inside* each other: reinstalling or removing the program
# must not touch anything the operator produced.

function Get-UserDesktopPath {
    try {
        $path = [System.Environment]::GetFolderPath(
            [System.Environment+SpecialFolder]::DesktopDirectory)
        if (-not [string]::IsNullOrWhiteSpace($path)) { return $path }
    } catch { }
    return ''
}

function Get-LauncherEntryNames {
    return @('start.bat', 'start.ps1', 'uninstall.bat', 'uninstall.ps1')
}

function Get-LauncherModuleRelative {
    return 'lib\AutoTuneDelivery.psm1'
}

function Get-LauncherRelativePaths {
    return @(Get-LauncherEntryNames) + @(Get-LauncherModuleRelative)
}

function Get-DeliveryLayout {
    # The installation directory is either given explicitly — the operator's own
    # choice, validated by Assert-InstallRootAllowed — or, without one, the
    # per-user default. Both are the same layout object, so every entry point and
    # every test sees one shape.
    param(
        [string]$LocalAppData = $env:LOCALAPPDATA,
        [int]$Port = 8000,
        [string]$Desktop = '',
        [string]$InstallRoot = ''
    )
    $root = ''
    if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
        $root = Assert-InstallRootAllowed -Root $InstallRoot
    } else {
        if ([string]::IsNullOrWhiteSpace($LocalAppData)) {
            throw (New-DeliveryFailure -Code 'LOCALAPPDATA_MISSING' `
                    -Message '无法确定 LOCALAPPDATA。请在普通用户桌面会话中运行安装程序，不要以管理员或服务账户运行。')
        }
        $base = [System.IO.Path]::GetFullPath($LocalAppData)
        $root = Join-Path $base 'AutoTuneStudio'
    }
    $data = Join-Path $root 'data'
    $runtime = Join-Path $root 'runtime'
    $launcher = Join-Path $root 'lib'

    # The shortcut is the only thing this delivery puts outside its own folder.
    # The desktop is always the current user's and is never created here.
    if ([string]::IsNullOrWhiteSpace($Desktop)) { $Desktop = Get-UserDesktopPath }
    $shortcut = ''
    if (-not [string]::IsNullOrWhiteSpace($Desktop)) {
        $shortcut = Join-Path ([System.IO.Path]::GetFullPath($Desktop)) 'Auto Tune Studio.lnk'
    }

    return [pscustomobject]@{
        Root            = $root
        App             = Join-Path $root 'app'
        Runtime         = $runtime
        Interpreter     = Join-Path (Join-Path $runtime 'py310') 'python.exe'
        CondaHome       = Join-Path $runtime 'py310'
        Data            = $data
        ConfigDir       = Join-Path $data 'config'
        ConfigPath      = Join-Path (Join-Path $data 'config') 'config.yaml'
        Datasets        = Join-Path $data 'datasets'
        Cache           = Join-Path $root 'installer-cache'
        Logs            = Join-Path $root 'logs'
        StateFile       = Join-Path $root 'install-state.json'
        DataNote        = Join-Path $root 'UNINSTALL-DATA-NOTE.txt'
        Launcher        = $launcher
        LibModule       = Join-Path $launcher 'AutoTuneDelivery.psm1'
        StartBat        = Join-Path $root 'start.bat'
        StartScript     = Join-Path $root 'start.ps1'
        UninstallBat    = Join-Path $root 'uninstall.bat'
        UninstallScript = Join-Path $root 'uninstall.ps1'
        Desktop         = $Desktop
        Shortcut        = $shortcut
        LocationFile    = (Get-InstallLocationPath -LocalAppData $LocalAppData)
        Port            = $Port
    }
}

function Get-LayoutDataDirectories {
    param($Layout)
    return @(
        $Layout.Data,
        (Join-Path $Layout.Data 'config'),
        (Join-Path $Layout.Data 'log'),
        (Join-Path $Layout.Data 'detect'),
        (Join-Path $Layout.Data 'runs'),
        (Join-Path $Layout.Data 'models\weights'),
        $Layout.Datasets
    )
}

function Initialize-DeliveryLayout {
    param([Parameter(Mandatory = $true)]$Layout)

    $created = New-Object System.Collections.ArrayList
    $directories = @($Layout.App, $Layout.Runtime, $Layout.Cache, $Layout.Logs, $Layout.Launcher) +
        @(Get-LayoutDataDirectories -Layout $Layout)

    foreach ($directory in $directories) {
        if (Test-Path $directory) {
            if (-not (Test-Path -PathType Container $directory)) {
                throw (New-DeliveryFailure -Code 'DIR_NOT_WRITABLE' `
                        -Message ("{0} 已经存在且不是目录，安装无法继续。" -f $directory))
            }
            continue
        }
        try {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        } catch {
            throw (New-DeliveryFailure -Code 'DIR_NOT_WRITABLE' `
                    -Message ("无法创建目录 {0}。请检查该位置是否存在同名文件或权限限制。" -f $directory))
        }
        [void]$created.Add($directory)
    }
    return @($created)
}

function Assert-DirectoryWritable {
    param($Layout, $Probe = $null)
    $directories = @($Layout.Root, $Layout.Logs, $Layout.Cache, $Layout.Launcher) +
        @(Get-LayoutDataDirectories -Layout $Layout)
    foreach ($directory in $directories) {
        $writable = Invoke-DeliveryProbe -Probe $Probe -Name 'IsWritable' `
            -Fallback { param($Path) Test-DirectoryWritable -Path $Path } -Arguments @($directory)
        if (-not $writable) {
            throw (New-DeliveryFailure -Code 'DIR_NOT_WRITABLE' `
                    -Message ("目录不可写：{0}。请检查权限后重试。" -f $directory))
        }
    }
    return $true
}

function Test-DirectoryWritable {
    param([string]$Path)
    $probe = Join-Path $Path ('.write-probe-' + [guid]::NewGuid().ToString('n'))
    try {
        [System.IO.File]::WriteAllText($probe, 'probe')
        Remove-Item -Path $probe -Force -ErrorAction SilentlyContinue
        return $true
    } catch {
        return $false
    }
}

function Assert-InstallPreconditions {
    # Everything a machine must provide *whatever* the run does. The disk budget
    # is deliberately not part of this: it is only owed when something is about
    # to be built, so it lives in Assert-SufficientFreeBytes.
    param(
        $Layout,
        $Probe = $null
    )

    $is64 = Invoke-DeliveryProbe -Probe $Probe -Name 'Is64BitOS' -Fallback { Test-Is64BitWindows }
    if (-not $is64) {
        throw (New-DeliveryFailure -Code 'NOT_64BIT_WINDOWS' -Message '仅支持 64 位 Windows。')
    }

    $driver = Invoke-DeliveryProbe -Probe $Probe -Name 'DriverPresent' -Fallback { Test-NvidiaDriverPresent }
    if (-not $driver) {
        throw (New-DeliveryFailure -Code 'NVIDIA_DRIVER_MISSING' `
                -Message '未检测到 NVIDIA 显卡驱动。Auto-Tune Studio 只提供 GPU 训练，不提供 CPU 回退；请先安装兼容的 NVIDIA 驱动。')
    }
    return $true
}

function Assert-SufficientFreeBytes {
    # The budget for a *fresh* installation: a private runtime and the program
    # files. Reusing or repairing an installation costs no such space, so this is
    # only ever called on the paths that are about to write a new runtime or a
    # new program.
    param(
        [Parameter(Mandatory = $true)]$Layout,
        $Probe = $null,
        [long]$RequiredFreeBytes = 0
    )

    if ($RequiredFreeBytes -le 0) { return $true }
    $free = [long](Invoke-DeliveryProbe -Probe $Probe -Name 'FreeBytes' `
            -Fallback { param($Path) Get-FreeDiskBytes -Path $Path } -Arguments @($Layout.Root))
    if ($free -lt $RequiredFreeBytes) {
        throw (New-DeliveryFailure -Code 'DISK_SPACE_INSUFFICIENT' `
                -Message ("磁盘可用空间不足：安装约需 {0:N1} GB，当前可用 {1:N1} GB。请清理磁盘后重新运行 install.bat。" -f `
                    ($RequiredFreeBytes / 1GB), ($free / 1GB)))
    }
    return $true
}

# ── package manifest, payload rules and integrity ───────────────────────────

function Get-PackageManifest {
    param([Parameter(Mandatory = $true)][string]$PackageRoot)
    $path = Join-Path $PackageRoot 'package-manifest.json'
    if (-not (Test-Path -PathType Leaf $path)) {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message "安装包缺少 package-manifest.json：$path")
    }
    try {
        $manifest = Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message 'package-manifest.json 不是有效的 JSON。')
    }
    [void](Assert-PackageManifest -Manifest $manifest -PackageRoot $PackageRoot)
    return $manifest
}

function Assert-PackageManifest {
    param($Manifest, [string]$PackageRoot = '')

    $missing = New-Object System.Collections.ArrayList
    foreach ($field in @('schema_version', 'product', 'version', 'python_version', 'payload', 'runtime', 'install')) {
        if ($null -eq $Manifest.PSObject.Properties[$field]) { [void]$missing.Add($field) }
    }
    if (@($missing).Count -gt 0) {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message ("package-manifest.json 缺少字段：{0}" -f (@($missing) -join ', ')))
    }
    if ([string]$Manifest.schema_version -ne '1.0') {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message ("不支持的 package-manifest 版本：{0}" -f $Manifest.schema_version))
    }

    $installer = $Manifest.runtime.conda_installer
    if ($null -eq $installer -or [string]::IsNullOrWhiteSpace([string]$installer.url) -or
        [string]::IsNullOrWhiteSpace([string]$installer.sha256)) {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message 'package-manifest.json 没有固定运行环境的官方下载地址与 SHA-256。')
    }
    Assert-TrustedDownloadUrl -Url ([string]$installer.url) -Manifest $Manifest

    if ([string]::IsNullOrWhiteSpace([string]$Manifest.runtime.pip_requirements) -or
        [string]::IsNullOrWhiteSpace([string]$Manifest.runtime.pip_requirements_sha256)) {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message 'package-manifest.json 没有固定依赖清单及其 SHA-256。')
    }
    if ($null -ne $Manifest.runtime.torch -and
        -not [string]::IsNullOrWhiteSpace([string]$Manifest.runtime.torch.index_url)) {
        # Provenance only: the delivery no longer fetches anything, but the
        # address the pinned wheels came from stays recorded and controlled.
        Assert-TrustedDownloadUrl -Url ([string]$Manifest.runtime.torch.index_url) -Manifest $Manifest
    }

    $offline = $Manifest.runtime.offline
    foreach ($field in @('directory', 'miniconda_directory', 'wheelhouse_directory', 'lock_file')) {
        if ($null -eq $offline -or [string]::IsNullOrWhiteSpace([string]$offline.$field)) {
            throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                    -Message 'package-manifest.json 没有声明包内离线依赖目录结构。')
        }
    }
    $wheels = @($Manifest.runtime.torch.wheels | Where-Object { $null -ne $_ })
    if (@($wheels).Count -lt 2) {
        throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                -Message 'package-manifest.json 没有固定包内 CUDA PyTorch 离线 wheel。')
    }
    foreach ($wheel in $wheels) {
        if ([string]::IsNullOrWhiteSpace([string]$wheel.file_name) -or
            [string]::IsNullOrWhiteSpace([string]$wheel.sha256) -or
            [string]::IsNullOrWhiteSpace([string]$wheel.purpose) -or
            [string]::IsNullOrWhiteSpace([string]$wheel.version)) {
            throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                    -Message 'package-manifest.json 的离线 wheel 缺少文件名、版本、大小或 SHA-256。')
        }
    }

    if (-not [string]::IsNullOrEmpty($PackageRoot)) {
        $lockFile = Join-Path $PackageRoot ([string]$Manifest.runtime.pip_requirements)
        if (-not (Test-Path -PathType Leaf $lockFile)) {
            throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                    -Message ("安装包缺少依赖清单 {0}。" -f $Manifest.runtime.pip_requirements))
        }
        if (-not (Test-FileHashMatches -Path $lockFile -Sha256 ([string]$Manifest.runtime.pip_requirements_sha256))) {
            throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                    -Message '依赖清单与 package-manifest.json 记录的 SHA-256 不一致。')
        }
    }
    return $true
}

function Assert-TrustedDownloadUrl {
    param([string]$Url, $Manifest = $null)

    if (-not $Url.StartsWith('https://')) {
        throw (New-DeliveryFailure -Code 'DOWNLOAD_INSECURE_URL' `
                -Message '安装包只允许从 HTTPS 官方地址下载，不会回退到 HTTP 或第三方镜像。')
    }
    $uri = $null
    try { $uri = [System.Uri]$Url } catch {
        throw (New-DeliveryFailure -Code 'DOWNLOAD_INSECURE_URL' -Message '安装包记录了无效的下载地址。')
    }
    $allowed = $script:TrustedDownloadHosts
    if ($null -ne $Manifest -and $Manifest.PSObject.Properties['runtime'] -and
        $Manifest.runtime.PSObject.Properties['trusted_hosts']) {
        $allowed = @($Manifest.runtime.trusted_hosts)
    }
    if (-not ($allowed -contains $uri.Host)) {
        throw (New-DeliveryFailure -Code 'DOWNLOAD_UNTRUSTED_HOST' `
                -Message ("安装包记录的下载地址主机 {0} 不在受控官方来源列表中，已拒绝使用。" -f $uri.Host))
    }
    return $true
}

function Get-FileSha256 {
    # The CLR, not the shell. ``Get-FileHash`` comes from
    # Microsoft.PowerShell.Utility, which is resolved through PSModulePath: on a
    # machine that also has PowerShell 7 installed a 5.1 host can load the
    # incompatible module and the cmdlet disappears. System.Security.Cryptography
    # is always there, so every hash in the delivery goes through here.
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = $null
    $hasher = $null
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        $digest = $hasher.ComputeHash($stream)
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
        if ($null -ne $hasher) { $hasher.Dispose() }
    }
    return ([System.BitConverter]::ToString($digest) -replace '-', '').ToLower()
}

function Get-FileSha256OrEmpty {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return '' }
    try { return (Get-FileSha256 -Path $Path) } catch { return '' }
}

function Test-FileHashMatches {
    param([string]$Path, [string]$Sha256)
    if ([string]::IsNullOrWhiteSpace($Sha256)) { return $false }
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return $false }
    try { $actual = Get-FileSha256 -Path $Path }
    catch { return $false }
    return ($actual -eq ([string]$Sha256).ToLower())
}

function Test-PayloadPath {
    param([string]$RelativePath, $Manifest)

    $normalized = ([string]$RelativePath).Replace('\', '/').TrimStart('/')
    if ([string]::IsNullOrWhiteSpace($normalized)) { return $false }

    $segments = $normalized.Split('/')
    foreach ($segment in $segments) {
        if ($script:PayloadDeniedSegments -contains $segment.ToLower()) { return $false }
    }
    $leaf = $segments[-1]
    if ($script:PayloadDeniedNames -contains $leaf.ToLower()) { return $false }
    if ($leaf.ToLower().StartsWith('.env.')) { return $false }
    $extension = [System.IO.Path]::GetExtension($normalized).ToLower()
    if ($script:PayloadDeniedExtensions -contains $extension) { return $false }

    $included = $false
    foreach ($pattern in @($Manifest.payload.include)) {
        if (Test-GlobMatch -Path $normalized -Pattern ([string]$pattern)) { $included = $true }
    }
    if (-not $included) { return $false }
    foreach ($pattern in @($Manifest.payload.exclude)) {
        if (Test-GlobMatch -Path $normalized -Pattern ([string]$pattern)) { return $false }
    }
    return $true
}

function ConvertTo-GlobRegex {
    param([string]$Pattern)
    # ``**/`` crosses directory boundaries, ``*`` and ``?`` stay inside one
    # segment, everything else is literal.
    $escaped = [regex]::Escape($Pattern)
    $escaped = $escaped.Replace('\*\*/', '(?:.*/)?')
    $escaped = $escaped.Replace('/\*\*', '(?:/.*)?')
    $escaped = $escaped.Replace('\*\*', '.*')
    $escaped = $escaped.Replace('\*', '[^/]*')
    $escaped = $escaped.Replace('\?', '[^/]')
    return '^' + $escaped + '$'
}

function Test-GlobMatch {
    param([string]$Path, [string]$Pattern)
    if ([string]::IsNullOrWhiteSpace($Pattern)) { return $false }
    $normalizedPattern = $Pattern.Replace('\', '/').TrimStart('/')
    if ($normalizedPattern -eq '*') { return $true }
    if (-not $normalizedPattern.Contains('*') -and -not $normalizedPattern.Contains('?')) {
        # a plain directory or file name also covers everything below it
        return ($Path -eq $normalizedPattern -or $Path.StartsWith($normalizedPattern))
    }
    return [bool][regex]::IsMatch($Path, (ConvertTo-GlobRegex -Pattern $normalizedPattern))
}

function Test-ScriptPath {
    param([string]$RelativePath, $Manifest)

    $normalized = ([string]$RelativePath).Replace('\', '/').TrimStart('/')
    if ([string]::IsNullOrWhiteSpace($normalized)) { return $false }
    foreach ($segment in $normalized.Split('/')) {
        if ($script:PayloadDeniedSegments -contains $segment.ToLower()) { return $false }
    }
    $leaf = [System.IO.Path]::GetFileName($normalized).ToLower()
    if ($leaf.StartsWith('.env')) { return $false }
    $extension = [System.IO.Path]::GetExtension($normalized).ToLower()
    if (@('.zip', '.exe', '.pt', '.onnx', '.db', '.pyc', '.log', '.bak') -contains $extension) { return $false }

    $included = $false
    foreach ($pattern in @($Manifest.package.scripts_include)) {
        if (Test-GlobMatch -Path $normalized -Pattern ([string]$pattern)) { $included = $true }
    }
    if (-not $included) { return $false }
    foreach ($pattern in @($Manifest.package.scripts_exclude)) {
        if (Test-GlobMatch -Path $normalized -Pattern ([string]$pattern)) { return $false }
    }
    return $true
}

function Get-PackageFileList {
    param([Parameter(Mandatory = $true)][string]$PackageRoot, $Manifest)

    if ($null -eq $Manifest) { $Manifest = Get-PackageManifest -PackageRoot $PackageRoot }
    $files = New-Object System.Collections.ArrayList
    $payload = Join-Path $PackageRoot 'payload'
    if (Test-Path $payload) {
        foreach ($item in Get-ChildItem -Path $payload -Recurse -File -Force) {
            $relative = Get-RelativePathUnderRoot -Root $payload -Path $item.FullName
            if ([string]::IsNullOrEmpty($relative)) { continue }
            if (Test-PayloadPath -RelativePath $relative -Manifest $Manifest) {
                [void]$files.Add('payload/' + $relative.Replace('\', '/'))
            }
        }
    }
    # The delivery scripts ride at the archive root and are copied into the
    # installation, so they are verified exactly like the payload.
    foreach ($item in Get-ChildItem -Path $PackageRoot -Recurse -File -Force) {
        $relative = Get-RelativePathUnderRoot -Root $PackageRoot -Path $item.FullName
        if ([string]::IsNullOrEmpty($relative)) { continue }
        if ($relative -like 'payload*') { continue }
        if (Test-ScriptPath -RelativePath $relative -Manifest $Manifest) {
            [void]$files.Add($relative.Replace('\', '/'))
        }
    }
    foreach ($name in @('package-manifest.json', [string]$Manifest.runtime.pip_requirements)) {
        if ([string]::IsNullOrWhiteSpace($name)) { continue }
        if (Test-Path -Path (Join-Path $PackageRoot $name) -PathType Leaf) { [void]$files.Add($name) }
    }
    # The offline bundle ships with the package, so it is covered by the same
    # integrity lock: the installer verifies the whole 2.5 GB before it starts.
    $offlineRoot = Join-Path $PackageRoot ([string]$Manifest.runtime.offline.directory)
    if (Test-Path -PathType Container $offlineRoot) {
        foreach ($item in Get-ChildItem -Path $offlineRoot -Recurse -File -Force) {
            $relative = Get-RelativePathUnderRoot -Root $PackageRoot -Path $item.FullName
            if ([string]::IsNullOrEmpty($relative)) { continue }
            if (Test-OfflineBundlePath -RelativePath $relative -Manifest $Manifest) {
                [void]$files.Add($relative.Replace('\', '/'))
            }
        }
    }
    return @($files | Sort-Object -Unique)
}

function Test-OfflineBundlePath {
    # What may ship inside ``offline\``: the declared directory, nothing hidden
    # and nothing from the denied segments, whatever the archive layout is.
    param([string]$RelativePath, $Manifest)

    $normalized = ([string]$RelativePath).Replace('\', '/').TrimStart('/')
    if ([string]::IsNullOrWhiteSpace($normalized)) { return $false }
    $prefix = ([string]$Manifest.runtime.offline.directory).TrimEnd('/') + '/'
    if (-not $normalized.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    foreach ($segment in $normalized.Split('/')) {
        if ($script:PayloadDeniedSegments -contains $segment.ToLower()) { return $false }
    }
    return $true
}

function New-PackageLock {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [string]$Version = '',
        $Manifest = $null
    )
    if ($null -eq $Manifest) { $Manifest = Get-PackageManifest -PackageRoot $PackageRoot }
    if ([string]::IsNullOrWhiteSpace($Version)) { $Version = [string]$Manifest.version }

    $entries = New-Object System.Collections.ArrayList
    foreach ($relative in Get-PackageFileList -PackageRoot $PackageRoot -Manifest $Manifest) {
        $path = Join-Path $PackageRoot $relative
        $item = Get-Item -Path $path
        [void]$entries.Add([ordered]@{
                path   = $relative
                size   = [long]$item.Length
                sha256 = Get-FileSha256 -Path $path
            })
    }
    return [ordered]@{
        schema_version = '1.0'
        product        = [string]$Manifest.product
        version        = $Version
        built_at       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        files          = @($entries)
    }
}

function Get-PackageLock {
    param([Parameter(Mandatory = $true)][string]$PackageRoot)
    $path = Join-Path $PackageRoot 'package-manifest.lock.json'
    if (-not (Test-Path -PathType Leaf $path)) {
        throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                -Message '安装包缺少完整性记录 package-manifest.lock.json，已拒绝安装。')
    }
    try { return (Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json) }
    catch {
        throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                -Message '安装包的完整性记录无法解析，已拒绝安装。')
    }
}

function Test-PackageIntegrity {
    param([Parameter(Mandatory = $true)][string]$PackageRoot, $Lock = $null)

    if ($null -eq $Lock) { $Lock = Get-PackageLock -PackageRoot $PackageRoot }
    $mismatches = New-Object System.Collections.ArrayList
    foreach ($entry in @($Lock.files)) {
        $path = Join-Path $PackageRoot ([string]$entry.path)
        if (-not (Test-Path -Path $path -PathType Leaf)) {
            [void]$mismatches.Add(@{ path = [string]$entry.path; reason = 'MISSING' })
        } elseif (-not (Test-FileHashMatches -Path $path -Sha256 ([string]$entry.sha256))) {
            [void]$mismatches.Add(@{ path = [string]$entry.path; reason = 'HASH_MISMATCH' })
        }
    }
    return @{ Ok = (@($mismatches).Count -eq 0); Mismatches = @($mismatches) }
}

function Assert-PackageIntegrity {
    param([Parameter(Mandatory = $true)][string]$PackageRoot, $Lock = $null)

    $result = Test-PackageIntegrity -PackageRoot $PackageRoot -Lock $Lock
    if (-not $result.Ok) {
        $names = (@($result.Mismatches) | ForEach-Object { $_.path }) -join ', '
        throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                -Message ("安装包校验失败，以下文件缺失或被修改：{0}。请重新获取安装包。" -f $names))
    }
    return $true
}

function Copy-PackagePayload {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [Parameter(Mandatory = $true)]$Lock,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$LogFile = $null,
        [string]$InstallRoot = ''
    )

    $copied = 0
    foreach ($entry in @($Lock.files)) {
        $relative = [string]$entry.path
        if (-not $relative.StartsWith('payload/')) { continue }
        $inner = $relative.Substring('payload/'.Length)
        $source = Join-Path $PackageRoot $relative
        $target = Join-Path $Destination $inner
        $directory = Split-Path -Path $target -Parent
        if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
        Copy-Item -Path $source -Destination $target -Force
        if (-not (Test-FileHashMatches -Path $target -Sha256 ([string]$entry.sha256))) {
            throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                    -Message ("复制后校验失败：{0}" -f $inner))
        }
        $copied = $copied + 1
    }
    Write-DeliveryLog -LogFile $LogFile -InstallRoot $InstallRoot `
        -Message ("已复制程序文件 {0} 个" -f $copied)
    return $copied
}

# ── the offline dependency bundle ───────────────────────────────────────────
# The installation never downloads anything. The Miniconda installer, the CUDA
# PyTorch wheels and every ordinary wheel travel inside the package under
# ``offline\`` and are described by a deterministic ``offline-lock.json``; the
# installer verifies the whole bundle against the pinned manifest and that lock
# before it starts a single process.

function Write-JsonFile {
    # One spelling of "write this object as JSON": UTF-8 without a BOM, no
    # trailing newline, key order as built — so the same input gives the same
    # bytes and a lock can be compared and re-verified.
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Object,
        [int]$Depth = 10
    )
    $directory = Split-Path -Path $Path -Parent
    if ($directory -and -not (Test-Path $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, ($Object | ConvertTo-Json -Depth $Depth),
        (New-Object System.Text.UTF8Encoding($false)))
    return $Path
}

function Get-NormalizedPackageName {
    # PEP 503 normalisation: ``SQLAlchemy`` and ``SQLAlchemy-2.0.52`` name the
    # same project, so the wheel a requirement needs can be found by name.
    param([Parameter(Mandatory = $true)][string]$Name)
    return ([string]$Name).Trim().ToLower().Replace('_', '-').Replace('.', '-')
}

function Get-OfflineLayout {
    param(
        [string]$PackageRoot = '',
        $Manifest = $null,
        [string]$OfflineRoot = ''
    )
    if ($null -eq $Manifest) {
        if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                    -Message '无法确定离线依赖目录：既没有安装包目录，也没有离线目录。')
        }
        $Manifest = Get-PackageManifest -PackageRoot $PackageRoot
    }
    if ([string]::IsNullOrWhiteSpace($OfflineRoot)) {
        $OfflineRoot = Join-Path $PackageRoot ([string]$Manifest.runtime.offline.directory)
    }
    $OfflineRoot = [System.IO.Path]::GetFullPath($OfflineRoot).TrimEnd('\', '/')
    return [pscustomobject]@{
        Root       = $OfflineRoot
        Miniconda  = Join-Path (Join-Path $OfflineRoot ([string]$Manifest.runtime.offline.miniconda_directory)) `
            ([string]$Manifest.runtime.conda_installer.file_name)
        MinicondaDirectory = Join-Path $OfflineRoot ([string]$Manifest.runtime.offline.miniconda_directory)
        Wheelhouse = Join-Path $OfflineRoot ([string]$Manifest.runtime.offline.wheelhouse_directory)
        Lock       = Join-Path $OfflineRoot ([string]$Manifest.runtime.offline.lock_file)
    }
}

function Get-OfflineCoreExpectations {
    # The three files the operator downloaded once: without them there is no
    # offline installation, so they are pinned with a size and a SHA-256.
    param([Parameter(Mandatory = $true)]$Manifest)

    $offline = $Manifest.runtime.offline
    $installer = $Manifest.runtime.conda_installer
    $items = New-Object System.Collections.ArrayList
    [void]$items.Add([pscustomobject]@{
            Name         = [string]$installer.file_name
            RelativePath = (Join-Path ([string]$offline.miniconda_directory) ([string]$installer.file_name))
            Sha256       = ([string]$installer.sha256).ToLower()
            Size         = [long]$installer.size
            Purpose      = 'private-python-runtime'
        })
    foreach ($wheel in @($Manifest.runtime.torch.wheels)) {
        [void]$items.Add([pscustomobject]@{
                Name         = [string]$wheel.file_name
                RelativePath = (Join-Path ([string]$offline.wheelhouse_directory) ([string]$wheel.file_name))
                Sha256       = ([string]$wheel.sha256).ToLower()
                Size         = [long]$wheel.size
                Purpose      = [string]$wheel.purpose
            })
    }
    return @($items)
}

function Test-OfflineWheelFileName {
    # A wheel that pip may install into the private runtime: Windows x64, the
    # CPython 3.10 ABI (or pure Python), and — for torch — the CUDA 12.1 build.
    # A CPU torch, a Linux wheel or an sdist would silently change what the
    # product runs, so each is refused with its own reason.
    param([Parameter(Mandatory = $true)][string]$FileName)

    $leaf = [System.IO.Path]::GetFileName($FileName)
    if (-not $leaf.ToLower().EndsWith('.whl')) {
        return @{ Ok = $false; Reason = '不是二进制 wheel（禁止 sdist 与其它压缩包）' }
    }
    $parts = @($leaf.Substring(0, $leaf.Length - 4).Split('-'))
    if (@($parts).Count -lt 5) {
        return @{ Ok = $false; Reason = 'wheel 文件名不完整' }
    }
    $platform = [string]$parts[-1]
    $abi = [string]$parts[-2]
    $pythonTag = [string]$parts[-3]
    $projectName = ([string]$parts[0]).ToLower()
    $version = [string]$parts[1]

    if (@('win_amd64', 'any') -notcontains $platform) {
        return @{ Ok = $false; Reason = '不是 Windows x64 平台的 wheel' }
    }
    if (@('cp310', 'abi3', 'none') -notcontains $abi) {
        return @{ Ok = $false; Reason = 'ABI 标记不是 cp310' }
    }
    $supported = @('cp310', 'py310', 'py3', 'cp3')
    $matchesPython = $false
    foreach ($tag in @($pythonTag.Split('.'))) {
        if ($supported -contains $tag) { $matchesPython = $true }
        # An abi3 tag names the *oldest* CPython that can load the wheel, so a
        # cp37/cp38/cp39 minimum still installs into this 3.10 runtime (opencv
        # and psutil really ship cp37-abi3 Windows wheels). cp311 and newer need
        # an interpreter this runtime is not, and stay refused. The rule applies
        # only to abi3: cp37-cp37 names one exact ABI and is never admitted.
        if ($abi -eq 'abi3' -and $tag -match '^cp3(\d+)$' -and [int]$Matches[1] -le 10) {
            $matchesPython = $true
        }
    }
    if (-not $matchesPython) {
        return @{ Ok = $false; Reason = 'Python 标记不是 CPython 3.10' }
    }
    if (($projectName -eq 'torch' -or $projectName -eq 'torchvision') -and
        (-not $version.ToLower().Contains('+cu121'))) {
        return @{ Ok = $false; Reason = 'torch/torchvision 不是 CUDA 12.1 构建（不安装 CPU 版）' }
    }
    return @{ Ok = $true; Reason = ''; Project = $projectName; Version = $version }
}

function Get-OfflineLock {
    param([Parameter(Mandatory = $true)][string]$OfflineRoot)

    $path = Join-Path $OfflineRoot 'offline-lock.json'
    if (-not (Test-Path -PathType Leaf $path)) {
        throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                -Message '离线依赖缺少 offline-lock.json，安装包不完整，已拒绝安装。')
    }
    try {
        $lock = Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_HASH_MISMATCH' `
                -Message '离线依赖清单 offline-lock.json 无法解析，已拒绝安装。')
    }
    if ([string]$lock.schema_version -ne '1.0') {
        throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_HASH_MISMATCH' `
                -Message '离线依赖清单 offline-lock.json 的版本不受支持，已拒绝安装。')
    }
    return $lock
}

function New-OfflineLock {
    # The bundle's own integrity record, built from what is on disk: the same
    # inputs give the same bytes, so two builds of the same package are
    # comparable and the installer can re-verify every file's size and hash.
    param(
        [Parameter(Mandatory = $true)][string]$OfflineRoot,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$RequirementsPath
    )
    $purposeByName = @{}
    foreach ($wheel in @($Manifest.runtime.torch.wheels)) {
        $purposeByName[([string]$wheel.file_name).ToLower()] = [string]$wheel.purpose
    }
    $purposeByName[([string]$Manifest.runtime.conda_installer.file_name).ToLower()] = 'private-python-runtime'

    $entries = New-Object System.Collections.ArrayList
    $directories = @(
        [pscustomobject]@{ Directory = [string]$Manifest.runtime.offline.miniconda_directory
            Purpose = 'private-python-runtime' },
        [pscustomobject]@{ Directory = [string]$Manifest.runtime.offline.wheelhouse_directory
            Purpose = 'runtime-dependency' }
    )
    foreach ($declared in $directories) {
        $directory = Join-Path $OfflineRoot $declared.Directory
        if (-not (Test-Path -PathType Container $directory)) { continue }
        foreach ($item in @(Get-ChildItem -Path $directory -File -Force | Sort-Object -Property Name)) {
            $purpose = [string]$declared.Purpose
            $key = $item.Name.ToLower()
            if ($purposeByName.ContainsKey($key)) { $purpose = $purposeByName[$key] }
            [void]$entries.Add([ordered]@{
                    path    = ($declared.Directory + '/' + $item.Name)
                    size    = [long]$item.Length
                    sha256  = Get-FileSha256 -Path $item.FullName
                    purpose = $purpose
                })
        }
    }
    return [ordered]@{
        schema_version      = '1.0'
        product             = [string]$Manifest.product
        python_version      = [string]$Manifest.python_version
        platform            = 'win_amd64'
        requirements_file   = [System.IO.Path]::GetFileName($RequirementsPath)
        requirements_sha256 = (Get-FileSha256OrEmpty -Path $RequirementsPath)
        files               = @($entries)
    }
}

function Get-MissingOfflineRequirements {
    # Which pinned requirements have no wheel in the given set. Used by the
    # build to name the package it could not get — never to loosen the pin.
    param(
        [Parameter(Mandatory = $true)][string]$RequirementsPath,
        [string[]]$WheelFileNames = @()
    )
    $present = @{}
    foreach ($name in @($WheelFileNames)) {
        $leaf = [System.IO.Path]::GetFileName([string]$name)
        if (-not $leaf.ToLower().EndsWith('.whl')) { continue }
        $parts = @($leaf.Substring(0, $leaf.Length - 4).Split('-'))
        if (@($parts).Count -lt 2) { continue }
        $present[(Get-NormalizedPackageName -Name $parts[0])] = $true
    }

    $missing = New-Object System.Collections.ArrayList
    if (-not (Test-Path -PathType Leaf $RequirementsPath)) { return @() }
    foreach ($line in @(Get-Content -Path $RequirementsPath -Encoding UTF8)) {
        $entry = ([string]$line).Trim()
        if ([string]::IsNullOrWhiteSpace($entry) -or $entry.StartsWith('#')) { continue }
        $match = [regex]::Match($entry, '^([A-Za-z0-9._-]+)==([^\s;]+)$')
        if (-not $match.Success) {
            [void]$missing.Add([ordered]@{ Name = $entry; Version = ''; Wheels = @(); Url = '' })
            continue
        }
        $name = Get-NormalizedPackageName -Name $match.Groups[1].Value
        if ($present.ContainsKey($name)) { continue }
        $version = [string]$match.Groups[2].Value
        [void]$missing.Add([ordered]@{
                Name    = $name
                Version = $version
                Wheels  = @(
                    ("{0}-{1}-cp310-cp310-win_amd64.whl" -f $name, $version),
                    ("{0}-{1}-py3-none-any.whl" -f $name, $version),
                    ("{0}-{1}-py2.py3-none-any.whl" -f $name.Replace('-', '_'), $version)
                )
                Url     = ("https://pypi.org/project/{0}/{1}/#files" -f $name, $version)
            })
    }
    return @($missing)
}

function Test-OfflineLockPath {
    # A lock entry must be a plain relative path that stays inside the offline
    # root: no drive, no leading separator, no empty/``.``/``..`` segment and no
    # character that cannot appear in a file name. Anything else could name a
    # file outside the bundle on the machine that verifies it.
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = ([string]$RelativePath).Replace('\', '/')
    if ([string]::IsNullOrWhiteSpace($normalized)) { return $false }
    if ($normalized.StartsWith('/')) { return $false }
    if ($normalized -match '^[A-Za-z]:') { return $false }
    foreach ($segment in $normalized.Split('/')) {
        if ([string]::IsNullOrWhiteSpace($segment)) { return $false }
        if ($segment -eq '.' -or $segment -eq '..') { return $false }
        if ($segment.IndexOfAny([char[]]@('<', '>', ':', '"', '|', '?', '*')) -ge 0) { return $false }
    }
    return $true
}

function Test-PipRequirementSpecifier {
    # ``name``, ``name==1.2.3``, ``name>=1.2``, ``name[extra]==1.2`` — a package
    # specifier and nothing else. A string that is not exactly this shape (a
    # path, a URL, a credential) is refused, so pip's own output can never be
    # echoed verbatim.
    param([string]$Spec)

    return [bool]([regex]::IsMatch(([string]$Spec).Trim(),
            '^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?([<>=!~]=[A-Za-z0-9._+!*-]+)?$'))
}

function Get-MissingPipRequirement {
    # Build machine only: which package pip could not get, including a
    # dependency the requirements file never pins directly. pip's full output is
    # never echoed — it can name local build directories and index credentials —
    # so only the specifier of pip's two stable "no distribution" lines survives,
    # and only when it really looks like a requirement.
    param([string]$Text, [int]$Limit = 5)

    $found = New-Object System.Collections.ArrayList
    $seen = @{}
    $patterns = @(
        'Could not find a version that satisfies the requirement\s+([^\s,;]+)',
        'No matching distribution found for\s+([^\s,;]+)'
    )
    foreach ($pattern in $patterns) {
        foreach ($match in [regex]::Matches([string]$Text, $pattern, 'IgnoreCase')) {
            $spec = ([string]$match.Groups[1].Value).Trim('(', ')', '[', ']', '.', ',', ';', '"', "'")
            if (-not (Test-PipRequirementSpecifier -Spec $spec)) { continue }
            if ($seen.ContainsKey($spec.ToLower())) { continue }
            $seen[$spec.ToLower()] = $true
            [void]$found.Add($spec)
            if (@($found).Count -ge $Limit) { return @($found) }
        }
    }
    return @($found)
}

function Test-OfflineBundle {
    # The whole offline promise in one answer: the three pinned files with their
    # exact bytes, a lock that belongs to this package and names nothing outside
    # the bundle, a tree that holds exactly the locked files at every depth, only
    # Windows CPython 3.10 binary wheels, and no missing pinned dependency.
    #
    # ``-PackageRoot`` covers the installation (a package's own offline folder);
    # the build machine passes the source folder and the offline directory it is
    # about to ship separately.
    param(
        [string]$PackageRoot = '',
        $Manifest = $null,
        [string]$OfflineRoot = '',
        [string]$RequirementsPath = ''
    )
    if ($null -eq $Manifest) {
        if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
            throw (New-DeliveryFailure -Code 'PACKAGE_MANIFEST_INVALID' `
                    -Message '无法确定安装包清单：缺少 package-manifest.json 所在目录。')
        }
        $Manifest = Get-PackageManifest -PackageRoot $PackageRoot
    }
    if ([string]::IsNullOrWhiteSpace($OfflineRoot)) {
        $OfflineRoot = (Get-OfflineLayout -PackageRoot $PackageRoot -Manifest $Manifest).Root
    }
    if ([string]::IsNullOrWhiteSpace($RequirementsPath) -and
        -not [string]::IsNullOrWhiteSpace($PackageRoot)) {
        $RequirementsPath = Join-Path $PackageRoot ([string]$Manifest.runtime.pip_requirements)
    }
    $offline = Get-OfflineLayout -Manifest $Manifest -OfflineRoot $OfflineRoot

    if (-not (Test-Path -PathType Container $offline.Root)) {
        return @{ Ok = $false; Code = 'OFFLINE_BUNDLE_MISSING'; Offenders = @()
            Message = ("安装包内缺少离线依赖目录 {0}，安装包不完整，已拒绝安装。" -f `
                    [string]$Manifest.runtime.offline.directory) }
    }

    $verified = @{}
    foreach ($expectation in @(Get-OfflineCoreExpectations -Manifest $Manifest)) {
        $relative = ([string]$expectation.RelativePath).Replace('\', '/')
        $path = Join-Path $offline.Root $relative
        if (-not (Test-Path -PathType Leaf $path)) {
            return @{ Ok = $false; Code = 'OFFLINE_BUNDLE_MISSING'; Offenders = @($relative)
                Message = ("离线依赖缺少 {0}，安装包不完整，已拒绝安装。" -f $expectation.Name) }
        }
        if (-not (Test-FileHashMatches -Path $path -Sha256 ([string]$expectation.Sha256)) -or
            ([long](Get-Item -LiteralPath $path).Length -ne [long]$expectation.Size)) {
            return @{ Ok = $false; Code = 'OFFLINE_BUNDLE_HASH_MISMATCH'; Offenders = @($relative)
                Message = ("离线文件 {0} 与 package-manifest.json 记录的大小或 SHA-256 不一致，已拒绝安装。" -f `
                        $expectation.Name) }
        }
        $verified[$relative] = $true
    }

    try { $lock = Get-OfflineLock -OfflineRoot $offline.Root }
    catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        return @{ Ok = $false; Code = $info.Code; Message = $info.Message; Offenders = @() }
    }
    if (-not [string]::IsNullOrWhiteSpace($RequirementsPath) -and
        [string]$lock.requirements_sha256 -ne (Get-FileSha256OrEmpty -Path $RequirementsPath)) {
        return @{ Ok = $false; Code = 'OFFLINE_BUNDLE_HASH_MISMATCH'
            Offenders = @([string]$Manifest.runtime.offline.lock_file)
            Message = '离线依赖清单与安装包的依赖清单不一致，安装包不完整，已拒绝安装。' }
    }

    # The lock's own file list has to be a well-formed set before anything is
    # compared against it: a path that leaves the bundle, or the same path
    # recorded twice, is a broken lock rather than a missing wheel.
    $lockOffenders = New-Object System.Collections.ArrayList
    $locked = @{}
    foreach ($entry in @($lock.files)) {
        $relative = ([string]$entry.path).Replace('\', '/')
        if (-not (Test-OfflineLockPath -RelativePath $relative)) {
            [void]$lockOffenders.Add('offline-lock.json（记录了越界的文件路径）')
            continue
        }
        if ($locked.ContainsKey($relative)) {
            [void]$lockOffenders.Add(("{0}（离线清单重复登记）" -f $relative))
            continue
        }
        $locked[$relative] = $entry
    }
    if (@($lockOffenders).Count -gt 0) {
        return @{ Ok = $false; Code = 'OFFLINE_BUNDLE_HASH_MISMATCH'; Offenders = @($lockOffenders)
            Message = ("离线依赖清单 offline-lock.json 不能作为依据，已拒绝安装：{0}。" -f `
                    (@($lockOffenders) -join '、')) }
    }

    # Every recorded file, present with exactly the size and bytes promised. The
    # three core files were already hashed above.
    $offenders = New-Object System.Collections.ArrayList
    foreach ($relative in @($locked.Keys)) {
        $entry = $locked[$relative]
        if ($verified.ContainsKey($relative)) { continue }
        $path = Join-Path $offline.Root $relative
        if (-not (Test-Path -PathType Leaf $path)) {
            [void]$offenders.Add(("{0}（缺失）" -f $relative))
            continue
        }
        if ([long](Get-Item -LiteralPath $path).Length -ne [long]$entry.size) {
            [void]$offenders.Add(("{0}（大小与离线清单不一致）" -f $relative))
            continue
        }
        if (-not (Test-FileHashMatches -Path $path -Sha256 ([string]$entry.sha256))) {
            [void]$offenders.Add(("{0}（内容与离线清单不一致）" -f $relative))
        }
    }
    if (@($offenders).Count -gt 0) {
        return @{ Ok = $false; Code = 'OFFLINE_WHEELHOUSE_INCOMPLETE'; Offenders = @($offenders)
            Message = ("离线依赖与离线清单不一致，已拒绝安装：{0}。" -f (@($offenders) -join '、')) }
    }

    # Only the promised wheels, and only wheels this runtime can install.
    $wheelOffenders = New-Object System.Collections.ArrayList
    $wheelNames = New-Object System.Collections.ArrayList
    $wheelhouseDirectory = [string]$Manifest.runtime.offline.wheelhouse_directory
    if (Test-Path -PathType Container $offline.Wheelhouse) {
        foreach ($item in @(Get-ChildItem -Path $offline.Wheelhouse -File -Force | Sort-Object -Property Name)) {
            [void]$wheelNames.Add($item.Name)
            $relative = ($wheelhouseDirectory + '/' + $item.Name)
            $checked = Test-OfflineWheelFileName -FileName $item.Name
            if (-not $checked.Ok) {
                [void]$wheelOffenders.Add(("{0}（{1}）" -f $relative, $checked.Reason))
            } elseif (-not $locked.ContainsKey($relative)) {
                [void]$wheelOffenders.Add(("{0}（不在离线清单内）" -f $relative))
            }
        }
    }
    if (@($wheelOffenders).Count -gt 0) {
        return @{ Ok = $false; Code = 'OFFLINE_WHEELHOUSE_INCOMPLETE'; Offenders = @($wheelOffenders)
            Message = ("离线依赖 wheelhouse 不符合受控要求，已拒绝安装：{0}。" -f (@($wheelOffenders) -join '、')) }
    }

    # The exact file set: the whole bundle is walked, at every depth. Only the
    # lock itself may travel unnamed — an extra installer, a stray note, a wheel
    # left in a sub-directory or a second copy beside Miniconda would all change
    # what the machine installs without the lock saying so.
    $unlisted = New-Object System.Collections.ArrayList
    $lockLeaf = [System.IO.Path]::GetFileName($offline.Lock)
    foreach ($item in @(Get-ChildItem -Path $offline.Root -Recurse -File -Force)) {
        $relative = Get-RelativePathUnderRoot -Root $offline.Root -Path $item.FullName
        if ([string]::IsNullOrEmpty($relative)) {
            [void]$unlisted.Add('offline（有文件不在离线目录内）')
            continue
        }
        $relative = $relative.Replace('\', '/')
        if ($relative.Equals($lockLeaf, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
        if (-not $locked.ContainsKey($relative)) {
            [void]$unlisted.Add(("{0}（不在离线清单内）" -f $relative))
        }
    }
    if (@($unlisted).Count -gt 0) {
        return @{ Ok = $false; Code = 'OFFLINE_WHEELHOUSE_INCOMPLETE'; Offenders = @($unlisted)
            Message = ("离线依赖目录包含离线清单未登记的文件，安装包不完整，已拒绝安装：{0}。" -f `
                    (@($unlisted) -join '、')) }
    }

    $missing = @()
    if (-not [string]::IsNullOrWhiteSpace($RequirementsPath)) {
        $missing = @(Get-MissingOfflineRequirements -RequirementsPath $RequirementsPath `
                -WheelFileNames @($wheelNames))
    }
    if (@($missing).Count -gt 0) {
        $names = @($missing | ForEach-Object { "{0}=={1}" -f $_.Name, $_.Version })
        return @{ Ok = $false; Code = 'OFFLINE_WHEELHOUSE_INCOMPLETE'; Offenders = $names
            Message = ("离线依赖 wheelhouse 缺少锁定依赖：{0}，安装包不完整，已拒绝安装。" -f ($names -join '、')) }
    }
    return @{ Ok = $true; Code = $null; Message = ''; Offenders = @() }
}

function Assert-OfflineBundle {
    param(
        [string]$PackageRoot = '',
        $Manifest = $null,
        [string]$OfflineRoot = '',
        [string]$RequirementsPath = ''
    )
    $result = Test-OfflineBundle -PackageRoot $PackageRoot -Manifest $Manifest `
        -OfflineRoot $OfflineRoot -RequirementsPath $RequirementsPath
    if (-not $result.Ok) {
        throw (New-DeliveryFailure -Code $result.Code -Message $result.Message `
                -Detail @{ Offenders = @($result.Offenders) })
    }
    return $true
}

function Prepare-OfflineBundle {
    # Build-machine only: verify the three locally downloaded files against the
    # manifest, place them in the bundle, let pip fetch the ordinary wheels for
    # Windows/CPython 3.10 (binary only) and write the bundle lock. A failure
    # never removes a file that already verified.
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$DependencySource,
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [string]$Python = '',
        $Runner = $null,
        [string]$LogFile = $null
    )
    $repoRoot = Get-CanonicalPath -Path $RepoRoot
    $windowsRoot = Join-Path $repoRoot 'windows'
    if (-not (Test-Path -PathType Leaf (Join-Path $windowsRoot 'package-manifest.json'))) {
        throw (New-DeliveryFailure -Code 'PACKAGE_ROOT_INVALID' `
                -Message ("{0} 下没有 windows\package-manifest.json，不是有效的交付源码目录。" -f $repoRoot))
    }
    $manifest = Get-PackageManifest -PackageRoot $windowsRoot
    $requirementsPath = Join-Path $windowsRoot ([string]$manifest.runtime.pip_requirements)
    $source = Get-CanonicalPath -Path $DependencySource
    $output = [System.IO.Path]::GetFullPath($OutputDir).TrimEnd('\', '/')
    $offline = Get-OfflineLayout -Manifest $manifest -OfflineRoot $output

    foreach ($directory in @($offline.Root, $offline.MinicondaDirectory, $offline.Wheelhouse)) {
        if (-not (Test-Path $directory)) {
            try { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
            catch {
                throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                        -Message '无法创建离线依赖目录，请检查目标位置是否存在同名文件或权限限制。')
            }
        }
    }

    # 1. the three files the operator downloaded once
    $preserved = New-Object System.Collections.ArrayList
    foreach ($expectation in @(Get-OfflineCoreExpectations -Manifest $manifest)) {
        $candidate = Join-Path $source ([string]$expectation.Name)
        if (-not (Test-Path -PathType Leaf $candidate)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                    -Message ("离线源目录中缺少 {0}，请确认文件名与大小与 package-manifest.json 一致。" -f $expectation.Name) `
                    -Detail @{ Preserved = @($preserved); Hint = '已保留已校验的文件，补齐后重试即可继续。' })
        }
        if (-not (Test-FileHashMatches -Path $candidate -Sha256 ([string]$expectation.Sha256)) -or
            ([long](Get-Item -LiteralPath $candidate).Length -ne [long]$expectation.Size)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_HASH_MISMATCH' `
                    -Message ("{0} 的大小或 SHA-256 与 package-manifest.json 不一致，已拒绝使用。" -f $expectation.Name) `
                    -Detail @{ Preserved = @($preserved); Hint = '已保留已校验的文件，重新下载该文件后重试即可继续。' })
        }
        $target = Join-Path $offline.Root ([string]$expectation.RelativePath)
        if (-not (Test-FileHashMatches -Path $target -Sha256 ([string]$expectation.Sha256))) {
            Copy-Item -Path $candidate -Destination $target -Force
        }
        [void]$preserved.Add(([string]$expectation.RelativePath))
    }

    # 2. the ordinary wheels, binary only, for this platform and ABI
    $arguments = @('-m', 'pip', 'download', '--only-binary=:all:',
        '--platform', 'win_amd64', '--python-version', '3.10', '--implementation', 'cp',
        '--abi', 'cp310', '--no-cache-dir', '--dest', $offline.Wheelhouse,
        '--find-links', $offline.Wheelhouse, '--requirement', $requirementsPath)
    foreach ($wheel in @($manifest.runtime.torch.wheels)) {
        $arguments += ("{0}=={1}" -f (Get-NormalizedPackageName -Name ([string]$wheel.file_name.Split('-')[0])), `
                [string]$wheel.version)
    }
    if ([string]::IsNullOrWhiteSpace($Python)) {
        throw (New-DeliveryFailure -Code 'OFFLINE_WHEELHOUSE_INCOMPLETE' `
                -Message '需要指定用于下载依赖的 Python 3.10 解释器（-Python）。' `
                -Detail @{ Preserved = @($preserved); Missing = @(); Hint = '已保留已校验的文件。' })
    }
    if ($null -eq $Runner) { $Runner = ${function:Invoke-ExternalProcess} }
    $exitCode = 0
    $stdErr = ''
    try {
        $result = & $Runner -FilePath $Python -Arguments $arguments `
            -WorkingDirectory $repoRoot -Environment $null -LogFile $LogFile
        if ($null -ne $result -and $null -ne $result.ExitCode) { $exitCode = [int]$result.ExitCode }
        if ($null -ne $result) { $stdErr = [string]$result.StdErr }
    } catch {
        $exitCode = 1
        $stdErr = [string]$_.Exception.Message
    }
    if ($exitCode -ne 0) { Write-DeliveryLog -LogFile $LogFile -Level 'WARN' -Message $stdErr }

    # 3. name every dependency that still has no wheel, never a version change
    $wheelNames = @()
    if (Test-Path -PathType Container $offline.Wheelhouse) {
        $wheelNames = @(Get-ChildItem -Path $offline.Wheelhouse -File -Force |
            ForEach-Object { $_.Name })
    }
    $missing = @(Get-MissingOfflineRequirements -RequirementsPath $requirementsPath `
            -WheelFileNames $wheelNames)
    # pip can also fail on a package the requirements file never pins — a
    # dependency of a dependency. Reading pip's own lines is the only way to name
    # it, so the operator is not handed an exit code and nothing else.
    $fromPip = @(Get-MissingPipRequirement -Text $stdErr)
    if ($exitCode -ne 0 -or @($missing).Count -gt 0) {
        $listed = @($missing | ForEach-Object { "{0}=={1}" -f $_.Name, $_.Version })
        $description = '依赖下载未完成（pip 退出码 {0}）' -f $exitCode
        $hint = '已保留已校验的文件；请按上面的包名、版本与 wheel 文件名获取官方二进制 wheel 后重试。'
        if (@($listed).Count -gt 0) {
            $description = '以下锁定依赖没有可用的 Windows/CPython 3.10 二进制 wheel：' + ($listed -join '、')
        } elseif (@($fromPip).Count -gt 0) {
            $description = 'pip 未能获取以下依赖（含传递依赖，没有可用的 Windows/CPython 3.10 wheel）：' + ($fromPip -join '、')
            $hint = '已保留已校验的文件；请按上面的包名与版本从官方源获取对应二进制 wheel 后重试。'
        }
        throw (New-DeliveryFailure -Code 'OFFLINE_WHEELHOUSE_INCOMPLETE' `
                -Message ("离线依赖准备未完成：{0}。已校验的文件已保留，补齐后重试即可继续；不要修改锁定版本。" -f $description) `
                -Detail @{
                    Missing        = @($missing)
                    MissingFromPip = @($fromPip)
                    PipExitCode    = $exitCode
                    PreservedFiles = @($preserved)
                    Hint           = $hint
                })
    }

    # 4. the bundle lock
    $lock = New-OfflineLock -OfflineRoot $offline.Root -Manifest $manifest `
        -RequirementsPath $requirementsPath
    Write-JsonFile -Path $offline.Lock -Object $lock | Out-Null
    Write-DeliveryLog -LogFile $LogFile `
        -Message ("离线依赖准备完成：{0} 个文件" -f @($lock.files).Count)
    return @{ Ok = $true; Lock = $lock; OfflineRoot = $offline.Root
        Files = @($lock.files | ForEach-Object { [string]$_.path }) }
}

# ── where the product may be installed ──────────────────────────────────────

function Assert-InstallRootAllowed {
    # The operator chooses the destination, so the dangerous answers are refused
    # with one stable code: no relative path, no network path, no drive root, no
    # Windows/system directory and no user-profile root.
    param([Parameter(Mandatory = $true)][string]$Root)

    if ([string]::IsNullOrWhiteSpace($Root)) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' -Message '安装目录不能为空。')
    }
    if ($Root.StartsWith('\\')) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                -Message '安装目录必须是本地磁盘上的绝对路径，不支持网络路径（UNC）。')
    }
    if (-not [System.IO.Path]::IsPathRooted($Root)) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                -Message '请提供完整的绝对路径，例如 D:\AutoTuneStudio。')
    }

    $full = Get-CanonicalPath -Path $Root
    $trimmed = $full.TrimEnd('\', '/')
    $driveRoot = ([System.IO.Path]::GetPathRoot($full)).TrimEnd('\', '/')
    if ([string]::IsNullOrEmpty($driveRoot)) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                -Message '安装目录必须是本地磁盘上的绝对路径。')
    }
    if ($trimmed -eq $driveRoot) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' -Message '不能把产品安装在盘符根目录下。')
    }
    if (Test-Path -PathType Leaf $full) {
        throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                -Message '指定的安装目录已经存在且是一个文件，请换一个目录。')
    }

    $exact = @()
    foreach ($candidate in @($env:SystemRoot, (Join-Path $env:SystemRoot 'System32'), $env:ProgramFiles,
            ${env:ProgramFiles(x86)}, $env:ProgramData, $env:USERPROFILE, $env:LOCALAPPDATA,
            $env:APPDATA, $env:PUBLIC)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            $exact += [System.IO.Path]::GetFullPath($candidate).TrimEnd('\', '/')
        }
    }
    foreach ($candidate in $exact) {
        if ($trimmed.Equals($candidate, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                    -Message '不能把产品安装到系统目录或用户配置目录本身，请选择一个专门的安装目录。')
        }
    }
    foreach ($candidate in @($env:SystemRoot, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:ProgramData)) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $prefix = [System.IO.Path]::GetFullPath($candidate).TrimEnd('\', '/') + '\'
        if ($trimmed.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw (New-DeliveryFailure -Code 'INSTALL_ROOT_INVALID' `
                    -Message '不能把产品安装到 Windows、系统程序目录或 ProgramData 之内，请选择其它磁盘目录。')
        }
    }
    return $full
}

function Get-SystemDriveRoot {
    $root = ''
    try { $root = [System.IO.Path]::GetPathRoot([System.Environment]::GetFolderPath('Windows')) } catch { }
    if ([string]::IsNullOrWhiteSpace($root)) { $root = [System.IO.Path]::GetPathRoot($env:SystemRoot) }
    if ([string]::IsNullOrWhiteSpace($root)) { $root = 'C:\' }
    return $root.TrimEnd('\', '/')
}

function Get-RecommendedInstallRoot {
    # A fresh installation is large: the preferred destination is a fixed local
    # disk other than the system one with the most room. The system drive is only
    # proposed when the machine has nothing else, and the caller warns about it.
    param($Probe = $null)

    $systemDrive = Get-SystemDriveRoot
    $best = ''
    $bestFree = -1
    foreach ($drive in @([System.IO.DriveInfo]::GetDrives())) {
        if (-not $drive.IsReady) { continue }
        if ([string]$drive.DriveType -ne 'Fixed') { continue }
        $name = ([string]$drive.Name).TrimEnd('\', '/')
        if ($name.Equals($systemDrive, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
        $free = [long](Invoke-DeliveryProbe -Probe $Probe -Name 'FreeBytes' `
                -Fallback { param($Path) Get-FreeDiskBytes -Path $Path } -Arguments @($drive.Name))
        if ($free -gt $bestFree) { $bestFree = $free; $best = $name }
    }
    if (-not [string]::IsNullOrWhiteSpace($best)) { return (Join-Path ($best + '\') 'AutoTuneStudio') }
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) { return (Join-Path ($systemDrive + '\') 'AutoTuneStudio') }
    return (Join-Path $env:LOCALAPPDATA 'AutoTuneStudio')
}

function Get-InstallPromptPlan {
    # Whether a *first* installation may ask the operator where to install, and
    # which directory it takes when it must not wait. A double-clicked
    # install.bat owns a console and may ask; every automated run — an explicit
    # directory, -AcceptRecommended, AUTO_TUNE_NONINTERACTIVE or redirected input
    # — proceeds without a prompt and falls back to the recommended disk.
    #
    # ``install.bat`` must therefore launch PowerShell in a mode that still
    # permits Read-Host: the flag that forbids it is checked by the test suite,
    # and this decision is what keeps automation from ever reaching the prompt.
    param(
        [string]$Requested = '',
        [string]$LocalAppData = $env:LOCALAPPDATA,
        [bool]$AcceptRecommended = $false,
        [string]$NonInteractiveVariable = '',
        [bool]$InputRedirected = $false
    )
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        return @{ Ask = $false; Reason = 'requested'; Destination = $Requested }
    }
    $installed = Resolve-DeliveryInstallRoot -LocalAppData $LocalAppData
    if (-not [string]::IsNullOrWhiteSpace($installed)) {
        return @{ Ask = $false; Reason = 'recorded'; Destination = $installed }
    }
    if ($AcceptRecommended) {
        return @{ Ask = $false; Reason = 'accept-recommended'; Destination = '' }
    }
    if (-not [string]::IsNullOrWhiteSpace($NonInteractiveVariable)) {
        return @{ Ask = $false; Reason = 'noninteractive'; Destination = '' }
    }
    if ($InputRedirected) {
        return @{ Ask = $false; Reason = 'input-redirected'; Destination = '' }
    }
    return @{ Ask = $true; Reason = 'console'; Destination = '' }
}

function Test-InstallRootOnSystemDrive {
    param([Parameter(Mandatory = $true)][string]$Root)
    $full = Get-CanonicalPath -Path $Root
    $drive = ([System.IO.Path]::GetPathRoot($full)).TrimEnd('\', '/')
    return $drive.Equals((Get-SystemDriveRoot), [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-InstallDestinationAdvice {
    # Choosing the system drive is allowed when the operator really asks for it,
    # but never silently: the advice says where the directory is and how much
    # room the installation needs.
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        $Probe = $null,
        [long]$RequiredFreeBytes = 0
    )
    $onSystemDrive = Test-InstallRootOnSystemDrive -Root $Root
    $message = ''
    if ($onSystemDrive) {
        $free = [long](Invoke-DeliveryProbe -Probe $Probe -Name 'FreeBytes' `
                -Fallback { param($Path) Get-FreeDiskBytes -Path $Path } -Arguments @($Root))
        $line = '所选安装目录位于系统盘（{0}，当前可用 {1:N1} GB）；本产品需要约 {2:N1} GB。' +
        '建议改选其它本地固定磁盘，例如 D:\AutoTuneStudio。'
        $message = ($line -f $Root, ($free / 1GB), ($RequiredFreeBytes / 1GB))
    }
    return @{ Root = $Root; OnSystemDrive = $onSystemDrive; Message = $message }
}

function Get-InstallLocationPath {
    # Where the installation records the directory it chose, so start, upgrade
    # and uninstall find the same one instead of asking again. It is the only
    # file this delivery keeps outside its own folder besides the shortcut.
    param([string]$LocalAppData = $env:LOCALAPPDATA)
    if ([string]::IsNullOrWhiteSpace($LocalAppData)) { return '' }
    $base = [System.IO.Path]::GetFullPath($LocalAppData)
    return (Join-Path (Join-Path $base 'AutoTuneStudio') 'install-location.json')
}

function Write-InstallLocation {
    param([Parameter(Mandatory = $true)]$Layout)
    if ([string]::IsNullOrWhiteSpace([string]$Layout.LocationFile)) { return $false }
    $record = [ordered]@{
        schema_version = '1.0'
        product        = 'auto-tune-studio'
        root           = $Layout.Root
        updated_at     = (Get-Date).ToUniversalTime().ToString('o')
    }
    Write-InstallState -StateFile $Layout.LocationFile -State $record | Out-Null
    return $true
}

function Remove-InstallLocation {
    # Only ever the record that names *this* installation.
    param([Parameter(Mandatory = $true)]$Layout)
    $path = [string]$Layout.LocationFile
    if ([string]::IsNullOrWhiteSpace($path)) { return $false }
    if (-not (Test-Path -PathType Leaf $path)) { return $false }
    $record = Read-InstallState -StateFile $path
    if ($null -eq $record) { return $false }
    $recorded = ([string]$record.root).TrimEnd('\', '/')
    if (-not $recorded.Equals(([string]$Layout.Root).TrimEnd('\', '/'),
            [System.StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    Remove-Item -Path $path -Force -ErrorAction SilentlyContinue
    return $true
}

function Get-InstalledRootRecord {
    param([string]$LocalAppData = $env:LOCALAPPDATA)
    $path = Get-InstallLocationPath -LocalAppData $LocalAppData
    if ([string]::IsNullOrWhiteSpace($path)) { return '' }
    $record = Read-InstallState -StateFile $path
    if ($null -eq $record) { return '' }
    return [string]$record.root
}

function Resolve-DeliveryInstallRoot {
    # One answer for "which installation is this": an explicit choice wins, then
    # the directory the running entry point lives in (an installed launcher is
    # its own installation's neighbour), then the record it left behind. Returns
    # '' when nothing is installed, so the caller can say so honestly.
    param(
        [string]$InstallRoot = '',
        [string]$EntryRoot = '',
        [string]$LocalAppData = $env:LOCALAPPDATA
    )
    if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
        return (Assert-InstallRootAllowed -Root $InstallRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($EntryRoot) -and (Test-Path -PathType Container $EntryRoot)) {
        $candidate = Get-CanonicalPath -Path $EntryRoot
        if (Test-Path -PathType Leaf (Join-Path $candidate 'install-state.json')) {
            return $candidate
        }
    }
    $recorded = Get-InstalledRootRecord -LocalAppData $LocalAppData
    if ([string]::IsNullOrWhiteSpace($recorded)) { return '' }
    try { return (Assert-InstallRootAllowed -Root $recorded) } catch { return '' }
}

# ── configuration ───────────────────────────────────────────────────────────

function Get-ConfigTemplatePath {
    param([Parameter(Mandatory = $true)]$Layout)
    return Join-Path (Join-Path $Layout.App 'auto_tune') 'config.template.yaml'
}

function Initialize-UserConfig {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [string]$TemplatePath = '',
        [string]$LogFile = $null
    )
    $configPath = $Layout.ConfigPath
    if (Test-Path -Path $configPath -PathType Leaf) {
        return $false
    }
    if ([string]::IsNullOrWhiteSpace($TemplatePath)) { $TemplatePath = Get-ConfigTemplatePath -Layout $Layout }
    if (-not (Test-Path -Path $TemplatePath -PathType Leaf)) {
        throw (New-DeliveryFailure -Code 'CONFIG_TEMPLATE_MISSING' `
                -Message ("脱敏配置模板缺失：{0}，安装包不完整。" -f $TemplatePath))
    }
    $directory = Split-Path -Path $configPath -Parent
    if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    Copy-Item -Path $TemplatePath -Destination $configPath -Force
    Write-DeliveryLog -LogFile $LogFile -InstallRoot $Layout.Root `
        -Message ("已从脱敏模板初始化配置：{0}" -f $configPath)
    return $true
}

# ── permanent entry points and the desktop shortcut ─────────────────────────
# Once the installation finishes, the operator must be able to start and remove
# the product from the installation itself: keeping the extracted package
# around is not part of the deal. The launcher therefore lives in a controlled
# subdirectory of the installation, never in the folder the ZIP was unpacked
# into.

function New-DesktopShortcut {
    param(
        [string]$Path,
        [string]$Target,
        [string]$WorkingDirectory,
        $Linker = $null
    )
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if ($null -ne $Linker) {
        return [bool](& $Linker -Path $Path -Target $Target -WorkingDirectory $WorkingDirectory)
    }
    $parent = Split-Path -Path $Path -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    # A plain .lnk written by the shell: no administrator rights, no registry.
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $Target
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = 'Auto Tune Studio'
    $shortcut.Save()
    return $true
}

function Get-DesktopShortcutTarget {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return '' }
    try {
        $shell = New-Object -ComObject WScript.Shell
        return [string]$shell.CreateShortcut($Path).TargetPath
    } catch {
        return ''
    }
}

function Remove-DesktopShortcut {
    param([string]$Path, [string]$Target)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if (-not (Test-Path -Path $Path -PathType Leaf)) { return $false }
    # The desktop belongs to the operator. Only the shortcut this installation
    # owns is removed, and only when its target can be read back and matches.
    $existing = Get-DesktopShortcutTarget -Path $Path
    if ([string]::IsNullOrWhiteSpace($existing)) { return $false }
    if (-not [string]::IsNullOrWhiteSpace($Target)) {
        $left = $existing.TrimEnd('\', '/')
        $right = $Target.TrimEnd('\', '/')
        if (-not $left.Equals($right, [System.StringComparison]::OrdinalIgnoreCase)) { return $false }
    }
    Remove-Item -Path $Path -Force
    return $true
}

function Install-DeliveryLauncher {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        $Lock = $null,
        [string]$LogFile = $null,
        $Linker = $null,
        [switch]$SkipShortcut
    )
    $expected = @{}
    if ($null -ne $Lock) {
        foreach ($entry in @($Lock.files)) {
            $expected[[string]$entry.path] = ([string]$entry.sha256).ToLower()
        }
    }

    # Two phases, so that a failure leaves the entry points exactly as they
    # were: every file is first written beside its target and verified, and only
    # then are they all swapped in, each keeping the copy it replaced until the
    # whole set is in place. An installation therefore never keeps half of an
    # old launcher with half of a new one.
    $steps = New-Object System.Collections.ArrayList
    try {
        foreach ($relative in Get-LauncherRelativePaths) {
            $key = $relative.Replace('\', '/')
            $source = Join-Path $PackageRoot $relative
            if (-not (Test-Path -Path $source -PathType Leaf)) {
                throw (New-DeliveryFailure -Code 'LAUNCHER_MISSING' `
                        -Message ("安装包缺少启动入口 {0}，安装包不完整。" -f $key))
            }
            if ($expected.ContainsKey($key) -and
                -not (Test-FileHashMatches -Path $source -Sha256 $expected[$key])) {
                throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                        -Message ("启动入口 {0} 与安装包记录不一致，已拒绝安装。" -f $key))
            }

            $target = Join-Path $Layout.Root $relative
            $directory = Split-Path -Path $target -Parent
            if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
            $staged = $target + '.new'
            Copy-Item -Path $source -Destination $staged -Force
            if ($expected.ContainsKey($key) -and
                -not (Test-FileHashMatches -Path $staged -Sha256 $expected[$key])) {
                Remove-Item -Path $staged -Force -ErrorAction SilentlyContinue
                throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' `
                        -Message ("启动入口 {0} 复制后校验失败。" -f $key))
            }
            [void]$steps.Add(@{ Key = $key; Target = $target; Staged = $staged
                    Backup = ($target + '.bak'); HadBackup = $false; Swapped = $false })
        }

        foreach ($step in $steps) {
            if (Test-Path -Path $step.Target -PathType Leaf) {
                Copy-Item -Path $step.Target -Destination $step.Backup -Force
                $step['HadBackup'] = $true
            }
            # Written beside the live file and then moved over it: cmd.exe must
            # never be able to pick up a half-copied .bat.
            Move-Item -Path $step.Staged -Destination $step.Target -Force
            $step['Swapped'] = $true
        }
    } catch {
        $failure = $_
        foreach ($step in $steps) {
            try {
                if ($step.Swapped) {
                    if ($step.HadBackup) {
                        Copy-Item -Path $step.Backup -Destination $step.Target -Force
                    } else {
                        Remove-Item -Path $step.Target -Force -ErrorAction SilentlyContinue
                    }
                }
                Remove-Item -Path $step.Staged -Force -ErrorAction SilentlyContinue
                Remove-Item -Path $step.Backup -Force -ErrorAction SilentlyContinue
            } catch { }
        }
        $info = Get-DeliveryErrorInfo -ErrorRecord $failure
        if ($info.Code -eq 'HARNESS_ERROR') {
            # An unexpected failure (a file the operator has open, a read-only
            # directory) is reported as its own condition. The entry points are
            # as they were before the attempt.
            throw (New-DeliveryFailure -Code 'LAUNCHER_INSTALL_FAILED' `
                    -Message ("部署启动入口失败：{0}" -f $info.Message))
        }
        throw
    }
    foreach ($step in $steps) {
        Remove-Item -Path $step.Backup -Force -ErrorAction SilentlyContinue
    }

    $shortcutCreated = $false
    if (-not $SkipShortcut -and -not [string]::IsNullOrWhiteSpace($Layout.Shortcut)) {
        try {
            $shortcutCreated = New-DesktopShortcut -Path $Layout.Shortcut -Target $Layout.StartBat `
                -WorkingDirectory $Layout.Root -Linker $Linker
        } catch {
            $shortcutCreated = $false
            Write-DeliveryLog -LogFile $LogFile -Level 'WARN' -InstallRoot $Layout.Root `
                -Message '无法创建桌面快捷方式，可以直接运行安装目录下的 start.bat。'
        }
    }

    Write-DeliveryLog -LogFile $LogFile -InstallRoot $Layout.Root `
        -Message ("已部署启动与卸载入口：{0}" -f ((Get-LauncherRelativePaths) -join ', '))
    return @{ Files = @(Get-LauncherRelativePaths); ShortcutCreated = $shortcutCreated
        Shortcut = $Layout.Shortcut }
}

function Test-LauncherInstalled {
    param([Parameter(Mandatory = $true)]$Layout)
    foreach ($relative in Get-LauncherRelativePaths) {
        if (-not (Test-Path -Path (Join-Path $Layout.Root $relative) -PathType Leaf)) { return $false }
    }
    return $true
}

# ── install state ───────────────────────────────────────────────────────────

function Read-InstallState {
    param([Parameter(Mandatory = $true)][string]$StateFile)
    if (-not (Test-Path -Path $StateFile -PathType Leaf)) { return $null }
    try { return (Get-Content -Path $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json) }
    catch { return $null }
}

function Write-InstallState {
    param(
        [Parameter(Mandatory = $true)][string]$StateFile,
        [Parameter(Mandatory = $true)]$State
    )
    $directory = Split-Path -Path $StateFile -Parent
    if (-not (Test-Path $directory)) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    $temporary = $StateFile + '.tmp'
    $json = $State | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($temporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -Path $temporary -Destination $StateFile -Force
    return $true
}

function New-InstallState {
    param($Layout, $Manifest, [int]$Port, [string]$LockSha256, [string]$TransferMode, [string]$Status = 'complete')
    $stamp = (Get-Date).ToUniversalTime().ToString('o')
    return [ordered]@{
        schema_version = '1.0'
        product        = [string]$Manifest.product
        version        = [string]$Manifest.version
        status         = $Status
        installed_at   = $stamp
        updated_at     = $stamp
        port           = $Port
        layout         = [ordered]@{
            root        = $Layout.Root
            app         = $Layout.App
            runtime     = $Layout.Runtime
            data        = $Layout.Data
            config_path = $Layout.ConfigPath
            cache       = $Layout.Cache
            logs        = $Layout.Logs
        }
        runtime        = [ordered]@{
            python_version   = [string]$Manifest.python_version
            interpreter      = $Layout.Interpreter
            lock_sha256      = $LockSha256
            torch_packages   = @($Manifest.runtime.torch.packages)
            torch_index_url  = [string]$Manifest.runtime.torch.index_url
        }
        package        = [ordered]@{
            version         = [string]$Manifest.version
            transfer_mode   = $TransferMode
            manifest_sha256 = $LockSha256
        }
        upgrades       = @()
    }
}

function Test-InstallCoreUsable {
    # The part of an installation that costs a runtime and a download to build.
    # Returns '' when it is intact, otherwise the first thing that is missing.
    # The permanent entry points are deliberately not part of it: they are
    # written from the package in seconds and are repairable on their own.
    param($Layout, $State)

    if ($null -eq $State) { return 'MISSING' }
    if ([string]$State.schema_version -ne '1.0') { return 'SCHEMA' }
    if ([string]$State.status -ne 'complete') { return 'STATUS' }
    if (-not (Test-Path -Path (Join-Path $Layout.App 'auto_tune\main.py') -PathType Leaf)) { return 'APP' }
    if (-not (Test-Path -Path $Layout.Interpreter -PathType Leaf)) { return 'RUNTIME' }
    if (-not (Test-Path -Path $Layout.ConfigPath -PathType Leaf)) { return 'CONFIG' }
    return ''
}

function Test-InstallStateUsable {
    param($Layout, $State)

    $reason = Test-InstallCoreUsable -Layout $Layout -State $State
    if ($reason -eq 'MISSING') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' `
                -Message '安装状态文件不存在。请先运行 install.bat 完成安装。')
    }
    if ($reason -eq 'SCHEMA') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' `
                -Message ("安装状态版本不受支持（{0}），请重新运行 install.bat。" -f $State.schema_version))
    }
    if ($reason -eq 'STATUS') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' `
                -Message ("上一次安装未完成（状态：{0}）。请重新运行 install.bat，安装会从断点继续。" -f $State.status))
    }
    if ($reason -eq 'APP') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' -Message '程序文件缺失，请重新运行 install.bat。')
    }
    if ($reason -eq 'RUNTIME') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' -Message '私有运行环境缺失，请重新运行 install.bat。')
    }
    if ($reason -eq 'CONFIG') {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' -Message '配置文件缺失，请重新运行 install.bat。')
    }
    if (-not (Test-LauncherInstalled -Layout $Layout)) {
        throw (New-DeliveryFailure -Code 'INSTALL_STATE_INCOMPLETE' `
                -Message '启动入口缺失，安装不完整，请重新运行 install.bat。')
    }
    return $true
}

function Get-StudioPort {
    param($Layout, $State, [string]$EnvPort = $null)

    if ([string]::IsNullOrWhiteSpace($EnvPort)) { $EnvPort = $env:AUTO_TUNE_PORT }
    if (-not [string]::IsNullOrWhiteSpace($EnvPort)) {
        if ($EnvPort -notmatch '^[0-9]+$') {
            throw (New-DeliveryFailure -Code 'PORT_INVALID' `
                    -Message 'AUTO_TUNE_PORT 必须是 1-65535 之间的整数。')
        }
        $port = [int]$EnvPort
        if ($port -lt 1 -or $port -gt 65535) {
            throw (New-DeliveryFailure -Code 'PORT_INVALID' `
                    -Message 'AUTO_TUNE_PORT 必须是 1-65535 之间的整数。')
        }
        return $port
    }
    if ($null -ne $State -and $null -ne $State.port) {
        $parsed = 0
        if ([int]::TryParse([string]$State.port, [ref]$parsed) -and $parsed -ge 1 -and $parsed -le 65535) {
            return $parsed
        }
    }
    return 8000
}

# ── external processes ──────────────────────────────────────────────────────

function Set-ProcessEnvironment {
    param($Environment)
    if ($null -eq $Environment) { return @{} }
    $previous = @{}
    $keys = @()
    if ($Environment -is [System.Collections.IDictionary]) { $keys = @($Environment.Keys) }
    else { $keys = @($Environment.PSObject.Properties | ForEach-Object { $_.Name }) }
    foreach ($key in $keys) {
        $name = [string]$key
        $previous[$name] = [System.Environment]::GetEnvironmentVariable($name)
        $value = $Environment[$name]
        if ($Environment -isnot [System.Collections.IDictionary]) {
            $value = $Environment.PSObject.Properties[$name].Value
        }
        Set-Item -Path ("env:{0}" -f $name) -Value ([string]$value)
    }
    return $previous
}

function Restore-ProcessEnvironment {
    param($Previous)
    if ($null -eq $Previous) { return }
    foreach ($key in @($Previous.Keys)) {
        Set-Item -Path ("env:{0}" -f $key) -Value ([string]$Previous[$key])
    }
}

function Invoke-ExternalProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = '',
        $Environment = $null,
        [string]$LogFile = $null,
        [string]$InstallRoot = ''
    )
    if ($LogFile) {
        Write-DeliveryLog -LogFile $LogFile -InstallRoot $InstallRoot `
            -Message ("运行 {0}" -f ([System.IO.Path]::GetFileName($FilePath)))
    }
    $previous = Set-ProcessEnvironment -Environment $Environment
    $savedErrorAction = $ErrorActionPreference
    $location = $null
    try {
        # A native program that writes to stderr is not a failure — pip warns on
        # every install, for instance. Only the exit code decides. Under the
        # default 'Stop' preference PowerShell 5.1 turns the first stderr line
        # of a native command into a terminating error, which would abort a
        # perfectly good installation.
        $ErrorActionPreference = 'Continue'
        if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            $location = Get-Location
            Set-Location -Path $WorkingDirectory
        }
        $output = & $FilePath @Arguments 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) { $exitCode = 0 }
        return @{ ExitCode = [int]$exitCode; StdOut = $output; StdErr = '' }
    } finally {
        $ErrorActionPreference = $savedErrorAction
        if ($null -ne $location) { Set-Location -Path $location }
        Restore-ProcessEnvironment -Previous $previous
    }
}

function Invoke-RunnerChecked {
    param($Runner, [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory,
        $Environment, [string]$LogFile, [string]$InstallRoot, [string]$Code, [string]$Message)
    $result = & $Runner -FilePath $FilePath -Arguments $Arguments `
        -WorkingDirectory $WorkingDirectory -Environment $Environment -LogFile $LogFile
    $exitCode = 0
    if ($null -ne $result -and $null -ne $result.ExitCode) { $exitCode = [int]$result.ExitCode }
    if ($exitCode -ne 0) {
        $detail = ''
        if ($null -ne $result) { $detail = [string]$result.StdOut }
        Write-DeliveryLog -LogFile $LogFile -Level 'ERROR' -InstallRoot $InstallRoot -Message $detail
        throw (New-DeliveryFailure -Code $Code -Message $Message)
    }
    return $result
}

# ── private runtime ─────────────────────────────────────────────────────────
# The runtime is installed from the package's offline bundle and from nothing
# else: the pinned Miniconda becomes the private interpreter, and every wheel —
# the CUDA PyTorch pair first, then the locked dependencies — comes from the
# bundle's wheelhouse with ``pip --no-index --find-links``. No index, no mirror
# and no fallback exists, so a machine without a network installs identically.

function Install-PrivateRuntime {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        $Runner = $null,
        [string]$LogFile = $null,
        [switch]$RefreshOnly
    )
    if ($null -eq $Runner) { $Runner = ${function:Invoke-ExternalProcess} }
    $root = $Layout.Root
    $offline = Get-OfflineLayout -PackageRoot $PackageRoot -Manifest $Manifest
    $interpreterDirectory = Split-Path -Path $Layout.Interpreter -Parent

    if ($RefreshOnly) {
        # The interpreter is already there; an upgrade only re-installs the
        # dependencies the new lock pins.
        if (-not (Test-Path -Path $Layout.Interpreter -PathType Leaf)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                    -Message '私有运行环境缺失，无法只更新依赖。')
        }
        Write-DeliveryLog -LogFile $LogFile -InstallRoot $root -Message '依赖清单已变化，更新私有运行环境依赖。'
    } else {
        if (-not (Test-Path -PathType Leaf $offline.Miniconda)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                    -Message '安装包内缺少离线 Python 安装程序，安装包不完整，已拒绝安装。')
        }
        # Per-user, silent, without touching PATH or the registry, and straight
        # into the private runtime directory: that python.exe *is* the product's
        # interpreter, so no second environment has to be created anywhere.
        Invoke-RunnerChecked -Runner $Runner -FilePath $offline.Miniconda `
            -Arguments @('/InstallationType=JustMe', '/AddToPath=0', '/RegisterPython=0', '/S',
                ("/D={0}" -f $interpreterDirectory)) `
            -WorkingDirectory $root -Environment $null -LogFile $LogFile -InstallRoot $root `
            -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
            -Message '私有 Python 运行环境安装失败（离线静默安装未完成）。' | Out-Null

        if (-not (Test-Path -Path $Layout.Interpreter -PathType Leaf)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                    -Message ("Python {0} 运行环境安装失败：未找到 python.exe。" -f [string]$Manifest.python_version))
        }
    }

    $requirements = Join-Path $PackageRoot ([string]$Manifest.runtime.pip_requirements)
    if (-not (Test-Path -PathType Leaf $requirements)) {
        throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                -Message '安装包内缺少锁定依赖清单，安装包不完整，已拒绝安装。')
    }
    $torchWheels = @()
    foreach ($wheel in @($Manifest.runtime.torch.wheels)) {
        $path = Join-Path $offline.Wheelhouse ([string]$wheel.file_name)
        if (-not (Test-Path -PathType Leaf $path)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_BUNDLE_MISSING' `
                    -Message ("安装包内缺少离线 CUDA wheel {0}，安装包不完整，已拒绝安装。" -f $wheel.file_name))
        }
        $torchWheels += $path
    }

    $pipCache = Join-Path $Layout.Cache 'pip'
    if (-not (Test-Path $pipCache)) { New-Item -ItemType Directory -Force -Path $pipCache | Out-Null }
    # PIP_NO_INDEX and PIP_FIND_LINKS repeat what the command line says: even a
    # configuration file on the machine cannot make pip reach an index.
    $pipEnvironment = @{
        PIP_CACHE_DIR                 = $pipCache
        PIP_NO_INDEX                  = '1'
        PIP_FIND_LINKS                = $offline.Wheelhouse
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PIP_NO_INPUT                  = '1'
        PIP_NO_DEPS                   = '0'
        PYTHONNOUSERSITE              = '1'
    }

    # The CUDA build first and from the package: a plain `pip install torch`
    # could resolve a CPU wheel and silently deliver a product that cannot train.
    Invoke-RunnerChecked -Runner $Runner -FilePath $Layout.Interpreter `
        -Arguments (@('-m', 'pip', 'install', '--no-index', '--find-links', $offline.Wheelhouse) + $torchWheels) `
        -WorkingDirectory $root -Environment $pipEnvironment -LogFile $LogFile -InstallRoot $root `
        -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
        -Message '安装离线 CUDA 版 PyTorch 失败。本产品不安装 CPU 版 PyTorch。' | Out-Null

    Invoke-RunnerChecked -Runner $Runner -FilePath $Layout.Interpreter `
        -Arguments @('-m', 'pip', 'install', '--no-index', '--find-links', $offline.Wheelhouse,
            '-r', $requirements) `
        -WorkingDirectory $root -Environment $pipEnvironment -LogFile $LogFile -InstallRoot $root `
        -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
        -Message '安装离线锁定运行依赖失败。' | Out-Null

    # Finally the runtime answers for itself: interpreter, Python version, the
    # pinned CUDA PyTorch and Torchvision, CUDA 12.1, a visible GPU and the
    # delivery's importable components. A CPU wheel is refused here.
    $pythonPath = Join-Path $PackageRoot 'payload'
    if (-not (Test-Path -PathType Container $pythonPath)) { $pythonPath = $Layout.App }
    $checkEnvironment = @{
        PYTHONPATH               = $pythonPath
        PYTHONUNBUFFERED         = '1'
        PYTHONDONTWRITEBYTECODE  = '1'
    }
    Invoke-RunnerChecked -Runner $Runner -FilePath $Layout.Interpreter `
        -Arguments @('-m', 'auto_tune.delivery.preflight', '--require-offline-runtime',
            '--expect-python', $Layout.Interpreter) `
        -WorkingDirectory $Layout.Data -Environment $checkEnvironment -LogFile $LogFile -InstallRoot $root `
        -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
        -Message '离线预检未通过：私有运行环境不是交付锁定的 CUDA PyTorch 运行环境，本产品不提供 CPU 回退。' | Out-Null

    return $Layout.Interpreter
}

function Get-RuntimeKey {
    # The identity of the runtime a package asks for: the pinned requirements
    # *and* the offline bundle that provides them. The requirements file names
    # the versions; the verified offline lock names the exact bytes. Two packages
    # with the same pins but a rebuilt wheelhouse are two different runtimes, so
    # an upgrade must build the second one instead of reusing the first.
    #
    # Callers verify the bundle before asking, so the hash is of a lock the
    # installer has already accepted.
    #
    # ``v2`` marks the shape: a stamp written by an older delivery carries a key
    # without the bundle identity (or no key at all). Such a key never equals a
    # current one, so the runtime is rebuilt once — the honest answer, because
    # the bytes it was built from are unknowable.
    param($Manifest, [string]$PackageRoot)
    $lockPath = Join-Path $PackageRoot ([string]$Manifest.runtime.pip_requirements)
    $lockSha256 = ''
    if (Test-Path -Path $lockPath -PathType Leaf) {
        $lockSha256 = Get-FileSha256 -Path $lockPath
    }
    $offlineSha256 = ''
    $offlineLock = (Get-OfflineLayout -PackageRoot $PackageRoot -Manifest $Manifest).Lock
    if (Test-Path -Path $offlineLock -PathType Leaf) {
        $offlineSha256 = Get-FileSha256 -Path $offlineLock
    }
    $packages = @($Manifest.runtime.torch.packages) -join ','
    return ("v2|{0}|{1}|{2}" -f $lockSha256, $offlineSha256, $packages)
}

function Get-RuntimeStampPath {
    param([Parameter(Mandatory = $true)]$Layout)
    return Join-Path $Layout.Runtime 'runtime-stamp.json'
}

function Read-RuntimeStamp {
    param([Parameter(Mandatory = $true)]$Layout)
    return (Read-InstallState -StateFile (Get-RuntimeStampPath -Layout $Layout))
}

function Write-RuntimeStamp {
    param([Parameter(Mandatory = $true)]$Layout, [string]$RuntimeKey)
    $stamp = [ordered]@{
        schema_version = '1.0'
        runtime_key    = $RuntimeKey
        python_version = ''
        interpreter    = $Layout.Interpreter
        installed_at   = (Get-Date).ToUniversalTime().ToString('o')
    }
    Write-InstallState -StateFile (Get-RuntimeStampPath -Layout $Layout) -State $stamp | Out-Null
    return $true
}

function Ensure-PrivateRuntime {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        $Runner = $null,
        $RuntimeInstaller = $null,
        [string]$LogFile = $null,
        [switch]$RefreshDependencies
    )
    # Only a runtime whose stamp matches the package's dependency identity counts
    # as ready. An interrupted install (or a changed lock) leaves no matching
    # stamp, so re-running install.bat or upgrade.bat finishes the job instead of
    # trusting a half-built environment.
    $runtimeKey = Get-RuntimeKey -Manifest $Manifest -PackageRoot $PackageRoot
    $stamp = Read-RuntimeStamp -Layout $Layout
    $stamped = ($null -ne $stamp -and [string]$stamp.runtime_key -eq $runtimeKey)
    $interpreterExists = Test-Path -Path $Layout.Interpreter -PathType Leaf
    if ($null -eq $RuntimeInstaller) { $RuntimeInstaller = ${function:Install-PrivateRuntime} }

    if ($interpreterExists -and $stamped -and -not $RefreshDependencies) {
        return @{ Interpreter = $Layout.Interpreter; Installed = $false; TransferMode = 'reused' }
    }

    if ($interpreterExists) {
        # The environment stays; the pinned dependencies are installed into it
        # again (a changed lock, or an install that stopped before the pip step).
        [void](& $RuntimeInstaller -Layout $Layout -Manifest $Manifest `
                -Runner $Runner -LogFile $LogFile -PackageRoot $PackageRoot -RefreshOnly)
        if (-not (Test-Path -Path $Layout.Interpreter -PathType Leaf)) {
            throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                    -Message '私有运行环境更新后仍缺少解释器。')
        }
        [void](Write-RuntimeStamp -Layout $Layout -RuntimeKey $runtimeKey)
        return @{ Interpreter = $Layout.Interpreter; Installed = $true; TransferMode = 'dependency-update' }
    }

    [void](& $RuntimeInstaller -Layout $Layout -Manifest $Manifest `
            -Runner $Runner -LogFile $LogFile -PackageRoot $PackageRoot)

    if (-not (Test-Path -Path $Layout.Interpreter -PathType Leaf)) {
        throw (New-DeliveryFailure -Code 'OFFLINE_RUNTIME_INSTALL_FAILED' `
                -Message '私有运行环境安装后仍缺少解释器。')
    }
    [void](Write-RuntimeStamp -Layout $Layout -RuntimeKey $runtimeKey)
    return @{ Interpreter = $Layout.Interpreter; Installed = $true; TransferMode = 'offline' }
}

function Invoke-Preflight {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [string[]]$Arguments = @('--require-gpu'),
        $Runner = $null,
        [int]$Port = 8000,
        [string]$LogFile = $null
    )
    $environment = New-StudioEnvironment -Layout $Layout -Port $Port
    $fullArguments = @('-m', 'auto_tune.delivery.preflight') + $Arguments
    if ($null -ne $Runner) {
        return & $Runner -Python $Layout.Interpreter -Arguments $fullArguments `
            -WorkingDirectory $Layout.Data -Environment $environment -LogFile $LogFile
    }
    return Invoke-ExternalProcess -FilePath $Layout.Interpreter -Arguments $fullArguments `
        -WorkingDirectory $Layout.Data -Environment $environment -LogFile $LogFile `
        -InstallRoot $Layout.Root
}

function New-StudioEnvironment {
    param($Layout, [int]$Port)
    return [ordered]@{
        AUTO_TUNE_APP_ROOT     = $Layout.Data
        AUTO_TUNE_CONFIG_PATH  = $Layout.ConfigPath
        AUTO_TUNE_DATASETS_DIR = $Layout.Datasets
        AUTO_TUNE_HOST         = '127.0.0.1'
        AUTO_TUNE_PORT         = [string]$Port
        PYTHONPATH             = $Layout.App
        PYTHONUNBUFFERED       = '1'
        PYTHONDONTWRITEBYTECODE = '1'
    }
}

function Get-RequiredFreeBytes {
    param($Manifest)
    if ($null -eq $Manifest -or $null -eq $Manifest.install) { return 0 }
    $value = $Manifest.install.required_free_bytes
    if ($null -eq $value) { return 0 }
    try { return [long]$value } catch { return 0 }
}

# ── install ─────────────────────────────────────────────────────────────────

function Stage-Payload {
    param($Layout, [string]$PackageRoot, $Lock, [string]$LogFile)
    $stagingRoot = Join-Path $Layout.Root ('.staging-' + [guid]::NewGuid().ToString('n'))
    $stagingApp = Join-Path $stagingRoot 'app'
    try {
        New-Item -ItemType Directory -Force -Path $stagingApp | Out-Null
        $copied = Copy-PackagePayload -PackageRoot $PackageRoot -Lock $Lock `
            -Destination $stagingApp -LogFile $LogFile -InstallRoot $Layout.Root
        if ($copied -le 0) {
            throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' -Message '安装包中没有可安装的程序文件。')
        }
        return @{ Root = $stagingRoot; App = $stagingApp }
    } catch {
        Remove-Item -Path $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
        throw
    }
}

function Install-StagedPayload {
    param($Layout, $Staging, [string]$LogFile)
    if (Test-Path $Layout.App) {
        Remove-Item -Path $Layout.App -Recurse -Force
    }
    Move-Item -Path $Staging.App -Destination $Layout.App -Force
    Remove-Item -Path $Staging.Root -Recurse -Force -ErrorAction SilentlyContinue
    Write-DeliveryLog -LogFile $LogFile -InstallRoot $Layout.Root -Message '程序文件已就位。'
    return $Layout.App
}

function Invoke-Install {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        $Probe = $null,
        $Runner = $null,
        $RuntimeInstaller = $null,
        $PreflightRunner = $null,
        $Linker = $null
    )
    $logFile = Join-Path $Layout.Logs 'install.log'
    $manifest = $null
    $transferMode = 'none'
    $launcher = $null
    # What the state file was when this run began, whether it described a
    # complete installation, and whether this run ever started building: a
    # failure may only leave ``incomplete`` behind when it has something to
    # finish and nothing usable to fall back on.
    $priorStateBytes = $null
    $priorStateHash = ''
    $priorUsable = $false
    $building = $false

    try {
        # 1. the package is read and verified first: a tampered or incomplete
        #    archive must not influence anything that follows.
        if (Test-Path -Path $Layout.StateFile -PathType Leaf) {
            $priorStateBytes = [System.IO.File]::ReadAllBytes($Layout.StateFile)
            $priorStateHash = Get-FileSha256 -Path $Layout.StateFile
        }
        $manifest = Get-PackageManifest -PackageRoot $PackageRoot
        $port = [int]$Layout.Port
        if ($port -lt 1 -or $port -gt 65535) { $port = 8000 }

        $lock = Get-PackageLock -PackageRoot $PackageRoot
        # The offline bundle is verified before anything is written *and before
        # the whole-package lock*: a package without the pinned Miniconda, the
        # CUDA wheels or a complete wheelhouse is refused while the machine is
        # still untouched, and the operator is told which offline file is wrong
        # instead of a generic package mismatch. The package lock still covers
        # everything else, including the bundle.
        Assert-OfflineBundle -PackageRoot $PackageRoot -Manifest $manifest | Out-Null
        Assert-PackageIntegrity -PackageRoot $PackageRoot -Lock $lock | Out-Null

        # 2. what is already installed decides how much work this run has left to
        #    do, and therefore how much disk it may legitimately ask for.
        $previous = Read-InstallState -StateFile $Layout.StateFile
        $priorUsable = ($null -ne $previous -and [string]$previous.status -eq 'complete' -and
            ((Test-InstallCoreUsable -Layout $Layout -State $previous) -eq ''))
        $lockSha256 = Get-FileSha256 -Path (Join-Path $PackageRoot ([string]$manifest.runtime.pip_requirements))

        # 3. the checks that apply to every run, whatever the answer was
        Assert-InstallPreconditions -Layout $Layout -Probe $Probe | Out-Null
        [void](Initialize-DeliveryLayout -Layout $Layout)
        Assert-DirectoryWritable -Layout $Layout -Probe $Probe

        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root `
            -Message ("开始安装 {0} {1}" -f $manifest.product, $manifest.version)

        $repairable = $priorUsable -and ([string]$previous.version -eq [string]$manifest.version)
        if ($repairable) {
            # The installation is complete and of this version: only the permanent
            # entry points are refreshed, because a package of the same version can
            # carry a fixed launcher. Nothing is downloaded, nothing is rebuilt and
            # the budget for a fresh installation is not owed.
            $mode = 'launcher-repair'
            if (Test-LauncherInstalled -Layout $Layout) { $mode = 'already-installed' }
            $launcher = Install-DeliveryLauncher -Layout $Layout -PackageRoot $PackageRoot -Lock $lock `
                -LogFile $logFile -Linker $Linker
            [void](Write-InstallLocation -Layout $Layout)
            $message = '当前已经安装该版本。'
            if ($mode -eq 'launcher-repair') { $message = '当前已经安装该版本，已补齐启动与卸载入口。' }
            return @{ Ok = $true; Mode = $mode; ErrorCode = $null; Message = $message
                Interpreter = $Layout.Interpreter; LockSha256 = $lockSha256; TransferMode = 'reused'
                LauncherInstalled = $true; ShortcutCreated = [bool]$launcher.ShortcutCreated
                Shortcut = $launcher.Shortcut }
        }

        # 4. from here on a private runtime may have to be built, so the full
        #    installation budget is owed before anything is downloaded or written.
        Assert-SufficientFreeBytes -Layout $Layout -Probe $Probe `
            -RequiredFreeBytes (Get-RequiredFreeBytes -Manifest $manifest) | Out-Null

        $building = $true
        $runtime = Ensure-PrivateRuntime -Layout $Layout -Manifest $manifest -PackageRoot $PackageRoot `
            -Runner $Runner -RuntimeInstaller $RuntimeInstaller -LogFile $logFile
        $transferMode = $runtime.TransferMode

        $staging = Stage-Payload -Layout $Layout -PackageRoot $PackageRoot -Lock $lock -LogFile $logFile
        Install-StagedPayload -Layout $Layout -Staging $staging -LogFile $logFile | Out-Null

        $launcher = Install-DeliveryLauncher -Layout $Layout -PackageRoot $PackageRoot -Lock $lock `
            -LogFile $logFile -Linker $Linker

        [void](Initialize-UserConfig -Layout $Layout -LogFile $logFile)

        $preflight = Invoke-Preflight -Layout $Layout -Arguments @('--require-gpu') `
            -Runner $PreflightRunner -Port $port -LogFile $logFile
        $exitCode = 0
        if ($null -ne $preflight -and $null -ne $preflight.ExitCode) { $exitCode = [int]$preflight.ExitCode }
        if ($exitCode -ne 0) {
            throw (New-DeliveryFailure -Code 'PREFLIGHT_FAILED' `
                    -Message '共享交付预检未通过：所安装的运行环境看不到可用的 NVIDIA GPU。本产品不提供 CPU 回退，请检查显卡驱动。')
        }

        $state = New-InstallState -Layout $Layout -Manifest $manifest -Port $port `
            -LockSha256 $lockSha256 -TransferMode $transferMode
        if ($null -ne $previous -and $null -ne $previous.upgrades) {
            $state['upgrades'] = @($previous.upgrades)
        }
        Write-InstallState -StateFile $Layout.StateFile -State $state | Out-Null
        [void](Write-InstallLocation -Layout $Layout)

        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message '安装完成。'
        return @{ Ok = $true; Mode = 'install'; ErrorCode = $null; Message = '安装完成。'
            Interpreter = $Layout.Interpreter; LockSha256 = $lockSha256; TransferMode = $transferMode
            LauncherInstalled = $true; ShortcutCreated = [bool]$launcher.ShortcutCreated
            Shortcut = $launcher.Shortcut }
    } catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
            -Message ("安装失败 {0}：{1}" -f $info.Code, $info.Message)
        # ``incomplete`` is a promise to the operator that the next run will
        # finish the job. It may only be written when this run really started
        # building something and there is no complete installation to fall back
        # on: a failure at the package, the machine, the disk or the entry
        # points leaves the state file exactly as it was, so an installation
        # that works is never downgraded by a run that changed nothing.
        if ($building -and -not $priorUsable) {
            if (Test-Path $Layout.Root) {
                $partialVersion = ''
                if ($null -ne $manifest) { $partialVersion = [string]$manifest.version }
                $partial = @{
                    schema_version = '1.0'
                    product        = 'auto-tune-studio'
                    version        = $partialVersion
                    status         = 'incomplete'
                    updated_at     = (Get-Date).ToUniversalTime().ToString('o')
                    layout         = @{ root = $Layout.Root; app = $Layout.App; data = $Layout.Data
                        config_path = $Layout.ConfigPath }
                    runtime        = @{ interpreter = $Layout.Interpreter; lock_sha256 = '' }
                    upgrades       = @()
                }
                try { Write-InstallState -StateFile $Layout.StateFile -State $partial | Out-Null } catch { }
            }
        } elseif ($null -ne $priorStateBytes) {
            # Nothing that could justify touching the state was built, or a
            # complete installation already existed: whatever happens next, the
            # file must come out of this run byte for byte as it went in.
            $currentHash = Get-FileSha256OrEmpty -Path $Layout.StateFile
            if ($currentHash -ne $priorStateHash) {
                try { [System.IO.File]::WriteAllBytes($Layout.StateFile, $priorStateBytes) } catch { }
            }
        }
        return @{ Ok = $false; Mode = 'install'; ErrorCode = $info.Code; Message = $info.Message
            Interpreter = $Layout.Interpreter; LockSha256 = ''; TransferMode = $transferMode }
    }
}

# ── upgrade ─────────────────────────────────────────────────────────────────

function New-StagedLayout {
    # A layout that points at the staging tree instead of the live installation.
    # Everything else — the user data, the configuration, the verified cache —
    # is deliberately the live one: staging must not create a second copy of
    # anything the operator owns.
    param([Parameter(Mandatory = $true)]$Layout, [Parameter(Mandatory = $true)][string]$StagingRoot)
    $staged = $Layout.PSObject.Copy()
    $staged.App = Join-Path $StagingRoot 'app'
    $staged.Runtime = Join-Path $StagingRoot 'runtime'
    $staged.Interpreter = Join-Path (Join-Path $staged.Runtime 'py310') 'python.exe'
    $staged.CondaHome = Join-Path $staged.Runtime 'py310'
    return $staged
}

function Invoke-PreflightExitCode {
    param($Layout, $Runner, [int]$Port, [string]$LogFile)
    $preflight = Invoke-Preflight -Layout $Layout -Arguments @('--require-gpu') `
        -Runner $Runner -Port $Port -LogFile $LogFile
    if ($null -ne $preflight -and $null -ne $preflight.ExitCode) { return [int]$preflight.ExitCode }
    return 0
}

function Invoke-Upgrade {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        $Probe = $null,
        $Runner = $null,
        $RuntimeInstaller = $null,
        $PreflightRunner = $null,
        $Linker = $null
    )
    $logFile = Join-Path $Layout.Logs 'upgrade.log'
    $stagingRoot = Join-Path $Layout.Root ('.staging-' + [guid]::NewGuid().ToString('n'))
    $rollbackApp = Join-Path $Layout.Root '.rollback-app'
    $rollbackRuntime = Join-Path $Layout.Root '.rollback-runtime'
    $appSwapped = $false
    $runtimeSwapped = $false
    $runtimeUpdated = $false
    $stateRaw = $null
    $encoding = New-Object System.Text.UTF8Encoding($false)

    try {
        if (-not (Test-Path $Layout.Logs)) { New-Item -ItemType Directory -Force -Path $Layout.Logs | Out-Null }
        $state = Read-InstallState -StateFile $Layout.StateFile
        Test-InstallStateUsable -Layout $Layout -State $state | Out-Null
        $stateRaw = [System.IO.File]::ReadAllText($Layout.StateFile, [System.Text.Encoding]::UTF8)

        $manifest = Get-PackageManifest -PackageRoot $PackageRoot
        $lock = Get-PackageLock -PackageRoot $PackageRoot

        # The new package is verified *before* anything is replaced: a partly
        # downloaded or tampered archive must not disturb the running version.
        # A package whose offline bundle is not the verified one is refused
        # before the staged tree, the download or the swap — and before the
        # whole-package lock, so the report names the offline file: the installed
        # version must never be disturbed by an unusable package.
        Assert-OfflineBundle -PackageRoot $PackageRoot -Manifest $manifest | Out-Null
        Assert-PackageIntegrity -PackageRoot $PackageRoot -Lock $lock | Out-Null
        Assert-PackageManifest -Manifest $manifest -PackageRoot $PackageRoot | Out-Null

        $lockSha256 = Get-FileSha256 -Path (Join-Path $PackageRoot ([string]$manifest.runtime.pip_requirements))
        $previousVersion = [string]$state.version

        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root `
            -Message ("开始升级 {0} -> {1}" -f $previousVersion, $manifest.version)

        $runtimeKey = Get-RuntimeKey -Manifest $manifest -PackageRoot $PackageRoot
        $stamp = Read-RuntimeStamp -Layout $Layout
        # An unchanged dependency lock reuses the runtime as it is. A changed (or
        # unfinished) one is *built beside it*, never modified in place: a failed
        # dependency install must leave a version the operator can still run.
        $runtimeChanged = ($null -eq $stamp) -or ([string]$stamp.runtime_key -ne $runtimeKey)
        $port = Get-StudioPort -Layout $Layout -State $state -EnvPort ''

        if ($runtimeChanged) {
            # Only the side-by-side runtime needs the budget of a fresh
            # installation; replacing the program alone costs almost nothing and
            # must not be blocked on a full disk. Checked before the staging tree
            # exists, so a refusal leaves no staging or rollback directory, starts
            # no download and does not touch the installed version.
            Assert-SufficientFreeBytes -Layout $Layout -Probe $Probe `
                -RequiredFreeBytes (Get-RequiredFreeBytes -Manifest $manifest) | Out-Null
        }

        # 1. stage the new program, and when the lock changed a complete second
        #    runtime, beside the live installation
        $stagingApp = Join-Path $stagingRoot 'app'
        New-Item -ItemType Directory -Force -Path $stagingApp | Out-Null
        $copied = Copy-PackagePayload -PackageRoot $PackageRoot -Lock $lock -Destination $stagingApp `
            -LogFile $logFile -InstallRoot $Layout.Root
        if ($copied -le 0) {
            throw (New-DeliveryFailure -Code 'PACKAGE_HASH_MISMATCH' -Message '安装包中没有可安装的程序文件。')
        }

        if ($runtimeChanged) {
            $stagedLayout = New-StagedLayout -Layout $Layout -StagingRoot $stagingRoot
            Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root `
                -Message '依赖清单已变化：在独立暂存目录中构建新的私有运行环境。'
            [void](Ensure-PrivateRuntime -Layout $stagedLayout -Manifest $manifest -PackageRoot $PackageRoot `
                -Runner $Runner -RuntimeInstaller $RuntimeInstaller -LogFile $logFile)
            # The new program *and* the new runtime are preflighted together, in
            # the staging tree, while the installed pair is still untouched.
            if ((Invoke-PreflightExitCode -Layout $stagedLayout -Runner $PreflightRunner -Port $port `
                    -LogFile $logFile) -ne 0) {
                throw (New-DeliveryFailure -Code 'PREFLIGHT_FAILED' `
                        -Message '新运行环境未通过交付预检，升级已放弃，当前版本未被改动。')
            }
        }

        # 2. switch: move the live pair aside, then move the staged pair in
        if (Test-Path $rollbackApp) { Remove-Item -Path $rollbackApp -Recurse -Force }
        Move-Item -Path $Layout.App -Destination $rollbackApp
        $appSwapped = $true
        Move-Item -Path $stagingApp -Destination $Layout.App -Force

        if ($runtimeChanged) {
            if (Test-Path $rollbackRuntime) { Remove-Item -Path $rollbackRuntime -Recurse -Force }
            Move-Item -Path $Layout.Runtime -Destination $rollbackRuntime
            Move-Item -Path (Join-Path $stagingRoot 'runtime') -Destination $Layout.Runtime -Force
            $runtimeSwapped = $true
            [void](Write-RuntimeStamp -Layout $Layout -RuntimeKey $runtimeKey)
            $runtimeUpdated = $true
        }

        # 3. the installed program is verified with the runtime it will use
        if ((Invoke-PreflightExitCode -Layout $Layout -Runner $PreflightRunner -Port $port `
                -LogFile $logFile) -ne 0) {
            throw (New-DeliveryFailure -Code 'PREFLIGHT_FAILED' `
                    -Message '升级后共享交付预检未通过，已恢复到上一个版本。')
        }

        $launcher = Install-DeliveryLauncher -Layout $Layout -PackageRoot $PackageRoot -Lock $lock `
            -LogFile $logFile -Linker $Linker

        # 4. only now is what was replaced removed
        Remove-Item -Path $rollbackApp -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $rollbackRuntime -Recurse -Force -ErrorAction SilentlyContinue

        $state.version = [string]$manifest.version
        $state.status = 'complete'
        $state.updated_at = (Get-Date).ToUniversalTime().ToString('o')
        $state.runtime.lock_sha256 = $lockSha256
        $state.runtime.interpreter = $Layout.Interpreter
        $state.package.version = [string]$manifest.version
        $history = @()
        if ($null -ne $state.upgrades) { $history = @($state.upgrades) }
        $state.upgrades = $history + @([ordered]@{
                at             = (Get-Date).ToUniversalTime().ToString('o')
                from           = $previousVersion
                to             = [string]$manifest.version
                runtime_updated = $runtimeUpdated
            })
        Write-InstallState -StateFile $Layout.StateFile -State $state | Out-Null
        [void](Write-InstallLocation -Layout $Layout)

        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message '升级完成，用户数据未改动。'
        return @{ Ok = $true; ErrorCode = $null; Message = '升级完成。'; RuntimeUpdated = $runtimeUpdated
            Version = [string]$manifest.version
            LauncherInstalled = $true; ShortcutCreated = [bool]$launcher.ShortcutCreated
            Shortcut = $launcher.Shortcut }
    } catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        $restored = $true
        try {
            # The runtime goes back first: the program moved into place must not
            # be able to find a half-built interpreter.
            if ($runtimeSwapped) {
                if (Test-Path $Layout.Runtime) {
                    Remove-Item -Path $Layout.Runtime -Recurse -Force -ErrorAction SilentlyContinue
                }
                if (Test-Path $rollbackRuntime) {
                    Move-Item -Path $rollbackRuntime -Destination $Layout.Runtime -Force
                }
            }
            if ($appSwapped) {
                if (Test-Path $Layout.App) {
                    Remove-Item -Path $Layout.App -Recurse -Force -ErrorAction SilentlyContinue
                }
                if (Test-Path $rollbackApp) {
                    Move-Item -Path $rollbackApp -Destination $Layout.App -Force
                }
            }
            if ($null -ne $stateRaw -and (Test-Path $Layout.StateFile)) {
                $current = [System.IO.File]::ReadAllText($Layout.StateFile, [System.Text.Encoding]::UTF8)
                if ($current -ne $stateRaw) {
                    [System.IO.File]::WriteAllText($Layout.StateFile, $stateRaw, $encoding)
                }
            }
        } catch {
            $restored = $false
        }
        if ($restored) {
            Write-DeliveryLog -LogFile $logFile -Level 'WARN' -InstallRoot $Layout.Root `
                -Message '升级失败，程序、运行环境与安装状态均已恢复到上一个版本。'
        } else {
            Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
                -Message '升级失败且未完全恢复，请重新运行 install.bat。'
        }
        Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
            -Message ("升级失败 {0}：{1}" -f $info.Code, $info.Message)
        return @{ Ok = $false; ErrorCode = $info.Code; Message = $info.Message
            RuntimeUpdated = $runtimeUpdated; Version = '' }
    } finally {
        Remove-Item -Path $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
        # A rollback copy is only discarded once something is in its place again;
        # otherwise it is the only remaining copy of the installed version.
        if ((Test-Path $Layout.App) -and (Test-Path $rollbackApp)) {
            Remove-Item -Path $rollbackApp -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ((Test-Path $Layout.Runtime) -and (Test-Path $rollbackRuntime)) {
            Remove-Item -Path $rollbackRuntime -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# ── start ───────────────────────────────────────────────────────────────────

function Get-StudioInstanceRecord {
    param($Layout, $Probe = $null)
    $path = Join-Path $Layout.Logs 'studio.json'
    $record = Read-InstallState -StateFile $path
    if ($null -eq $record) { return @{ Alive = $false; Path = $path; Record = $null } }
    $alive = Invoke-DeliveryProbe -Probe $Probe -Name 'ProcessAlive' `
        -Fallback { param($ProcessId, $StartedAt) Test-ProcessIdentity -ProcessId $ProcessId -StartedAt $StartedAt } `
        -Arguments @([int]$record.pid, [string]$record.started_at)
    return @{ Alive = [bool]$alive; Path = $path; Record = $record }
}

function Stop-StudioProcess {
    # Only ever the process the record names, and only after its identity was
    # re-checked: an unrelated process that happens to reuse the PID is left
    # alone, and so is an instance that was already healthy before the launch.
    param(
        [int]$ProcessId,
        [string]$StartedAt = '',
        [string]$LogFile = $null,
        [string]$InstallRoot = ''
    )
    if ($ProcessId -le 0) { return $false }
    if (-not (Test-ProcessIdentity -ProcessId $ProcessId -StartedAt $StartedAt)) {
        Write-DeliveryLog -LogFile $LogFile -Level 'WARN' -InstallRoot $InstallRoot `
            -Message '待终止的进程身份与记录不一致，已跳过终止。'
        return $false
    }
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    } catch {
        Write-DeliveryLog -LogFile $LogFile -Level 'WARN' -InstallRoot $InstallRoot `
            -Message '无法终止该服务进程，请手动结束该进程。'
        return $false
    }
    return $true
}

function Start-StudioProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = '',
        $Environment = $null,
        [string]$LogFile = $null
    )
    $stdOut = Join-Path (Split-Path -Path $LogFile -Parent) 'studio.out.log'
    $stdErr = Join-Path (Split-Path -Path $LogFile -Parent) 'studio.err.log'
    $previous = Set-ProcessEnvironment -Environment $Environment
    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
            -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr
    } finally {
        Restore-ProcessEnvironment -Previous $previous
    }
    Start-Sleep -Milliseconds 300
    if ($process.HasExited) {
        throw (New-DeliveryFailure -Code 'SERVICE_EXITED' `
                -Message 'Studio 服务进程启动后立即退出，请查看 logs\studio.err.log。')
    }
    return @{ Pid = $process.Id; StartedAt = $process.StartTime.ToUniversalTime().ToString('o') }
}

function Start-Studio {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [string]$EnvPort = $null,
        $Probe = $null,
        $Runner = $null,
        $PreflightRunner = $null,
        $Launcher = $null,
        $HealthProbe = $null,
        $BrowserOpener = $null,
        $ProcessStopper = $null,
        [int]$SleepMs = 1000,
        [int]$TimeoutSeconds = 180
    )
    $logFile = Join-Path $Layout.Logs 'start.log'
    try {
        if (-not (Test-Path $Layout.Logs)) { New-Item -ItemType Directory -Force -Path $Layout.Logs | Out-Null }
        $state = Read-InstallState -StateFile $Layout.StateFile
        Test-InstallStateUsable -Layout $Layout -State $state | Out-Null
        $port = Get-StudioPort -Layout $Layout -State $state -EnvPort $EnvPort

        $instance = Get-StudioInstanceRecord -Layout $Layout -Probe $Probe
        if ($instance.Alive) {
            $url = "http://127.0.0.1:{0}/" -f $port
            # The record is a claim about a process, not evidence of a product:
            # the endpoint itself is asked before an existing instance is called
            # running. A studio.json whose process was killed, crashed its worker
            # or lost the port would otherwise be reported as healthy forever.
            $aliveHealthy = Invoke-HealthProbe -HealthProbe $HealthProbe -Port $port
            if ($aliveHealthy) {
                Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message 'Studio 已在运行，只打开浏览器。'
                Open-StudioBrowser -BrowserOpener $BrowserOpener -Url $url | Out-Null
                return @{ Ok = $true; AlreadyRunning = $true; ErrorCode = $null; Message = 'Studio 已在运行。'
                    Pid = [int]$instance.Record.pid; InstanceFile = $instance.Path; Port = $port; Url = $url
                    Healthy = $true }
            }
            # Alive but silent: the instance is repaired, never reported as
            # running. The PID and started_at are re-checked by the stop itself,
            # so a pid the system reused for an unrelated process is left alone;
            # the stale record is withdrawn either way and the start continues.
            Write-DeliveryLog -LogFile $logFile -Level 'WARN' -InstallRoot $Layout.Root `
                -Message 'STALE_INSTANCE_DETECTED 记录的 Studio 进程仍存在但健康检查未通过，准备清理后重新启动。'
            $stalePid = [int]$instance.Record.pid
            $staleStartedAt = [string]$instance.Record.started_at
            $stopped = $false
            if ($null -ne $ProcessStopper) {
                $stopped = [bool](& $ProcessStopper -ProcessId $stalePid -StartedAt $staleStartedAt `
                        -LogFile $logFile -InstallRoot $Layout.Root)
            } else {
                $stopped = Stop-StudioProcess -ProcessId $stalePid -StartedAt $staleStartedAt `
                    -LogFile $logFile -InstallRoot $Layout.Root
            }
            if (-not $stopped) {
                # A second server on the same port would be worse than a start
                # that refuses: the stale process holds the identity the record
                # names and it could not be ended.
                Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
                    -Message 'STALE_INSTANCE_UNSTOPPABLE 无法安全停止失效的 Studio 进程，本次未启动第二个实例。'
                return @{ Ok = $false; AlreadyRunning = $false; ErrorCode = 'STALE_INSTANCE_UNSTOPPABLE'
                    Message = '检测到失效的 Studio 进程，且无法安全停止该进程；为避免启动第二个实例，本次启动已取消。请查看 logs\studio.err.log。'
                    Pid = $stalePid; InstanceFile = $instance.Path; Port = $port; Url = ''
                    Healthy = $false; Stopped = $false }
            }
            Remove-Item -Path $instance.Path -Force -ErrorAction SilentlyContinue
            Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root `
                -Message 'STALE_INSTANCE_CLEARED 失效的 Studio 实例已清理，继续正常启动流程。'
        }

        $occupied = Invoke-DeliveryProbe -Probe $Probe -Name 'PortInUse' `
            -Fallback { param($Port) Test-TcpPortInUse -Port $Port } -Arguments @($port)
        if ($occupied) {
            return @{ Ok = $false; AlreadyRunning = $false; ErrorCode = 'PORT_IN_USE'
                Message = ("端口 {0} 已被其它程序占用，请关闭占用该端口的程序后重试。" -f $port)
                Pid = 0; InstanceFile = $instance.Path; Port = $port; Url = '' }
        }

        $preflight = Invoke-Preflight -Layout $Layout -Arguments @('--require-gpu') `
            -Runner $PreflightRunner -Port $port -LogFile $logFile
        $exitCode = 0
        if ($null -ne $preflight -and $null -ne $preflight.ExitCode) { $exitCode = [int]$preflight.ExitCode }
        if ($exitCode -ne 0) {
            return @{ Ok = $false; AlreadyRunning = $false; ErrorCode = 'PREFLIGHT_FAILED'
                Message = '共享交付预检失败：未检测到可用的 NVIDIA GPU。Auto-Tune Studio 只提供 GPU 训练，不提供 CPU 回退。'
                Pid = 0; InstanceFile = $instance.Path; Port = $port; Url = '' }
        }

        $environment = New-StudioEnvironment -Layout $Layout -Port $port
        $pythonDirectory = Split-Path -Path $Layout.Interpreter -Parent
        $environment['PATH'] = (@($pythonDirectory,
                (Join-Path $pythonDirectory 'Scripts'),
                $env:PATH) | Where-Object { $_ }) -join ';'
        $arguments = @('-m', 'auto_tune.main')
        if ($null -ne $Launcher) {
            $launched = & $Launcher -FilePath $Layout.Interpreter -Arguments $arguments `
                -WorkingDirectory $Layout.Data -Environment $environment -LogFile $logFile
        } else {
            $launched = Start-StudioProcess -FilePath $Layout.Interpreter -Arguments $arguments `
                -WorkingDirectory $Layout.Data -Environment $environment -LogFile $logFile
        }

        Write-InstallState -StateFile $instance.Path -State ([ordered]@{
                pid        = [int]$launched.Pid
                started_at = [string]$launched.StartedAt
                port       = $port
            }) | Out-Null

        $url = "http://127.0.0.1:{0}/" -f $port
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $healthy = $false
        while ((Get-Date) -lt $deadline) {
            $healthy = Invoke-HealthProbe -HealthProbe $HealthProbe -Port $port
            if ($healthy) { break }
            Start-Sleep -Milliseconds ([math]::Max(1, $SleepMs))
        }
        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message ("健康检查结果：{0}" -f $healthy)

        if (-not $healthy) {
            # A process that never answers is not a running product. The launch
            # is undone the way it was made: the process this call started is
            # terminated (after re-checking its identity) and the instance record
            # it wrote is withdrawn, so the next start does not trust a dead
            # service. No browser is opened.
            $stopped = $false
            if ($null -ne $ProcessStopper) {
                $stopped = [bool](& $ProcessStopper -ProcessId ([int]$launched.Pid) `
                        -StartedAt ([string]$launched.StartedAt) -LogFile $logFile -InstallRoot $Layout.Root)
            } else {
                $stopped = Stop-StudioProcess -ProcessId ([int]$launched.Pid) `
                    -StartedAt ([string]$launched.StartedAt) -LogFile $logFile -InstallRoot $Layout.Root
            }
            Remove-Item -Path $instance.Path -Force -ErrorAction SilentlyContinue
            # A fixed, redacted line: the pid and the captured output stay out of
            # the log, the stable code is the only thing an operator needs.
            Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
                -Message 'HEALTH_CHECK_FAILED 健康检查未在超时内通过，本次启动已终止并清理实例记录。'
            return @{ Ok = $false; AlreadyRunning = $false; ErrorCode = 'HEALTH_CHECK_FAILED'
                Message = '服务进程已启动，但健康检查未在超时内通过，本次启动已终止。请查看 logs\studio.err.log。'
                Pid = 0; InstanceFile = $instance.Path; Port = $port; Url = ''
                Healthy = $false; Stopped = $stopped }
        }

        Open-StudioBrowser -BrowserOpener $BrowserOpener -Url $url | Out-Null

        return @{ Ok = $true; AlreadyRunning = $false; ErrorCode = $null
            Message = 'Studio 已启动。'
            Pid = [int]$launched.Pid; InstanceFile = $instance.Path; Port = $port; Url = $url
            Healthy = $true }
    } catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
            -Message ("启动失败 {0}：{1}" -f $info.Code, $info.Message)
        return @{ Ok = $false; AlreadyRunning = $false; ErrorCode = $info.Code; Message = $info.Message
            Pid = 0; InstanceFile = (Join-Path $Layout.Logs 'studio.json'); Port = 0; Url = ''; Healthy = $false }
    }
}

function Open-StudioBrowser {
    param($BrowserOpener, [string]$Url)
    if ($null -ne $BrowserOpener) { return & $BrowserOpener -Url $Url }
    return Open-DefaultBrowser -Url $Url
}

function Invoke-HealthProbe {
    param($HealthProbe, [int]$Port, [int]$TimeoutMs = 2000)
    if ($null -ne $HealthProbe) { return & $HealthProbe -Port $Port -TimeoutMs $TimeoutMs }
    return Test-HealthEndpoint -Port $Port -TimeoutMs $TimeoutMs
}

# ── uninstall ───────────────────────────────────────────────────────────────

function Assert-SafeRemovalTarget {
    param([Parameter(Mandatory = $true)]$Layout, [string]$Target)

    if ([string]::IsNullOrWhiteSpace($Target)) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '删除目标为空，已拒绝执行。')
    }
    # Canonicalised first: a short spelling of the target and a long spelling of
    # the installation (or the other way round) must still be recognised as the
    # same place, or a legitimate removal inside the installation is refused.
    $full = Get-CanonicalPath -Path $Target
    $root = Get-CanonicalPath -Path $Layout.Root
    $trimmed = $full.TrimEnd('\', '/')
    $rootTrimmed = $root.TrimEnd('\', '/')

    if ($trimmed -eq [System.IO.Path]::GetPathRoot($full).TrimEnd('\', '/')) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '拒绝对盘符根目录执行删除。')
    }
    if ($trimmed -eq $rootTrimmed) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '拒绝对安装根目录本身执行删除。')
    }
    $parent = [System.IO.Path]::GetDirectoryName($rootTrimmed)
    if ($parent -and $parent.TrimEnd('\', '/') -eq [System.IO.Path]::GetPathRoot($root).TrimEnd('\', '/')) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '安装目录过于接近盘符根目录，已拒绝执行删除。')
    }
    $prefix = $rootTrimmed + '\'
    if (-not $full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' `
                -Message '删除目标不在 AutoTuneStudio 安装目录之内，已拒绝执行。')
    }
    foreach ($dangerous in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA, $env:SystemRoot, $env:ProgramFiles)) {
        if ([string]::IsNullOrWhiteSpace($dangerous)) { continue }
        $candidate = [System.IO.Path]::GetFullPath($dangerous).TrimEnd('\', '/')
        if ($trimmed -eq $candidate) {
            throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '删除目标是系统或用户目录根，已拒绝执行。')
        }
    }
    foreach ($marker in @('.git', 'AGENTS.md', 'CLAUDE.md', 'package.json')) {
        if (Test-Path (Join-Path $full $marker)) {
            throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' `
                    -Message '删除目标看起来是源码仓库根目录，已拒绝执行。')
        }
    }
    return $full
}

function Remove-DeliveryTree {
    param([string]$Path, $Remover = $null)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw (New-DeliveryFailure -Code 'UNSAFE_REMOVAL_TARGET' -Message '删除目标为空，已拒绝执行。')
    }
    if (-not (Test-Path $Path)) { return $false }
    if ($null -ne $Remover) {
        & $Remover -Path $Path
        return $true
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
    return $true
}

function Start-DeferredDeletion {
    # cmd.exe re-reads a running .bat as it goes and PowerShell may still hold
    # the entry script: deleting the uninstall entry from inside the running
    # removal is how a half uninstall happens. A detached helper waits for this
    # process to end and only then removes the named paths.
    param(
        [string[]]$Paths,
        [int]$DelaySeconds = 3,
        $Deferrer = $null
    )
    $targets = @($Paths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if (@($targets).Count -eq 0) { return @() }
    if ($DelaySeconds -lt 1) { $DelaySeconds = 1 }
    if ($null -ne $Deferrer) {
        & $Deferrer -Paths $targets -DelaySeconds $DelaySeconds | Out-Null
        return $targets
    }

    $commands = New-Object System.Collections.ArrayList
    foreach ($path in $targets) {
        if (Test-Path -PathType Container $path) {
            [void]$commands.Add(('rd /s /q "{0}"' -f $path))
        } else {
            [void]$commands.Add(('del /f /q "{0}"' -f $path))
        }
    }
    $waiter = 'ping -n {0} 127.0.0.1' -f ($DelaySeconds + 1)
    $script = $waiter + ' > nul & ' + (@($commands) -join ' & ')
    $shell = $env:ComSpec
    if ([string]::IsNullOrWhiteSpace($shell)) { $shell = 'cmd.exe' }
    try {
        Start-Process -FilePath $shell -ArgumentList @('/c', $script) -WindowStyle Hidden | Out-Null
    } catch {
        return @()
    }
    return $targets
}

function Invoke-Uninstall {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        $State = $null,
        [switch]$RemoveData,
        [switch]$Confirm,
        $Probe = $null,
        $Remover = $null,
        $Deferrer = $null,
        [int]$DeferSeconds = 3
    )
    $logFile = Join-Path $Layout.Logs 'uninstall.log'
    try {
        if (-not (Test-Path $Layout.Logs)) { New-Item -ItemType Directory -Force -Path $Layout.Logs | Out-Null }
        if ($null -eq $State) { $State = Read-InstallState -StateFile $Layout.StateFile }

        $instance = Get-StudioInstanceRecord -Layout $Layout -Probe $Probe
        if ($instance.Alive) {
            return @{ Ok = $false; ErrorCode = 'UNINSTALL_BLOCKED_RUNNING'
                Message = 'Studio 仍在运行，请先关闭服务窗口（或结束 python 进程）后重试卸载。'
                DataRemoved = $false; DataNoteFile = $Layout.DataNote }
        }

        if ($RemoveData -and -not $Confirm) {
            return @{ Ok = $false; ErrorCode = 'CONFIRMATION_REQUIRED'
                Message = '删除用户数据需要显式二次确认：请使用 uninstall.bat --remove-data --confirm。'
                DataRemoved = $false; DataNoteFile = $Layout.DataNote }
        }

        $dataTarget = $null
        if ($RemoveData) {
            $dataTarget = Assert-SafeRemovalTarget -Layout $Layout -Target $Layout.Data
        }

        [void](Remove-DeliveryTree -Path $Layout.App -Remover $Remover)
        [void](Remove-DeliveryTree -Path $Layout.Runtime -Remover $Remover)
        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message '已删除程序与私有运行环境。'

        # The launcher is part of the installation and goes with it — except for
        # the two files the running uninstall is being read from. The module is
        # already in memory and the shortcut is an ordinary file, so both can go
        # immediately; the entries are withdrawn by the deferred helper below.
        [void](Remove-DesktopShortcut -Path $Layout.Shortcut -Target $Layout.StartBat)
        [void](Remove-InstallLocation -Layout $Layout)
        [void](Remove-DeliveryTree -Path $Layout.Launcher -Remover $Remover)
        [void](Remove-DeliveryTree -Path $Layout.StartBat -Remover $Remover)
        [void](Remove-DeliveryTree -Path $Layout.StartScript -Remover $Remover)
        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message '已删除启动入口与桌面快捷方式。'

        $dataRemoved = $false
        $message = ''
        if ($RemoveData) {
            [void](Remove-DeliveryTree -Path $dataTarget -Remover $Remover)
            $dataRemoved = $true
            $message = ("用户数据已按要求删除：{0}" -f $dataTarget)
        } else {
            $message = ("用户数据已保留（配置、历史、SQLite、权重与训练记录）：{0}" -f $Layout.Data)
        }

        $note = @(
            'Auto-Tune Studio 已卸载。',
            ("程序与私有运行环境：已删除"),
            ("启动入口与桌面快捷方式：已删除"),
            ("安装缓存（保留，重新安装可复用）：{0}" -f $Layout.Cache),
            ("用户数据：{0}" -f $(if ($dataRemoved) { '已删除' } else { '已保留' })),
            ("数据目录：{0}" -f $Layout.Data),
            ("卸载时间：{0}" -f (Get-Date).ToUniversalTime().ToString('o'))
        ) -join "`r`n"
        [System.IO.File]::WriteAllText($Layout.DataNote, $note,
            (New-Object System.Text.UTF8Encoding($false)))

        Remove-Item -Path $Layout.StateFile -Force -ErrorAction SilentlyContinue

        # Only a removal that went all the way through withdraws its own entry
        # point: a failed uninstall must stay runnable so it can be retried.
        $deferred = Start-DeferredDeletion -DelaySeconds $DeferSeconds -Deferrer $Deferrer -Paths @(
            $Layout.UninstallBat, $Layout.UninstallScript)
        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root `
            -Message ("已安排删除卸载入口 {0} 个。" -f @($deferred).Count)
        Write-DeliveryLog -LogFile $logFile -InstallRoot $Layout.Root -Message $message

        return @{ Ok = $true; ErrorCode = $null; Message = $message
            DataRemoved = $dataRemoved; DataNoteFile = $Layout.DataNote
            DataPath = $Layout.Data; CachePath = $Layout.Cache
            DeferredPaths = @($deferred) }
    } catch {
        $info = Get-DeliveryErrorInfo -ErrorRecord $_
        Write-DeliveryLog -LogFile $logFile -Level 'ERROR' -InstallRoot $Layout.Root `
            -Message ("卸载失败 {0}：{1}" -f $info.Code, $info.Message)
        return @{ Ok = $false; ErrorCode = $info.Code; Message = $info.Message
            DataRemoved = $false; DataNoteFile = $Layout.DataNote }
    }
}

Export-ModuleMember -Function @(
    'New-DeliveryFailure', 'Get-DeliveryErrorInfo', 'ConvertTo-SafeLogLine', 'Write-DeliveryLog',
    'Invoke-DeliveryProbe', 'Get-CanonicalPath', 'Get-RelativePathUnderRoot',
    'Test-Is64BitWindows', 'Get-FreeDiskBytes', 'Test-NvidiaDriverPresent',
    'Test-ProcessIdentity', 'Test-TcpPortInUse', 'Test-HealthEndpoint', 'Open-DefaultBrowser',
    'Get-DeliveryLayout', 'Get-UserDesktopPath', 'Get-LayoutDataDirectories',
    'Initialize-DeliveryLayout', 'Get-LauncherEntryNames', 'Get-LauncherModuleRelative',
    'Get-LauncherRelativePaths',
    'Assert-DirectoryWritable', 'Test-DirectoryWritable', 'Assert-InstallPreconditions',
    'Assert-SufficientFreeBytes',
    'Get-PackageManifest', 'Assert-PackageManifest', 'Assert-TrustedDownloadUrl',
    'Get-FileSha256', 'Get-FileSha256OrEmpty',
    'Test-FileHashMatches', 'Test-PayloadPath', 'Test-GlobMatch', 'Test-ScriptPath',
    'Test-OfflineBundlePath',
    'Get-PackageFileList', 'New-PackageLock',
    'Get-PackageLock', 'Test-PackageIntegrity', 'Assert-PackageIntegrity', 'Copy-PackagePayload',
    'Write-JsonFile', 'Get-NormalizedPackageName',
    'Get-OfflineLayout', 'Get-OfflineCoreExpectations', 'New-OfflineLock', 'Get-OfflineLock',
    'Test-OfflineWheelFileName', 'Test-OfflineLockPath', 'Get-MissingOfflineRequirements',
    'Test-PipRequirementSpecifier', 'Get-MissingPipRequirement',
    'Test-OfflineBundle', 'Assert-OfflineBundle', 'Prepare-OfflineBundle',
    'Assert-InstallRootAllowed', 'Get-RecommendedInstallRoot', 'Get-InstallPromptPlan',
    'Get-SystemDriveRoot',
    'Test-InstallRootOnSystemDrive', 'Get-InstallDestinationAdvice',
    'Get-InstallLocationPath', 'Get-InstalledRootRecord', 'Write-InstallLocation',
    'Remove-InstallLocation', 'Resolve-DeliveryInstallRoot',
    'Get-ConfigTemplatePath', 'Initialize-UserConfig',
    'New-DesktopShortcut', 'Get-DesktopShortcutTarget', 'Remove-DesktopShortcut',
    'Install-DeliveryLauncher', 'Test-LauncherInstalled',
    'Read-InstallState', 'Write-InstallState', 'New-InstallState',
    'Test-InstallStateUsable', 'Test-InstallCoreUsable',
    'Get-StudioPort', 'Set-ProcessEnvironment', 'Restore-ProcessEnvironment',
    'Invoke-ExternalProcess', 'Invoke-RunnerChecked', 'Install-PrivateRuntime',
    'Get-RuntimeKey', 'Get-RuntimeStampPath', 'Read-RuntimeStamp', 'Write-RuntimeStamp',
    'Ensure-PrivateRuntime', 'Invoke-Preflight', 'New-StudioEnvironment', 'Get-RequiredFreeBytes',
    'Stage-Payload', 'Install-StagedPayload', 'Invoke-Install', 'Invoke-Upgrade',
    'New-StagedLayout', 'Invoke-PreflightExitCode',
    'Get-StudioInstanceRecord', 'Start-StudioProcess', 'Stop-StudioProcess', 'Start-Studio',
    'Open-StudioBrowser', 'Invoke-HealthProbe',
    'Assert-SafeRemovalTarget', 'Remove-DeliveryTree', 'Start-DeferredDeletion', 'Invoke-Uninstall'
)
