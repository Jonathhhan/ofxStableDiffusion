param(
    [string]$SourceDir = "",
    [string]$BuildDir = "",
    [string]$InstallIncludeDir = "",
    [string]$InstallLibDir = "",
    [string]$InstallGgmlIncludeDir = "",
    [string]$InstallGgmlLibDir = "",
    [string]$VariantRootDir = "",
    [string]$ExampleBinDir = "",
    [string]$InstallBinDir = "",
    [string]$Configuration = "Release",
    [string]$SourceReleaseTag = "",
    [string]$Generator = "",
    [int]$Jobs = 0,
    [switch]$Clean,
    [string]$GgmlReleaseTag = "",
    [Alias('Cpu')][switch]$CpuOnly,
    [Alias('Gpu')][switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Metal,
    [switch]$All,
    [switch]$BuildCli,
    [switch]$SkipSourceRefresh,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$DefaultSourceReleaseTag = "master-572-1b4e9be"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Test-IsWindowsHost {
    return $env:OS -eq 'Windows_NT'
}

function Test-IsMacHost {
    return [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::MacOSX
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-CommandPathOrNull {
    param([string]$Name)
    try {
        return (Get-Command $Name -ErrorAction Stop).Source
    } catch {
        return $null
    }
}

function Test-CudaAvailable {
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath $env:CUDA_PATH)) {
        return $true
    }
    return [bool](Get-CommandPathOrNull 'nvcc.exe')
}

function Test-VulkanAvailable {
    if ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK)) {
        return $true
    }
    return [bool](Get-CommandPathOrNull 'glslc.exe') -or
        [bool](Get-CommandPathOrNull 'vulkaninfo.exe')
}

function Test-MetalAvailable {
    if (-not (Test-IsMacHost)) {
        return $false
    }
    return [bool](Get-CommandPathOrNull 'xcrun')
}

function Invoke-SelfBuild {
    param(
        [string]$BackendName,
        [string]$ScriptPath,
        [string]$SourceDir,
        [string]$BuildDir,
        [string]$InstallIncludeDir,
        [string]$InstallLibDir,
        [string]$InstallGgmlIncludeDir,
        [string]$InstallGgmlLibDir,
        [string]$VariantRootDir,
        [string]$ExampleBinDir,
        [string]$InstallBinDir,
        [string]$Configuration,
        [string]$SourceReleaseTag,
        [string]$Generator,
        [int]$Jobs,
        [string]$GgmlReleaseTag,
        [switch]$Clean,
        [switch]$BuildCli,
        [switch]$DryRun,
        [switch]$SkipSourceRefresh
    )

    $invokeArgs = @{
        SourceDir = $SourceDir
        BuildDir = $BuildDir
        InstallIncludeDir = $InstallIncludeDir
        InstallLibDir = $InstallLibDir
        InstallGgmlIncludeDir = $InstallGgmlIncludeDir
        InstallGgmlLibDir = $InstallGgmlLibDir
        VariantRootDir = $VariantRootDir
        ExampleBinDir = $ExampleBinDir
        InstallBinDir = $InstallBinDir
        Configuration = $Configuration
        Generator = $Generator
        Jobs = $Jobs
        GgmlReleaseTag = $GgmlReleaseTag
    }

    if (-not [string]::IsNullOrWhiteSpace($SourceReleaseTag)) {
        $invokeArgs.SourceReleaseTag = $SourceReleaseTag
    }
    if ($Clean) {
        $invokeArgs.Clean = $true
    }
    if ($DryRun) {
        $invokeArgs.DryRun = $true
    }
    if ($BuildCli) {
        $invokeArgs.BuildCli = $true
    }
    if ($SkipSourceRefresh) {
        $invokeArgs.SkipSourceRefresh = $true
    }

    switch ($BackendName) {
        'cpu-only' { $invokeArgs.CpuOnly = $true }
        'cuda' { $invokeArgs.Cuda = $true }
        'vulkan' { $invokeArgs.Vulkan = $true }
        'metal' { $invokeArgs.Metal = $true }
        default { throw "Unknown backend '$BackendName'." }
    }

    & $ScriptPath @invokeArgs
}

function Get-ShortBuildDir {
    param(
        [string]$AddonRoot,
        [switch]$DryRun
    )

    $hashBytes = [System.Security.Cryptography.SHA1]::Create().ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($AddonRoot))
    $hash = [System.BitConverter]::ToString($hashBytes).Replace('-', '').Substring(0, 10).ToLowerInvariant()
    $candidateRoots = @()

    if (-not [string]::IsNullOrWhiteSpace($env:OFXSD_SHORT_BUILD_ROOT)) {
        $candidateRoots += $env:OFXSD_SHORT_BUILD_ROOT
    }
    if (-not [string]::IsNullOrWhiteSpace($env:PUBLIC)) {
        $candidateRoots += (Join-Path $env:PUBLIC 'sd')
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidateRoots += (Join-Path $env:LOCALAPPDATA 'sd')
    }

    $tempRoot = [System.IO.Path]::GetTempPath()
    if (-not [string]::IsNullOrWhiteSpace($tempRoot)) {
        $candidateRoots += (Join-Path $tempRoot 'sd')
    }

    foreach ($root in ($candidateRoots | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        $candidate = Join-Path $root $hash
        if ($DryRun) {
            return $candidate
        }

        try {
            New-Item -ItemType Directory -Force -Path $root | Out-Null
            return $candidate
        } catch {
            continue
        }
    }

    return Join-Path $tempRoot "ofxsd-build-$hash"
}

function Find-FirstFile {
    param(
        [string]$Root,
        [string[]]$Names
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        return $null
    }

    foreach ($name in $Names) {
        $match = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter $name -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    return $null
}

function Copy-IfPresent {
    param(
        [string]$Path,
        [string]$Destination,
        [switch]$AllowLockedDestination
    )

    if (-not $Path) {
        return
    }

    if ($DryRun) {
        Write-Host "Copy $Path -> $Destination"
        return
    }

    try {
        Copy-Item -LiteralPath $Path -Destination $Destination -Force
    } catch [System.IO.IOException] {
        if ($AllowLockedDestination) {
            Write-Warning "Skipping copy into $Destination because the destination file is in use. Close the running example app and re-run setup if you want the latest runtime staged there."
            return
        }
        throw
    }
}

function Get-VendoredValue {
    param(
        [string]$Path,
        [string]$Prefix
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $line = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue |
        Where-Object { $_ -like "$Prefix*" } |
        Select-Object -First 1
    if (-not $line) {
        return $null
    }

    return $line.Substring($Prefix.Length).Trim()
}

function Invoke-GitHubJsonRequest {
    param([string]$Uri)

    $headers = @{
        'Accept' = 'application/vnd.github+json'
        'User-Agent' = 'ofxStableDiffusion-build'
        'X-GitHub-Api-Version' = '2022-11-28'
    }

    try {
        return Invoke-RestMethod -Uri $Uri -Headers $headers
    } catch {
        throw "GitHub API request failed for $Uri. $($_.Exception.Message)"
    }
}

function Require-GitPath {
    $git = Get-CommandPathOrNull 'git.exe'
    if (-not $git) {
        $git = Get-CommandPathOrNull 'git'
    }
    if (-not $git) {
        throw "git was not found in PATH. A recursive clone is required so the latest release-tag source snapshot includes ggml/libwebp/libwebm submodules."
    }
    return $git
}

function Get-ReleaseMetadata {
    param([string]$Tag)

    $repoApiBase = 'https://api.github.com/repos/leejet/stable-diffusion.cpp/releases'
    if ([string]::IsNullOrWhiteSpace($Tag)) {
        return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/latest')
    }

    $escapedTag = [System.Uri]::EscapeDataString($Tag)
    return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/tags/' + $escapedTag)
}

function Get-GgmlReleaseMetadata {
    param([string]$Tag)

    $repoApiBase = 'https://api.github.com/repos/ggml-org/ggml/releases'
    if ([string]::IsNullOrWhiteSpace($Tag)) {
        return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/latest')
    }

    $escapedTag = [System.Uri]::EscapeDataString($Tag)
    return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/tags/' + $escapedTag)
}

function Remove-DirectoryContents {
    param([string]$LiteralPath)

    if (-not (Test-Path -LiteralPath $LiteralPath)) {
        return
    }

    Get-ChildItem -LiteralPath $LiteralPath -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
        }
}

function Write-VendorPinFile {
    param(
        [string]$Path,
        [object]$ReleaseMetadata,
        [string]$Repository = "",
        [string]$ResolvedCommit = "",
        [string]$Notes = ""
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('Vendored by ofxStableDiffusion')
    if (-not [string]::IsNullOrWhiteSpace($Repository)) {
        $lines.Add('Upstream repository: ' + $Repository)
    }
    $lines.Add('Upstream release tag: ' + $ReleaseMetadata.tag_name)
    if (-not [string]::IsNullOrWhiteSpace($ResolvedCommit)) {
        $lines.Add('Upstream commit: ' + $ResolvedCommit)
    }
    if (-not [string]::IsNullOrWhiteSpace($ReleaseMetadata.target_commitish)) {
        $lines.Add('Upstream target commitish: ' + $ReleaseMetadata.target_commitish)
    }
    $lines.Add('Release URL: ' + $ReleaseMetadata.html_url)
    $lines.Add('Zipball URL: ' + $ReleaseMetadata.zipball_url)
    $lines.Add('Fetched at: ' + [DateTimeOffset]::UtcNow.ToString('o'))
    if (-not [string]::IsNullOrWhiteSpace($Notes)) {
        $lines.Add('Notes: ' + $Notes)
    }

    [System.IO.File]::WriteAllLines($Path, $lines)
}

function Get-GitHeadCommit {
    param(
        [string]$GitPath,
        [string]$RepositoryRoot
    )

    $commit = (& $GitPath -C $RepositoryRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
        throw "Failed to resolve HEAD commit for '$RepositoryRoot'."
    }

    return $commit
}

function Refresh-GgmlVendorTree {
    param(
        [string]$GitPath,
        [string]$Tag,
        [string]$TargetDir,
        [switch]$DryRun
    )

    $releaseMetadata = Get-GgmlReleaseMetadata -Tag $Tag
    $resolvedReleaseTag = $releaseMetadata.tag_name
    $downloadRoot = Join-Path $env:TEMP 'ofxsd-ggml-release'
    $cloneRoot = Join-Path $downloadRoot ('clone-' + $resolvedReleaseTag)

    Write-Step "Refreshing ggml source from upstream release snapshot"
    Write-Host ("    Release tag: {0}" -f $resolvedReleaseTag)
    Write-Host ("    Destination: {0}" -f $TargetDir)

    if ($DryRun) {
        Write-Host ("Clone https://github.com/ggml-org/ggml.git tag {0}" -f $resolvedReleaseTag)
        Write-Host ("Clone to {0}" -f $cloneRoot)
        Write-Host ("Replace contents of {0}" -f $TargetDir)
        return
    }

    New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
    if (Test-Path -LiteralPath $cloneRoot) {
        Remove-Item -LiteralPath $cloneRoot -Recurse -Force
    }

    Invoke-External -FilePath $GitPath -Arguments @(
        'clone',
        '--depth', '1',
        '--branch', $resolvedReleaseTag,
        'https://github.com/ggml-org/ggml.git',
        $cloneRoot
    )

    $cmakeLists = Join-Path $cloneRoot 'CMakeLists.txt'
    if (-not (Test-Path -LiteralPath $cmakeLists)) {
        throw "The ggml clone for '$resolvedReleaseTag' did not contain a top-level CMakeLists.txt."
    }

    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Remove-DirectoryContents -LiteralPath $TargetDir

    Get-ChildItem -LiteralPath $cloneRoot -Force |
        ForEach-Object {
            if ($_.Name -eq '.git') {
                return
            }
            Copy-Item -LiteralPath $_.FullName -Destination $TargetDir -Recurse -Force
        }

    $resolvedCommit = Get-GitHeadCommit -GitPath $GitPath -RepositoryRoot $cloneRoot
    Write-VendorPinFile `
        -Path (Join-Path $TargetDir 'OFX_VENDOR_PIN.txt') `
        -ReleaseMetadata $releaseMetadata `
        -Repository 'https://github.com/ggml-org/ggml' `
        -ResolvedCommit $resolvedCommit `
        -Notes 'Source-only vendor refresh staged under stable-diffusion.cpp.'
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $addonRoot 'libs\stable-diffusion\source'
}
$buildDirWasImplicit = [string]::IsNullOrWhiteSpace($BuildDir)
if ([string]::IsNullOrWhiteSpace($InstallIncludeDir)) {
    $InstallIncludeDir = Join-Path $addonRoot 'libs\stable-diffusion\include'
}
if ([string]::IsNullOrWhiteSpace($InstallLibDir)) {
    $InstallLibDir = Join-Path $addonRoot 'libs\stable-diffusion\lib\vs'
}

function Copy-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        return
    }

    if ($DryRun) {
        Write-Host "Copy contents $Source -> $Destination"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
        }
}
if ([string]::IsNullOrWhiteSpace($InstallGgmlIncludeDir)) {
    $InstallGgmlIncludeDir = Join-Path $addonRoot 'libs\ggml\include'
}
if ([string]::IsNullOrWhiteSpace($InstallGgmlLibDir)) {
    $InstallGgmlLibDir = Join-Path $addonRoot 'libs\ggml\lib\vs'
}
if ([string]::IsNullOrWhiteSpace($VariantRootDir)) {
    $VariantRootDir = Join-Path $addonRoot 'libs\variants'
}
if ([string]::IsNullOrWhiteSpace($ExampleBinDir)) {
    $ExampleBinDir = Join-Path $addonRoot 'ofxStableDiffusionExample\bin'
}
if ([string]::IsNullOrWhiteSpace($InstallBinDir)) {
    $InstallBinDir = Join-Path $addonRoot 'libs\stable-diffusion\bin\vs'
}
if ($Jobs -le 0) {
    $Jobs = [Math]::Max(1, [Environment]::ProcessorCount)
}

$selectedBackendCount = 0
if ($CpuOnly) { $selectedBackendCount++ }
if ($Cuda) { $selectedBackendCount++ }
if ($Vulkan) { $selectedBackendCount++ }
if ($Metal) { $selectedBackendCount++ }
if ($selectedBackendCount -gt 1) {
    throw "Select only one backend flag: -CpuOnly, -Cuda, -Vulkan, or -Metal."
}
if ($All -and $selectedBackendCount -gt 0) {
    throw "Do not combine -All with explicit backend flags."
}

if ($All) {
    $scriptPath = $MyInvocation.MyCommand.Path
    $allBackends = New-Object System.Collections.Generic.List[string]
    $allBackends.Add('cpu-only')

    if (Test-VulkanAvailable) {
        $allBackends.Add('vulkan')
    } else {
        Write-Warning "Skipping Vulkan in -All mode because no Vulkan SDK/runtime was detected."
    }

    if (Test-CudaAvailable) {
        $allBackends.Add('cuda')
    } else {
        Write-Warning "Skipping CUDA in -All mode because no CUDA toolkit/runtime was detected."
    }

    Write-Step "Building all available backend variants"
    Write-Host ("    Order: {0}" -f ($allBackends -join ' -> '))
    Write-Host "    Final canonical runtime priority: cuda > vulkan > cpu-only"

    $firstBuild = $true
    foreach ($backendName in $allBackends) {
        Invoke-SelfBuild `
            -BackendName $backendName `
            -ScriptPath $scriptPath `
            -SourceDir $SourceDir `
            -BuildDir $BuildDir `
            -InstallIncludeDir $InstallIncludeDir `
            -InstallLibDir $InstallLibDir `
            -InstallGgmlIncludeDir $InstallGgmlIncludeDir `
            -InstallGgmlLibDir $InstallGgmlLibDir `
            -VariantRootDir $VariantRootDir `
            -ExampleBinDir $ExampleBinDir `
            -InstallBinDir $InstallBinDir `
            -Configuration $Configuration `
            -SourceReleaseTag $SourceReleaseTag `
            -Generator $Generator `
            -Jobs $Jobs `
            -GgmlReleaseTag $GgmlReleaseTag `
            -Clean `
            -BuildCli:$BuildCli `
            -DryRun:$DryRun `
            -SkipSourceRefresh:$(-not $firstBuild)
        $firstBuild = $false
    }

    return
}

$enableCuda = $false
$enableVulkan = $false
$enableMetal = $false
$backendMode = "cpu-only"

if ($Cuda) {
    $enableCuda = $true
    $backendMode = "cuda"
} elseif ($Vulkan) {
    $enableVulkan = $true
    $backendMode = "vulkan"
} elseif ($Metal) {
    $enableMetal = $true
    $backendMode = "metal"
}

if ($buildDirWasImplicit) {
    if ((Test-IsWindowsHost) -and $enableVulkan) {
        $BuildDir = Get-ShortBuildDir -AddonRoot $addonRoot -DryRun:$DryRun
    } else {
        $BuildDir = Join-Path $addonRoot 'libs\stable-diffusion\build'
    }
}

$cmake = Get-CommandPathOrNull 'cmake.exe'
if (-not $cmake) {
    throw "cmake.exe was not found in PATH."
}

$effectiveSourceReleaseTag = if ([string]::IsNullOrWhiteSpace($SourceReleaseTag)) { $DefaultSourceReleaseTag } else { $SourceReleaseTag }
$releaseMetadata = Get-ReleaseMetadata -Tag $effectiveSourceReleaseTag
$resolvedReleaseTag = $releaseMetadata.tag_name
$downloadRoot = Join-Path $env:TEMP 'ofxsd-source-release'
$cloneRoot = Join-Path $downloadRoot ('clone-' + $resolvedReleaseTag)
$git = Require-GitPath

if (-not $SkipSourceRefresh) {
    Write-Step "Refreshing stable-diffusion source from upstream snapshot"
    Write-Host ("    Release tag: {0}" -f $resolvedReleaseTag)
    Write-Host ("    Destination: {0}" -f $SourceDir)

    if ($DryRun) {
        Write-Host ("Clone https://github.com/leejet/stable-diffusion.cpp.git tag {0} with submodules" -f $resolvedReleaseTag)
        Write-Host ("Clone to {0}" -f $cloneRoot)
        Write-Host ("Replace contents of {0}" -f $SourceDir)
    } else {
        New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
        if (Test-Path -LiteralPath $cloneRoot) {
            Remove-Item -LiteralPath $cloneRoot -Recurse -Force
        }

        Invoke-External -FilePath $git -Arguments @(
            'clone',
            '--depth', '1',
            '--branch', $resolvedReleaseTag,
            '--recurse-submodules',
            '--shallow-submodules',
            'https://github.com/leejet/stable-diffusion.cpp.git',
            $cloneRoot
        )

        $cmakeLists = Join-Path $cloneRoot 'CMakeLists.txt'
        if (-not (Test-Path -LiteralPath $cmakeLists)) {
            throw "The recursive clone for '$resolvedReleaseTag' did not contain a top-level CMakeLists.txt."
        }

        New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
        Remove-DirectoryContents -LiteralPath $SourceDir

        Get-ChildItem -LiteralPath $cloneRoot -Force |
            ForEach-Object {
                if ($_.Name -eq '.git') {
                    return
                }
                Copy-Item -LiteralPath $_.FullName -Destination $SourceDir -Recurse -Force
            }

        $resolvedCommit = Get-GitHeadCommit -GitPath $git -RepositoryRoot $cloneRoot
        Write-VendorPinFile `
            -Path (Join-Path $SourceDir 'OFX_VENDOR_PIN.txt') `
            -ReleaseMetadata $releaseMetadata `
            -Repository 'https://github.com/leejet/stable-diffusion.cpp' `
            -ResolvedCommit $resolvedCommit
    }

    Refresh-GgmlVendorTree -GitPath $git -Tag $GgmlReleaseTag -TargetDir (Join-Path $SourceDir 'ggml') -DryRun:$DryRun
} else {
    Write-Step "Using existing vendored stable-diffusion source snapshot"
    Write-Host ("    Source dir: {0}" -f $SourceDir)
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw @"
stable-diffusion.cpp source was not found at:
  $SourceDir

Recommended workflow:
  1. Re-run scripts/build-stable-diffusion.ps1 so it can refresh the latest upstream source snapshot
  2. Re-run scripts/build-stable-diffusion.ps1

This addon intentionally keeps stable-diffusion.cpp standalone rather than sharing
the ggml build from ofxGgml, to avoid ABI/version coupling across addons.
"@
}

$sourceCmakeLists = Join-Path $SourceDir 'CMakeLists.txt'
if (-not (Test-Path -LiteralPath $sourceCmakeLists)) {
    throw "No CMakeLists.txt was found in $SourceDir. Vendor the full stable-diffusion.cpp source tree first."
}

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Write-Step "Cleaning previous stable-diffusion build"
    if (-not $DryRun) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallIncludeDir | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallLibDir | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallBinDir | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallGgmlIncludeDir | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallGgmlLibDir | Out-Null
    New-Item -ItemType Directory -Force -Path $VariantRootDir | Out-Null
    if ($ExampleBinDir) {
        New-Item -ItemType Directory -Force -Path $ExampleBinDir | Out-Null
    }
}

$vendorPinPath = Join-Path $SourceDir 'OFX_VENDOR_PIN.txt'
$vendoredCommit = Get-VendoredValue -Path $vendorPinPath -Prefix 'Upstream commit:'
$vendoredTargetCommit = Get-VendoredValue -Path $vendorPinPath -Prefix 'Upstream target commitish:'
$vendoredReleaseTag = Get-VendoredValue -Path $vendorPinPath -Prefix 'Upstream release tag:'
$vendoredVersion = $null
if ($vendoredCommit) {
    $vendoredVersion = "vendored-$($vendoredCommit.Substring(0, [Math]::Min(7, $vendoredCommit.Length)))"
} elseif ($vendoredTargetCommit) {
    $vendoredVersion = "vendored-$($vendoredTargetCommit.Substring(0, [Math]::Min(7, $vendoredTargetCommit.Length)))"
} elseif ($vendoredReleaseTag) {
    $vendoredVersion = "release-$vendoredReleaseTag"
}

$configureArgs = @(
    '-S', $SourceDir,
    '-B', $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Configuration",
    '-DSD_BUILD_SHARED_LIBS=ON',
    ('-DSD_BUILD_EXAMPLES=' + ($(if ($BuildCli) { 'ON' } else { 'OFF' })))
)

if ($vendoredCommit -or $vendoredTargetCommit) {
    $cmakeVendoredCommit = if ($vendoredCommit) { $vendoredCommit } else { $vendoredTargetCommit }
    $configureArgs += @(
        "-DSDCPP_BUILD_COMMIT=$cmakeVendoredCommit",
        "-DSDCPP_BUILD_VERSION=$vendoredVersion"
    )
}

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $configureArgs += @('-G', $Generator)
}

$configureArgs += @(
    ('-DSD_CUDA=' + ($(if ($enableCuda) { 'ON' } else { 'OFF' }))),
    ('-DSD_VULKAN=' + ($(if ($enableVulkan) { 'ON' } else { 'OFF' }))),
    ('-DSD_METAL=' + ($(if ($enableMetal) { 'ON' } else { 'OFF' })))
    '-DSD_WEBP=ON',
    '-DSD_WEBM=ON'
)

Write-Step "Configuring stable-diffusion native library"
Write-Host ("    Backend mode: {0} (CUDA={1}, Vulkan={2}, Metal={3})" -f
    $backendMode,
    $(if ($enableCuda) { 'ON' } else { 'OFF' }),
    $(if ($enableVulkan) { 'ON' } else { 'OFF' }),
    $(if ($enableMetal) { 'ON' } else { 'OFF' }))
if ($buildDirWasImplicit -and (Test-IsWindowsHost) -and $enableVulkan) {
    Write-Host "    Using short Windows build dir for Vulkan: $BuildDir"
}
if ($DryRun) {
    Write-Host "$cmake $($configureArgs -join ' ')"
} else {
    Invoke-External -FilePath $cmake -Arguments $configureArgs
}

$buildArgs = @(
    '--build', $BuildDir,
    '--config', $Configuration,
    '--parallel', $Jobs
)

Write-Step "Building stable-diffusion ($Configuration)"
if ($DryRun) {
    Write-Host "$cmake $($buildArgs -join ' ')"
} else {
    Invoke-External -FilePath $cmake -Arguments $buildArgs
}

if ($DryRun) {
    return
}

$headerPath = Find-FirstFile -Root $SourceDir -Names @('stable-diffusion.h')
$ggmlIncludeSourceDir = Join-Path $SourceDir 'ggml\include'
$dllPath = Find-FirstFile -Root $BuildDir -Names @('stable-diffusion.dll')
$libPath = Find-FirstFile -Root $BuildDir -Names @('stable-diffusion.lib')
$ggmlLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml.lib')
$ggmlBaseLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml-base.lib')
$ggmlCpuLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml-cpu.lib')
$ggmlCudaLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml-cuda.lib')
$ggmlVulkanLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml-vulkan.lib')
$ggmlMetalLibPath = Find-FirstFile -Root $BuildDir -Names @('ggml-metal.lib')
$webpLibPath = Find-FirstFile -Root $BuildDir -Names @('webp.lib')
$webpmuxLibPath = Find-FirstFile -Root $BuildDir -Names @('libwebpmux.lib', 'webpmux.lib')
$webmLibPath = Find-FirstFile -Root $BuildDir -Names @('webm.lib', 'libwebm.lib')
$sdCliPath = Find-FirstFile -Root $BuildDir -Names @('sd-cli.exe')

if (-not $headerPath) {
    throw "Build completed, but stable-diffusion.h was not found under $SourceDir."
}
if (-not $dllPath) {
    throw "Build completed, but stable-diffusion.dll was not found under $BuildDir."
}
if (-not $libPath) {
    throw "Build completed, but stable-diffusion.lib was not found under $BuildDir."
}
if ($BuildCli -and -not $sdCliPath) {
    throw "Build completed, but sd-cli.exe was not found under $BuildDir even though -BuildCli was requested."
}

$optionalSupportLibs = @(
    @{ Name = 'webp.lib'; Path = $webpLibPath },
    @{ Name = 'libwebpmux.lib'; Path = $webpmuxLibPath },
    @{ Name = 'webm.lib'; Path = $webmLibPath }
)
$missingSupportLibs = @($optionalSupportLibs | Where-Object { -not $_.Path })
if ($missingSupportLibs.Count -gt 0) {
    $missingNames = ($missingSupportLibs | ForEach-Object { $_.Name }) -join ', '
    Write-Warning "Optional WebP/WebM support libraries were not found under $BuildDir ($missingNames). Upstream builds may fold these into the main target or avoid staging standalone import libraries, so packaging will continue with the core stable-diffusion artifacts."
}

Write-Step "Staging stable-diffusion artifacts into the addon"
Copy-IfPresent -Path $headerPath -Destination $InstallIncludeDir
Copy-IfPresent -Path $dllPath -Destination $InstallLibDir
Copy-IfPresent -Path $libPath -Destination $InstallLibDir
Copy-IfPresent -Path $sdCliPath -Destination $InstallBinDir
Copy-IfPresent -Path $webpLibPath -Destination $InstallLibDir
Copy-IfPresent -Path $webpmuxLibPath -Destination $InstallLibDir
Copy-IfPresent -Path $webmLibPath -Destination $InstallLibDir
if ($ExampleBinDir) {
    Copy-IfPresent -Path $dllPath -Destination $ExampleBinDir -AllowLockedDestination
}

Write-Step "Staging ggml artifacts into the addon"
if (Test-Path -LiteralPath $ggmlIncludeSourceDir) {
    Get-ChildItem -LiteralPath $ggmlIncludeSourceDir -File |
        ForEach-Object {
            Copy-IfPresent -Path $_.FullName -Destination $InstallGgmlIncludeDir
        }
} else {
    Write-Warning "ggml headers were not found under $ggmlIncludeSourceDir. Separate ggml headers will not be staged."
}
Copy-IfPresent -Path $ggmlLibPath -Destination $InstallGgmlLibDir
Copy-IfPresent -Path $ggmlBaseLibPath -Destination $InstallGgmlLibDir
Copy-IfPresent -Path $ggmlCpuLibPath -Destination $InstallGgmlLibDir
Copy-IfPresent -Path $ggmlCudaLibPath -Destination $InstallGgmlLibDir
Copy-IfPresent -Path $ggmlVulkanLibPath -Destination $InstallGgmlLibDir
Copy-IfPresent -Path $ggmlMetalLibPath -Destination $InstallGgmlLibDir

$variantStableDiffusionIncludeDir = Join-Path $VariantRootDir "$backendMode\stable-diffusion\include"
$variantStableDiffusionLibDir = Join-Path $VariantRootDir "$backendMode\stable-diffusion\lib\vs"
$variantStableDiffusionBinDir = Join-Path $VariantRootDir "$backendMode\stable-diffusion\bin\vs"
$variantGgmlIncludeDir = Join-Path $VariantRootDir "$backendMode\ggml\include"
$variantGgmlLibDir = Join-Path $VariantRootDir "$backendMode\ggml\lib\vs"

Write-Step "Snapshotting backend variant artifacts"
Copy-DirectoryContents -Source $InstallIncludeDir -Destination $variantStableDiffusionIncludeDir
Copy-DirectoryContents -Source $InstallLibDir -Destination $variantStableDiffusionLibDir
Copy-DirectoryContents -Source $InstallBinDir -Destination $variantStableDiffusionBinDir
Copy-DirectoryContents -Source $InstallGgmlIncludeDir -Destination $variantGgmlIncludeDir
Copy-DirectoryContents -Source $InstallGgmlLibDir -Destination $variantGgmlLibDir

Write-Host ""
Write-Host "stable-diffusion native build complete."
Write-Host "  source:  $SourceDir"
Write-Host "  build:   $BuildDir"
Write-Host "  include: $InstallIncludeDir"
Write-Host "  libs:    $InstallLibDir"
Write-Host "  bins:    $InstallBinDir"
Write-Host "  ggml include: $InstallGgmlIncludeDir"
Write-Host "  ggml libs:    $InstallGgmlLibDir"
Write-Host "  variant snapshot: $VariantRootDir\$backendMode"
if ($ExampleBinDir) {
    Write-Host "  runtime: $ExampleBinDir"
}


