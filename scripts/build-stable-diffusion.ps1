param(
    [string]$SourceDir = "",
    [string]$BuildDir = "",
    [string]$InstallIncludeDir = "",
    [string]$InstallLibDir = "",
    [string]$ExampleBinDir = "",
    [string]$Configuration = "Release",
    [string]$Generator = "",
    [int]$Jobs = 0,
    [switch]$Clean,
    [Alias('Cpu')][switch]$CpuOnly,
    [Alias('Gpu')][switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Metal,
    [switch]$Auto,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

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
if ([string]::IsNullOrWhiteSpace($ExampleBinDir)) {
    $ExampleBinDir = Join-Path $addonRoot 'ofxStableDiffusionExample\bin'
}
if ($Jobs -le 0) {
    $Jobs = [Math]::Max(1, [Environment]::ProcessorCount)
}

$enableCuda = $false
$enableVulkan = $false
$enableMetal = $false
$backendMode = "auto-detect"

if ($CpuOnly) {
    $backendMode = "cpu-only"
} elseif ($Cuda -or $Vulkan -or $Metal) {
    $backendMode = "explicit"
    $enableCuda = $Cuda
    $enableVulkan = $Vulkan
    $enableMetal = $Metal
} else {
    $enableCuda = Test-CudaAvailable
    $enableVulkan = Test-VulkanAvailable
    $enableMetal = Test-MetalAvailable
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

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw @"
stable-diffusion.cpp source was not found at:
  $SourceDir

Recommended workflow:
  1. Vendor a known-compatible stable-diffusion.cpp snapshot into libs/stable-diffusion/source
  2. Re-run scripts/build-stable-diffusion.ps1

This addon intentionally keeps stable-diffusion.cpp standalone rather than sharing
the ggml build from ofxGgml, to avoid ABI/version coupling across addons.
"@
}

$cmakeLists = Join-Path $SourceDir 'CMakeLists.txt'
if (-not (Test-Path -LiteralPath $cmakeLists)) {
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
    if ($ExampleBinDir) {
        New-Item -ItemType Directory -Force -Path $ExampleBinDir | Out-Null
    }
}

$vendorPinPath = Join-Path $SourceDir 'OFX_VENDOR_PIN.txt'
$vendoredCommit = Get-VendoredValue -Path $vendorPinPath -Prefix 'Upstream commit:'
$vendoredVersion = $null
if ($vendoredCommit) {
    $vendoredVersion = "vendored-$($vendoredCommit.Substring(0, [Math]::Min(7, $vendoredCommit.Length)))"
}

$configureArgs = @(
    '-S', $SourceDir,
    '-B', $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Configuration",
    '-DSD_BUILD_SHARED_LIBS=ON',
    '-DSD_BUILD_EXAMPLES=OFF'
)

if ($vendoredCommit) {
    $configureArgs += @(
        "-DSDCPP_BUILD_COMMIT=$vendoredCommit",
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

$headerPath = Find-FirstFile -Root $SourceDir -Names @('stable-diffusion.h')
$dllPath = Find-FirstFile -Root $BuildDir -Names @('stable-diffusion.dll')
$libPath = Find-FirstFile -Root $BuildDir -Names @('stable-diffusion.lib')

if (-not $headerPath) {
    throw "Build completed, but stable-diffusion.h was not found under $SourceDir."
}
if (-not $dllPath) {
    throw "Build completed, but stable-diffusion.dll was not found under $BuildDir."
}
if (-not $libPath) {
    throw "Build completed, but stable-diffusion.lib was not found under $BuildDir."
}

Write-Step "Staging stable-diffusion artifacts into the addon"
Copy-IfPresent -Path $dllPath -Destination $InstallLibDir
Copy-IfPresent -Path $libPath -Destination $InstallLibDir
if ($ExampleBinDir) {
    Copy-IfPresent -Path $dllPath -Destination $ExampleBinDir -AllowLockedDestination
}

Write-Host ""
Write-Host "stable-diffusion native build complete."
Write-Host "  source:  $SourceDir"
Write-Host "  build:   $BuildDir"
Write-Host "  include: $InstallIncludeDir"
Write-Host "  libs:    $InstallLibDir"
if ($ExampleBinDir) {
    Write-Host "  runtime: $ExampleBinDir"
}
