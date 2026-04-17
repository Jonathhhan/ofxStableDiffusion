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
    [switch]$CpuOnly,
    [switch]$Cuda,
    [switch]$Vulkan,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Get-CommandPathOrNull {
    param([string]$Name)
    try {
        return (Get-Command $Name -ErrorAction Stop).Source
    } catch {
        return $null
    }
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
        [string]$Destination
    )

    if (-not $Path) {
        return
    }

    if ($DryRun) {
        Write-Host "Copy $Path -> $Destination"
        return
    }

    Copy-Item -LiteralPath $Path -Destination $Destination -Force
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $addonRoot 'libs\stable-diffusion\source'
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $addonRoot 'libs\stable-diffusion\build'
}
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

$configureArgs = @(
    '-S', $SourceDir,
    '-B', $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Configuration",
    '-DBUILD_SHARED_LIBS=ON',
    '-DSD_BUILD_SHARED_LIB=ON',
    '-DSD_BUILD_TESTS=OFF',
    '-DSD_BUILD_EXAMPLES=OFF'
)

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $configureArgs += @('-G', $Generator)
}

if ($CpuOnly) {
    $configureArgs += @('-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF')
} else {
    $configureArgs += @(
        ('-DGGML_CUDA=' + ($(if ($Cuda) { 'ON' } else { 'OFF' }))),
        ('-DGGML_VULKAN=' + ($(if ($Vulkan) { 'ON' } else { 'OFF' })))
    )
}

Write-Step "Configuring stable-diffusion native library"
if ($DryRun) {
    Write-Host "$cmake $($configureArgs -join ' ')"
} else {
    & $cmake @configureArgs
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
    & $cmake @buildArgs
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
Copy-IfPresent -Path $headerPath -Destination $InstallIncludeDir
Copy-IfPresent -Path $dllPath -Destination $InstallLibDir
Copy-IfPresent -Path $libPath -Destination $InstallLibDir
if ($ExampleBinDir) {
    Copy-IfPresent -Path $dllPath -Destination $ExampleBinDir
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
