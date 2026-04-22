param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cpu-only', 'cuda', 'vulkan', 'metal')]
    [string]$Backend,
    [string]$VariantRootDir = "",
    [string]$InstallIncludeDir = "",
    [string]$InstallLibDir = "",
    [string]$InstallGgmlIncludeDir = "",
    [string]$InstallGgmlLibDir = "",
    [string]$ExampleBinDir = ""
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
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

function Copy-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Variant path was not found: $Source"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
        }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($VariantRootDir)) {
    $VariantRootDir = Join-Path $addonRoot 'libs\variants'
}
if ([string]::IsNullOrWhiteSpace($InstallIncludeDir)) {
    $InstallIncludeDir = Join-Path $addonRoot 'libs\stable-diffusion\include'
}
if ([string]::IsNullOrWhiteSpace($InstallLibDir)) {
    $InstallLibDir = Join-Path $addonRoot 'libs\stable-diffusion\lib\vs'
}
if ([string]::IsNullOrWhiteSpace($InstallGgmlIncludeDir)) {
    $InstallGgmlIncludeDir = Join-Path $addonRoot 'libs\ggml\include'
}
if ([string]::IsNullOrWhiteSpace($InstallGgmlLibDir)) {
    $InstallGgmlLibDir = Join-Path $addonRoot 'libs\ggml\lib\vs'
}
if ([string]::IsNullOrWhiteSpace($ExampleBinDir)) {
    $ExampleBinDir = Join-Path $addonRoot 'ofxStableDiffusionExample\bin'
}

$variantStableDiffusionIncludeDir = Join-Path $VariantRootDir "$Backend\stable-diffusion\include"
$variantStableDiffusionLibDir = Join-Path $VariantRootDir "$Backend\stable-diffusion\lib\vs"
$variantGgmlIncludeDir = Join-Path $VariantRootDir "$Backend\ggml\include"
$variantGgmlLibDir = Join-Path $VariantRootDir "$Backend\ggml\lib\vs"

if (-not (Test-Path -LiteralPath $variantStableDiffusionIncludeDir)) {
    throw "The '$Backend' stable-diffusion variant is not staged yet. Build it first with scripts/build-stable-diffusion.ps1."
}
if (-not (Test-Path -LiteralPath $variantStableDiffusionLibDir)) {
    throw "The '$Backend' stable-diffusion libraries are not staged yet. Build it first with scripts/build-stable-diffusion.ps1."
}

Write-Step "Selecting backend variant"
Write-Host "    Backend: $Backend"
Write-Host "    Variant root: $VariantRootDir"

New-Item -ItemType Directory -Force -Path $InstallIncludeDir | Out-Null
New-Item -ItemType Directory -Force -Path $InstallLibDir | Out-Null
New-Item -ItemType Directory -Force -Path $InstallGgmlIncludeDir | Out-Null
New-Item -ItemType Directory -Force -Path $InstallGgmlLibDir | Out-Null

Remove-DirectoryContents -LiteralPath $InstallIncludeDir
Remove-DirectoryContents -LiteralPath $InstallLibDir
Remove-DirectoryContents -LiteralPath $InstallGgmlIncludeDir
Remove-DirectoryContents -LiteralPath $InstallGgmlLibDir

Copy-DirectoryContents -Source $variantStableDiffusionIncludeDir -Destination $InstallIncludeDir
Copy-DirectoryContents -Source $variantStableDiffusionLibDir -Destination $InstallLibDir

if (Test-Path -LiteralPath $variantGgmlIncludeDir) {
    Copy-DirectoryContents -Source $variantGgmlIncludeDir -Destination $InstallGgmlIncludeDir
}
if (Test-Path -LiteralPath $variantGgmlLibDir) {
    Copy-DirectoryContents -Source $variantGgmlLibDir -Destination $InstallGgmlLibDir
}

$selectedDll = Join-Path $InstallLibDir 'stable-diffusion.dll'
if ((Test-Path -LiteralPath $selectedDll) -and (Test-Path -LiteralPath $ExampleBinDir)) {
    try {
        Copy-Item -LiteralPath $selectedDll -Destination $ExampleBinDir -Force
    } catch [System.IO.IOException] {
        Write-Warning "Skipping example runtime copy because the destination DLL is in use. Close the running example app and rerun this selector if you want the chosen backend staged there too."
    }
}

Write-Host ""
Write-Host "Selected backend variant: $Backend"
Write-Host "  stable-diffusion include: $InstallIncludeDir"
Write-Host "  stable-diffusion libs:    $InstallLibDir"
Write-Host "  ggml include:             $InstallGgmlIncludeDir"
Write-Host "  ggml libs:                $InstallGgmlLibDir"
