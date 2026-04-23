param(
    [string]$Configuration = "Release",
    [int]$Jobs = 0,
    [Alias('Cpu')][switch]$CpuOnly,
    [Alias('Gpu')][switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Metal,
    [switch]$All,
    [switch]$BuildCli,
    [ValidateSet('cpu-only', 'cuda', 'vulkan', 'metal')]
    [string]$SelectBackend = "",
    [string]$GgmlReleaseTag = "",
    [switch]$Clean,
    [switch]$SkipNative,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$nativeBuilder = Join-Path $scriptRoot 'build-stable-diffusion.ps1'
$backendSelector = Join-Path $scriptRoot 'select-stable-diffusion-backend.ps1'

Write-Host ""
Write-Host "  ============================================="
Write-Host "        ofxStableDiffusion Setup (Windows)"
Write-Host "  ============================================="
Write-Host ""

if (-not $SkipNative) {
    Write-Step "Step 1/1: Building stable-diffusion native runtime"
    $nativeArgs = @{
        Configuration = $Configuration
    }
    if ($Jobs -gt 0) {
        $nativeArgs.Jobs = $Jobs
    }
    if ($CpuOnly) {
        $nativeArgs.CpuOnly = $true
    }
    if ($Cuda) {
        $nativeArgs.Cuda = $true
    }
    if ($Vulkan) {
        $nativeArgs.Vulkan = $true
    }
    if ($Metal) {
        $nativeArgs.Metal = $true
    }
    if ($All) {
        $nativeArgs.All = $true
    }
    if ($BuildCli) {
        $nativeArgs.BuildCli = $true
    }
    if (-not [string]::IsNullOrWhiteSpace($GgmlReleaseTag)) {
        $nativeArgs.GgmlReleaseTag = $GgmlReleaseTag
    }
    if ($Clean) {
        $nativeArgs.Clean = $true
    }
    if ($DryRun) {
        $nativeArgs.DryRun = $true
    }
    & $nativeBuilder @nativeArgs
} elseif (-not [string]::IsNullOrWhiteSpace($SelectBackend)) {
    Write-Step "Step 1/1: Skipping native build and selecting staged backend variant"
} else {
    Write-Step "Step 1/1: Skipping stable-diffusion native runtime (--skip-native)"
}

if (-not [string]::IsNullOrWhiteSpace($SelectBackend)) {
    Write-Step "Selecting staged backend variant"
    $selectorArgs = @{
        Backend = $SelectBackend
    }
    & $backendSelector @selectorArgs
}

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Add ofxStableDiffusion to your project addons.make"
Write-Host "  2. Place your model files under your project's data folder"
Write-Host "  3. Build your OF project"
Write-Host ""
