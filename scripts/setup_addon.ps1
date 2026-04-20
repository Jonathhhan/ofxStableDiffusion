param(
    [string]$Configuration = "Release",
    [int]$Jobs = 0,
    [Alias('Cpu')][switch]$CpuOnly,
    [Alias('Gpu')][switch]$Cuda,
    [switch]$Vulkan,
    [switch]$Metal,
    [switch]$Auto,
    [switch]$UseRelease,
    [string]$ReleaseTag = "",
    [string]$ReleaseVariant = "auto",
    [string]$SourceReleaseTag = "",
    [switch]$Clean,
    [switch]$SkipNative,
    [switch]$SkipExampleBuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Get-MSBuildPath {
    $direct = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($direct) {
        return $direct.Source
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' |
            Select-Object -First 1
        if ($found) {
            return $found
        }
    }

    $fallbacks = @(
        'C:\Program Files\Microsoft Visual Studio\18\Professional\MSBuild\Current\Bin\MSBuild.exe',
        'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe',
        'C:\Program Files\Microsoft Visual Studio\17\Professional\MSBuild\Current\Bin\MSBuild.exe',
        'C:\Program Files\Microsoft Visual Studio\17\Community\MSBuild\Current\Bin\MSBuild.exe'
    )
    foreach ($path in $fallbacks) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }

    return $null
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path
$nativeBuilder = Join-Path $scriptRoot 'build-stable-diffusion.ps1'
$releaseDownloader = Join-Path $scriptRoot 'download-stable-diffusion-release.ps1'
$exampleProject = Join-Path $addonRoot 'ofxStableDiffusionExample\ofxStableDiffusionExample.vcxproj'

Write-Host ""
Write-Host "  ============================================="
Write-Host "        ofxStableDiffusion Setup (Windows)"
Write-Host "  ============================================="
Write-Host ""

if (-not $SkipNative) {
    Write-Step "Step 1/2: Building stable-diffusion native runtime"
    if ($UseRelease) {
        if ($Vulkan) {
            throw "Windows upstream releases do not currently provide a Vulkan runtime asset for this addon flow. Re-run without -UseRelease to build Vulkan from source."
        }
        if ($Metal) {
            throw "Metal release staging is not supported from the Windows setup flow."
        }

        $releaseArgs = @{
            ReleaseVariant = $ReleaseVariant
        }
        if (-not [string]::IsNullOrWhiteSpace($ReleaseTag)) {
            $releaseArgs.ReleaseTag = $ReleaseTag
        }
        if ($Cuda) {
            $releaseArgs.ReleaseVariant = 'cuda12'
        } elseif ($CpuOnly -and $ReleaseVariant -eq 'auto') {
            $releaseArgs.ReleaseVariant = 'cpu'
        }
        if ($DryRun) {
            $releaseArgs.DryRun = $true
        }
        & $releaseDownloader @releaseArgs
    } else {
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
        if ($Auto) {
            $nativeArgs.Auto = $true
        }
        if (-not [string]::IsNullOrWhiteSpace($SourceReleaseTag)) {
            $nativeArgs.SourceReleaseTag = $SourceReleaseTag
        }
        if ($Clean) {
            $nativeArgs.Clean = $true
        }
        if ($DryRun) {
            $nativeArgs.DryRun = $true
        }
        & $nativeBuilder @nativeArgs
    }
} else {
    Write-Step "Step 1/2: Skipping stable-diffusion native runtime (--skip-native)"
}

if (-not $SkipExampleBuild) {
    Write-Step "Step 2/2: Building ofxStableDiffusionExample"
    $msbuild = Get-MSBuildPath
    if (-not $msbuild) {
        throw "MSBuild.exe was not found. Re-run with -SkipExampleBuild or install Visual Studio Build Tools."
    }

    $buildArgs = @(
        $exampleProject,
        '/t:Build',
        "/p:Configuration=$Configuration",
        '/p:Platform=x64',
        '/m'
    )

    if ($DryRun) {
        Write-Host "$msbuild $($buildArgs -join ' ')"
    } else {
        & $msbuild @buildArgs
    }
} else {
    Write-Step "Step 2/2: Skipping example build (--skip-example-build)"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Add ofxStableDiffusion to your project addons.make"
Write-Host "  2. Place your model files under your project's data folder"
Write-Host "  3. Run ofxStableDiffusionExample or your own OF project"
Write-Host ""
