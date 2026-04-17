param(
    [string]$BuildDir = "",
    [string]$Configuration = "Release",
    [string]$Generator = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$testsRoot = Join-Path $addonRoot "tests"

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $testsRoot "build"
}

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Write-Step "Cleaning test build directory"
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source

$configureArgs = @(
    "-S", $testsRoot,
    "-B", $BuildDir
)

if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $configureArgs += @("-G", $Generator)
}

Write-Step "Configuring tests"
& $cmake @configureArgs

Write-Step "Building tests ($Configuration)"
& $cmake --build $BuildDir --config $Configuration

Write-Step "Running tests"
& ctest --test-dir $BuildDir -C $Configuration --output-on-failure
