param(
    [string]$SourceDir = "",
    [string]$ReleaseTag = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Invoke-GitHubJsonRequest {
    param([string]$Uri)

    $headers = @{
        'Accept' = 'application/vnd.github+json'
        'User-Agent' = 'ofxStableDiffusion-source-refresh'
        'X-GitHub-Api-Version' = '2022-11-28'
    }

    try {
        return Invoke-RestMethod -Uri $Uri -Headers $headers
    } catch {
        throw "GitHub API request failed for $Uri. $($_.Exception.Message)"
    }
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
        [object]$ReleaseMetadata
    )

    $lines = @(
        'Vendored by ofxStableDiffusion'
        ('Upstream release tag: ' + $ReleaseMetadata.tag_name)
        ('Upstream target commitish: ' + $ReleaseMetadata.target_commitish)
        ('Release URL: ' + $ReleaseMetadata.html_url)
        ('Zipball URL: ' + $ReleaseMetadata.zipball_url)
        ('Fetched at: ' + [DateTimeOffset]::UtcNow.ToString('o'))
    )

    [System.IO.File]::WriteAllLines($Path, $lines)
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $addonRoot 'libs\stable-diffusion\source'
}

$releaseMetadata = Get-ReleaseMetadata -Tag $ReleaseTag
$resolvedTag = $releaseMetadata.tag_name
$downloadRoot = Join-Path $env:TEMP 'ofxsd-source-release'
$zipPath = Join-Path $downloadRoot ($resolvedTag + '.zip')
$extractRoot = Join-Path $downloadRoot ('extract-' + $resolvedTag)

Write-Step "Refreshing stable-diffusion source from release snapshot"
Write-Host ("    Release tag: {0}" -f $resolvedTag)
Write-Host ("    Destination: {0}" -f $SourceDir)

if ($DryRun) {
    Write-Host ("Download {0}" -f $releaseMetadata.zipball_url)
    Write-Host ("Extract to {0}" -f $extractRoot)
    Write-Host ("Replace contents of {0}" -f $SourceDir)
    return
}

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
if (Test-Path -LiteralPath $extractRoot) {
    Remove-Item -LiteralPath $extractRoot -Recurse -Force
}

Invoke-WebRequest -Uri $releaseMetadata.zipball_url -OutFile $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force

$sourceRoot = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
if (-not $sourceRoot) {
    throw "The release source archive for '$resolvedTag' did not extract to a usable folder."
}

$cmakeLists = Join-Path $sourceRoot.FullName 'CMakeLists.txt'
if (-not (Test-Path -LiteralPath $cmakeLists)) {
    throw "The release source archive for '$resolvedTag' did not contain a top-level CMakeLists.txt."
}

New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
Remove-DirectoryContents -LiteralPath $SourceDir

Get-ChildItem -LiteralPath $sourceRoot.FullName -Force |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $SourceDir -Recurse -Force
    }

Write-VendorPinFile -Path (Join-Path $SourceDir 'OFX_VENDOR_PIN.txt') -ReleaseMetadata $releaseMetadata

Write-Host ""
Write-Host "stable-diffusion source refresh complete."
Write-Host "  source: $SourceDir"
Write-Host "  tag:    $resolvedTag"
