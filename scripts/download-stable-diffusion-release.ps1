param(
    [string]$GgmlReleaseTag = "",
    [string]$SourceDir = "",
    [string]$SourceReleaseTag = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$DefaultSourceReleaseTag = "master-585-44cca3d"

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

function Invoke-GitHubJsonRequest {
    param([string]$Uri)

    $headers = @{
        'Accept' = 'application/vnd.github+json'
        'User-Agent' = 'ofxStableDiffusion-source-download'
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

    & $GitPath clone --depth 1 --branch $resolvedReleaseTag https://github.com/ggml-org/ggml.git $cloneRoot
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed while refreshing ggml source snapshot for '$resolvedReleaseTag'."
    }

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

$effectiveSourceReleaseTag = if ([string]::IsNullOrWhiteSpace($SourceReleaseTag)) { $DefaultSourceReleaseTag } else { $SourceReleaseTag }
$releaseMetadata = Get-ReleaseMetadata -Tag $effectiveSourceReleaseTag
$resolvedTag = $releaseMetadata.tag_name
$downloadRoot = Join-Path $env:TEMP 'ofxsd-source-release'
$cloneRoot = Join-Path $downloadRoot ('clone-' + $resolvedTag)
$git = Require-GitPath

Write-Step "Refreshing stable-diffusion source from upstream snapshot"
Write-Host ("    Release tag: {0}" -f $resolvedTag)
Write-Host ("    Source dir: {0}" -f $SourceDir)

if ($DryRun) {
    Write-Host ("Clone https://github.com/leejet/stable-diffusion.cpp.git tag {0} with submodules" -f $resolvedTag)
    Write-Host ("Clone to {0}" -f $cloneRoot)
    Write-Host ("Replace contents of {0}" -f $SourceDir)
} else {
    New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
    if (Test-Path -LiteralPath $cloneRoot) {
        Remove-Item -LiteralPath $cloneRoot -Recurse -Force
    }

    & $git clone --depth 1 --branch $resolvedTag --recurse-submodules --shallow-submodules `
        https://github.com/leejet/stable-diffusion.cpp.git $cloneRoot
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed while refreshing source snapshot for '$resolvedTag'."
    }

    $cmakeLists = Join-Path $cloneRoot 'CMakeLists.txt'
    if (-not (Test-Path -LiteralPath $cmakeLists)) {
        throw "The recursive clone for '$resolvedTag' did not contain a top-level CMakeLists.txt."
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

Write-Host ""
Write-Host "stable-diffusion source refresh complete."
