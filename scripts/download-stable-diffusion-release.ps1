param(
    [string]$ReleaseTag = "",
    [string]$ReleaseVariant = "auto",
    [string]$InstallLibDir = "",
    [string]$ExampleBinDir = "",
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
        'User-Agent' = 'ofxStableDiffusion-release-downloader'
        'X-GitHub-Api-Version' = '2022-11-28'
    }

    try {
        return Invoke-RestMethod -Uri $Uri -Headers $headers
    } catch {
        throw "GitHub release query failed for $Uri. $($_.Exception.Message)"
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

function Find-VsToolPath {
    param([string]$ToolName)

    $fromPath = Get-CommandPathOrNull $ToolName
    if ($fromPath) {
        return $fromPath
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        return $null
    }

    $found = & $vswhere -latest -products * -find ("VC\Tools\MSVC\**\bin\Hostx64\x64\" + $ToolName) |
        Select-Object -First 1
    if ($found) {
        return $found
    }

    return $null
}

function Test-CudaAvailable {
    if ($env:CUDA_PATH -and (Test-Path -LiteralPath $env:CUDA_PATH)) {
        return $true
    }
    return [bool](Get-CommandPathOrNull 'nvcc.exe')
}

function New-ImportLibraryFromDll {
    param(
        [string]$DllPath,
        [string]$OutputLibPath
    )

    $dumpbin = Find-VsToolPath -ToolName 'dumpbin.exe'
    $libexe = Find-VsToolPath -ToolName 'lib.exe'
    if (-not $dumpbin -or -not $libexe) {
        throw "The upstream release did not include stable-diffusion.lib, and Visual Studio tools dumpbin.exe/lib.exe were not found to synthesize it."
    }

    $dumpOutput = & $dumpbin /exports $DllPath | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin.exe failed while reading exports from $DllPath."
    }

    $exportNames = New-Object System.Collections.Generic.List[string]
    foreach ($line in ($dumpOutput -split "`r?`n")) {
        if ($line -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z_][A-Za-z0-9_]*)$') {
            $exportNames.Add($Matches[1])
        }
    }

    if ($exportNames.Count -eq 0) {
        throw "No exported symbols were found in $DllPath."
    }

    $defPath = [System.IO.Path]::ChangeExtension($OutputLibPath, '.def')
    $defLines = @(
        'LIBRARY stable-diffusion.dll',
        'EXPORTS'
    ) + ($exportNames | Sort-Object -Unique | ForEach-Object { "    $_" })
    [System.IO.File]::WriteAllLines($defPath, $defLines)

    & $libexe /def:$defPath /machine:x64 /out:$OutputLibPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "lib.exe failed while creating $OutputLibPath from $DllPath."
    }

    return $OutputLibPath
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

function Resolve-ReleaseVariant {
    param([string]$RequestedVariant)

    $normalized = $RequestedVariant.Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($normalized) -or $normalized -eq 'auto') {
        if (Test-CudaAvailable) {
            return 'cuda12'
        }
        return 'avx2'
    }

    switch ($normalized) {
        'cpu' { return 'avx2' }
        'noavx' { return 'noavx' }
        'avx' { return 'avx' }
        'avx2' { return 'avx2' }
        'avx512' { return 'avx512' }
        'cuda' { return 'cuda12' }
        'cuda12' { return 'cuda12' }
        default {
            throw "Unsupported release variant '$RequestedVariant'. Supported variants: auto, cpu, noavx, avx, avx2, avx512, cuda12."
        }
    }
}

function Get-ReleaseMetadata {
    param(
        [string]$Tag
    )

    $repoApiBase = 'https://api.github.com/repos/leejet/stable-diffusion.cpp/releases'
    if ([string]::IsNullOrWhiteSpace($Tag)) {
        return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/latest')
    }

    $escapedTag = [System.Uri]::EscapeDataString($Tag)
    return Invoke-GitHubJsonRequest -Uri ($repoApiBase + '/tags/' + $escapedTag)
}

function Find-ReleaseAssetName {
    param(
        [object]$ReleaseMetadata,
        [string]$Pattern
    )

    $asset = $ReleaseMetadata.assets |
        Where-Object { $_.name -like $Pattern } |
        Select-Object -First 1
    if (-not $asset) {
        throw "The upstream release '$($ReleaseMetadata.tag_name)' did not contain an asset matching '$Pattern'."
    }

    return $asset.name
}

function Get-ReleaseAssetSet {
    param(
        [object]$ReleaseMetadata,
        [string]$Variant
    )

    switch ($Variant) {
        'noavx' {
            return @{
                Variant = $Variant
                Archives = @(
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'sd-*-bin-win-noavx-x64.zip')
                )
            }
        }
        'avx' {
            return @{
                Variant = $Variant
                Archives = @(
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'sd-*-bin-win-avx-x64.zip')
                )
            }
        }
        'avx2' {
            return @{
                Variant = $Variant
                Archives = @(
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'sd-*-bin-win-avx2-x64.zip')
                )
            }
        }
        'avx512' {
            return @{
                Variant = $Variant
                Archives = @(
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'sd-*-bin-win-avx512-x64.zip')
                )
            }
        }
        'cuda12' {
            return @{
                Variant = $Variant
                Archives = @(
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'sd-*-bin-win-cuda12-x64.zip'),
                    (Find-ReleaseAssetName -ReleaseMetadata $ReleaseMetadata -Pattern 'cudart-sd-bin-win-cu12-x64.zip')
                )
            }
        }
        default {
            throw "Internal error: unhandled release variant '$Variant'."
        }
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($InstallLibDir)) {
    $InstallLibDir = Join-Path $addonRoot 'libs\stable-diffusion\lib\vs'
}
if ([string]::IsNullOrWhiteSpace($ExampleBinDir)) {
    $ExampleBinDir = Join-Path $addonRoot 'ofxStableDiffusionExample\bin'
}

$resolvedVariant = Resolve-ReleaseVariant -RequestedVariant $ReleaseVariant
$releaseMetadata = Get-ReleaseMetadata -Tag $ReleaseTag
$ReleaseTag = $releaseMetadata.tag_name
$assetSet = Get-ReleaseAssetSet -ReleaseMetadata $releaseMetadata -Variant $resolvedVariant

$downloadRoot = Join-Path $env:TEMP 'ofxsd-release'
$extractRoot = Join-Path $downloadRoot ("extract-" + $ReleaseTag + '-' + $assetSet.Variant)

Write-Step "Staging stable-diffusion upstream release runtime"
Write-Host ("    Release tag: {0}" -f $ReleaseTag)
Write-Host ("    Release variant: {0}" -f $assetSet.Variant)

if ($DryRun) {
    foreach ($archive in $assetSet.Archives) {
        $uri = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$ReleaseTag/$archive"
        Write-Host "Download $uri"
    }
} else {
    New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $InstallLibDir | Out-Null
    if ($ExampleBinDir) {
        New-Item -ItemType Directory -Force -Path $ExampleBinDir | Out-Null
    }

    foreach ($archive in $assetSet.Archives) {
        $uri = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$ReleaseTag/$archive"
        $zipPath = Join-Path $downloadRoot $archive
        $assetExtractDir = Join-Path $extractRoot ([System.IO.Path]::GetFileNameWithoutExtension($archive))

        Write-Host ("    Downloading {0}" -f $archive)
        Invoke-WebRequest -Uri $uri -OutFile $zipPath

        if (Test-Path -LiteralPath $assetExtractDir) {
            Remove-Item -LiteralPath $assetExtractDir -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $assetExtractDir | Out-Null
        Expand-Archive -LiteralPath $zipPath -DestinationPath $assetExtractDir -Force
    }
}

$runtimeDll = Find-FirstFile -Root $extractRoot -Names @('stable-diffusion.dll')
$importLib = Find-FirstFile -Root $extractRoot -Names @('stable-diffusion.lib')
$allDlls = @()
if (Test-Path -LiteralPath $extractRoot) {
    $allDlls = Get-ChildItem -LiteralPath $extractRoot -Recurse -File -Filter '*.dll' -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName
}

if (-not $DryRun) {
    if (-not $runtimeDll) {
        throw "Downloaded release assets did not contain stable-diffusion.dll. The upstream release layout may have changed."
    }
    if (-not $importLib) {
        Write-Host "    stable-diffusion.lib was not present in the upstream release; synthesizing an import library from DLL exports."
        $importLib = New-ImportLibraryFromDll -DllPath $runtimeDll -OutputLibPath (Join-Path $extractRoot 'stable-diffusion.lib')
    }
}

Write-Step "Staging release artifacts into the addon"
Copy-IfPresent -Path $importLib -Destination $InstallLibDir
foreach ($dll in $allDlls) {
    Copy-IfPresent -Path $dll -Destination $InstallLibDir
    if ($ExampleBinDir) {
        Copy-IfPresent -Path $dll -Destination $ExampleBinDir -AllowLockedDestination
    }
}

Write-Host ""
Write-Host "stable-diffusion release staging complete."
Write-Host "  tag:     $ReleaseTag"
Write-Host "  variant: $($assetSet.Variant)"
Write-Host "  libs:    $InstallLibDir"
if ($ExampleBinDir) {
    Write-Host "  runtime: $ExampleBinDir"
}
