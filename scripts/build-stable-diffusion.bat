@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PS_ARGS="

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--cpu" (
    set "PS_ARGS=!PS_ARGS! -CpuOnly"
    shift
    goto parse_args
)
if /i "%~1"=="--cpu-only" (
    set "PS_ARGS=!PS_ARGS! -CpuOnly"
    shift
    goto parse_args
)
if /i "%~1"=="--auto" (
    set "PS_ARGS=!PS_ARGS! -Auto"
    shift
    goto parse_args
)
if /i "%~1"=="--cuda" (
    set "PS_ARGS=!PS_ARGS! -Cuda"
    shift
    goto parse_args
)
if /i "%~1"=="--gpu" (
    set "PS_ARGS=!PS_ARGS! -Cuda"
    shift
    goto parse_args
)
if /i "%~1"=="--vulkan" (
    set "PS_ARGS=!PS_ARGS! -Vulkan"
    shift
    goto parse_args
)
if /i "%~1"=="--metal" (
    set "PS_ARGS=!PS_ARGS! -Metal"
    shift
    goto parse_args
)
if /i "%~1"=="--clean" (
    set "PS_ARGS=!PS_ARGS! -Clean"
    shift
    goto parse_args
)
if /i "%~1"=="--source-release-tag" (
    if "%~2"=="" (
        echo Error: --source-release-tag requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -SourceReleaseTag ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--dry-run" (
    set "PS_ARGS=!PS_ARGS! -DryRun"
    shift
    goto parse_args
)
if /i "%~1"=="--config" (
    if "%~2"=="" (
        echo Error: --config requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -Configuration ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--jobs" (
    if "%~2"=="" (
        echo Error: --jobs requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -Jobs %~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--source-dir" (
    if "%~2"=="" (
        echo Error: --source-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -SourceDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--build-dir" (
    if "%~2"=="" (
        echo Error: --build-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -BuildDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--install-include-dir" (
    if "%~2"=="" (
        echo Error: --install-include-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -InstallIncludeDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--install-lib-dir" (
    if "%~2"=="" (
        echo Error: --install-lib-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -InstallLibDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--example-bin-dir" (
    if "%~2"=="" (
        echo Error: --example-bin-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -ExampleBinDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
set "PS_ARGS=!PS_ARGS! %~1"
shift
goto parse_args

:usage
echo build-stable-diffusion.bat - Build the bundled stable-diffusion.cpp runtime.
echo.
echo Usage:
echo   scripts\build-stable-diffusion.bat [OPTIONS]
echo.
echo Options:
echo   --cpu, --cpu-only      Build CPU backend only
echo   --auto                 Auto-detect GPU backends ^(default^)
echo   --gpu, --cuda          Enable CUDA backend
echo   --vulkan               Enable Vulkan backend
echo   --metal                Enable Metal backend
echo   --clean                Remove previous build directory before building
echo   --source-release-tag TAG       Override the upstream release tag used for the source snapshot ^(default: latest release^)
echo   --dry-run              Print commands without running them
echo   --config NAME          Build configuration ^(default: Release^)
echo   --jobs N               Parallel build jobs
echo   --source-dir DIR       Override vendored source directory
echo   --build-dir DIR        Override build directory
echo   --install-include-dir  Override staged include directory
echo   --install-lib-dir DIR  Override staged library directory
echo   --example-bin-dir DIR  Override runtime staging directory
echo   --help                 Show this help message
exit /b 0

:run
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build-stable-diffusion.ps1" %PS_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
