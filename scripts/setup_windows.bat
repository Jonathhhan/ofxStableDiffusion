@echo off
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM setup_windows.bat - One-command setup for ofxStableDiffusion on Windows.
REM
REM This script mirrors the backend flag style used by ofxGgml. It builds the
REM vendored stable-diffusion.cpp runtime and, by default, also builds the
REM example project.
REM
REM Usage:
REM   scripts\setup_windows.bat [OPTIONS]
REM
REM Options:
REM   --cpu, --cpu-only     Build CPU backend only
REM   --auto                Auto-detect GPU backends (default)
REM   --gpu, --cuda         Enable CUDA backend
REM   --vulkan              Enable Vulkan backend
REM   --use-release         Stage a pinned upstream Windows release instead of building from source
REM   --release-tag TAG     Override the upstream GitHub release tag used with --use-release
REM   --release-variant V   Choose auto, cpu, noavx, avx, avx2, avx512, or cuda12 for --use-release
REM   --skip-native         Skip native stable-diffusion build
REM   --skip-example-build  Skip example build
REM   --jobs N              Parallel build jobs (default: %NUMBER_OF_PROCESSORS%)
REM   --clean               Remove previous native build directory before building
REM   --help                Show this help message
REM ---------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "SETUP_SCRIPT=%SCRIPT_DIR%setup_addon.ps1"
set "JOBS=%NUMBER_OF_PROCESSORS%"
set "CPU_FLAG="
set "CUDA_FLAG="
set "VULKAN_FLAG="
set "AUTO_FLAG=-Auto"
set "USE_RELEASE_FLAG="
set "RELEASE_TAG_FLAG="
set "RELEASE_VARIANT_FLAG="
set "SKIP_NATIVE_FLAG="
set "SKIP_EXAMPLE_FLAG="
set "CLEAN_FLAG="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--cpu" (
    set "CPU_FLAG=-CpuOnly"
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "AUTO_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--cpu-only" (
    set "CPU_FLAG=-CpuOnly"
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "AUTO_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--auto" (
    set "CPU_FLAG="
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "AUTO_FLAG=-Auto"
    shift
    goto parse_args
)
if /i "%~1"=="--cuda" (
    set "CPU_FLAG="
    set "CUDA_FLAG=-Cuda"
    set "VULKAN_FLAG="
    set "AUTO_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--gpu" (
    set "CPU_FLAG="
    set "CUDA_FLAG=-Cuda"
    set "VULKAN_FLAG="
    set "AUTO_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--vulkan" (
    set "CPU_FLAG="
    set "CUDA_FLAG="
    set "VULKAN_FLAG=-Vulkan"
    set "AUTO_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--use-release" (
    set "USE_RELEASE_FLAG=-UseRelease"
    shift
    goto parse_args
)
if /i "%~1"=="--release-tag" (
    if "%~2"=="" (
        echo Error: --release-tag requires a value.
        exit /b 1
    )
    set "RELEASE_TAG_FLAG=-ReleaseTag ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--release-variant" (
    if "%~2"=="" (
        echo Error: --release-variant requires a value.
        exit /b 1
    )
    set "RELEASE_VARIANT_FLAG=-ReleaseVariant %~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--skip-native" (
    set "SKIP_NATIVE_FLAG=-SkipNative"
    shift
    goto parse_args
)
if /i "%~1"=="--skip-example-build" (
    set "SKIP_EXAMPLE_FLAG=-SkipExampleBuild"
    shift
    goto parse_args
)
if /i "%~1"=="--jobs" (
    if "%~2"=="" (
        echo Error: --jobs requires a value.
        exit /b 1
    )
    set "JOBS=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--clean" (
    set "CLEAN_FLAG=-Clean"
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
echo Error: Unknown option: %~1
exit /b 1

:usage
echo setup_windows.bat - One-command setup for ofxStableDiffusion on Windows.
echo.
echo Usage:
echo   scripts\setup_windows.bat [OPTIONS]
echo.
echo Options:
echo   --cpu, --cpu-only     Build CPU backend only
echo   --auto                Auto-detect GPU backends ^(default^)
echo   --gpu, --cuda         Enable CUDA backend
echo   --vulkan              Enable Vulkan backend
echo   --use-release         Stage a pinned upstream Windows release instead of building from source
echo   --release-tag TAG     Override the upstream GitHub release tag used with --use-release
echo   --release-variant V   Choose auto, cpu, noavx, avx, avx2, avx512, or cuda12 for --use-release
echo   --skip-native         Skip native stable-diffusion build
echo   --skip-example-build  Skip example build
echo   --jobs N              Parallel build jobs ^(default: %NUMBER_OF_PROCESSORS%^)
echo   --clean               Remove previous native build directory before building
echo   --help                Show this help message
exit /b 0

:done_args

set "PS_ARGS=-Configuration Release -Jobs %JOBS% %CPU_FLAG% %CUDA_FLAG% %VULKAN_FLAG% %AUTO_FLAG% %USE_RELEASE_FLAG% %RELEASE_TAG_FLAG% %RELEASE_VARIANT_FLAG% %SKIP_NATIVE_FLAG% %SKIP_EXAMPLE_FLAG% %CLEAN_FLAG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%" %PS_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
