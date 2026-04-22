@echo off
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM setup_windows.bat - One-command setup for ofxStableDiffusion on Windows.
REM
REM This script mirrors the backend flag style used by ofxGgml and builds the
REM vendored stable-diffusion.cpp runtime.
REM
REM Usage:
REM   scripts\setup_windows.bat [OPTIONS]
REM
REM Options:
REM   --cpu, --cpu-only     Build CPU backend only (default)
REM   --gpu, --cuda         Enable CUDA backend
REM   --vulkan              Enable Vulkan backend
REM   --metal               Enable Metal backend where supported
REM   --all                 Build every available backend variant and leave the best one active
REM   --select-backend NAME Select an already staged backend variant ^(cpu-only, cuda, vulkan, metal^)
REM   --ggml-release-tag TAG         Override the upstream ggml release tag used for source builds (default: latest release)
REM   --skip-native         Skip native stable-diffusion build
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
set "METAL_FLAG="
set "ALL_FLAG="
set "SELECT_BACKEND_FLAG="
set "GGML_RELEASE_TAG_FLAG="
set "SKIP_NATIVE_FLAG="
set "CLEAN_FLAG="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--cpu" (
    set "CPU_FLAG=-CpuOnly"
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "METAL_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--cpu-only" (
    set "CPU_FLAG=-CpuOnly"
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "METAL_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--cuda" (
    set "CPU_FLAG="
    set "CUDA_FLAG=-Cuda"
    set "VULKAN_FLAG="
    set "METAL_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--gpu" (
    set "CPU_FLAG="
    set "CUDA_FLAG=-Cuda"
    set "VULKAN_FLAG="
    set "METAL_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--vulkan" (
    set "CPU_FLAG="
    set "CUDA_FLAG="
    set "VULKAN_FLAG=-Vulkan"
    set "METAL_FLAG="
    shift
    goto parse_args
)
if /i "%~1"=="--metal" (
    set "CPU_FLAG="
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "METAL_FLAG=-Metal"
    shift
    goto parse_args
)
if /i "%~1"=="--all" (
    set "CPU_FLAG="
    set "CUDA_FLAG="
    set "VULKAN_FLAG="
    set "METAL_FLAG="
    set "ALL_FLAG=-All"
    shift
    goto parse_args
)
if /i "%~1"=="--select-backend" (
    if "%~2"=="" (
        echo Error: --select-backend requires a value.
        exit /b 1
    )
    set "SELECT_BACKEND_FLAG=-SelectBackend ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--ggml-release-tag" (
    if "%~2"=="" (
        echo Error: --ggml-release-tag requires a value.
        exit /b 1
    )
    set "GGML_RELEASE_TAG_FLAG=-GgmlReleaseTag ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--skip-native" (
    set "SKIP_NATIVE_FLAG=-SkipNative"
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
echo   --cpu, --cpu-only     Build CPU backend only ^(default^)
echo   --gpu, --cuda         Enable CUDA backend
echo   --vulkan              Enable Vulkan backend
echo   --metal               Enable Metal backend where supported
echo   --all                 Build every available backend variant and leave the best one active
echo   --select-backend NAME Select an already staged backend variant ^(cpu-only, cuda, vulkan, metal^)
echo   --ggml-release-tag TAG         Override the upstream ggml release tag used for source builds ^(default: latest release^)
echo   --skip-native         Skip native stable-diffusion build
echo   --jobs N              Parallel build jobs ^(default: %NUMBER_OF_PROCESSORS%^)
echo   --clean               Remove previous native build directory before building
echo   --help                Show this help message
exit /b 0

:done_args

set "PS_ARGS=-Configuration Release -Jobs %JOBS% %CPU_FLAG% %CUDA_FLAG% %VULKAN_FLAG% %METAL_FLAG% %ALL_FLAG% %SELECT_BACKEND_FLAG% %GGML_RELEASE_TAG_FLAG% %SKIP_NATIVE_FLAG% %CLEAN_FLAG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%" %PS_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
