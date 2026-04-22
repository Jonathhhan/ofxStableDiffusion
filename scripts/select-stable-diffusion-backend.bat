@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PS_ARGS="

:parse_args
if "%~1"=="" goto run
if /i "%~1"=="--cpu" (
    set "PS_ARGS=!PS_ARGS! -Backend cpu-only"
    shift
    goto parse_args
)
if /i "%~1"=="--cpu-only" (
    set "PS_ARGS=!PS_ARGS! -Backend cpu-only"
    shift
    goto parse_args
)
if /i "%~1"=="--cuda" (
    set "PS_ARGS=!PS_ARGS! -Backend cuda"
    shift
    goto parse_args
)
if /i "%~1"=="--gpu" (
    set "PS_ARGS=!PS_ARGS! -Backend cuda"
    shift
    goto parse_args
)
if /i "%~1"=="--vulkan" (
    set "PS_ARGS=!PS_ARGS! -Backend vulkan"
    shift
    goto parse_args
)
if /i "%~1"=="--metal" (
    set "PS_ARGS=!PS_ARGS! -Backend metal"
    shift
    goto parse_args
)
if /i "%~1"=="--backend" (
    if "%~2"=="" (
        echo Error: --backend requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -Backend ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--variant-root-dir" (
    if "%~2"=="" (
        echo Error: --variant-root-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -VariantRootDir ""%~2"""
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
if /i "%~1"=="--install-ggml-include-dir" (
    if "%~2"=="" (
        echo Error: --install-ggml-include-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -InstallGgmlIncludeDir ""%~2"""
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--install-ggml-lib-dir" (
    if "%~2"=="" (
        echo Error: --install-ggml-lib-dir requires a value.
        exit /b 1
    )
    set "PS_ARGS=!PS_ARGS! -InstallGgmlLibDir ""%~2"""
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
echo Error: Unknown option: %~1
exit /b 1

:usage
echo select-stable-diffusion-backend.bat - Install one staged backend variant into the canonical addon paths.
echo.
echo Usage:
echo   scripts\select-stable-diffusion-backend.bat [OPTIONS]
echo.
echo Options:
echo   --cpu, --cpu-only         Select the cpu-only variant
echo   --gpu, --cuda             Select the cuda variant
echo   --vulkan                  Select the vulkan variant
echo   --metal                   Select the metal variant
echo   --backend NAME            Explicit backend name ^(cpu-only, cuda, vulkan, metal^)
echo   --variant-root-dir DIR    Override the variant snapshot root
echo   --install-include-dir DIR Override staged stable-diffusion include dir
echo   --install-lib-dir DIR     Override staged stable-diffusion lib dir
echo   --install-ggml-include-dir DIR Override staged ggml include dir
echo   --install-ggml-lib-dir DIR Override staged ggml lib dir
echo   --example-bin-dir DIR     Override example runtime staging dir
echo   --help                    Show this help message
exit /b 0

:run
if "!PS_ARGS!"=="" (
    echo Error: choose a backend with --cpu-only, --cuda, --vulkan, --metal, or --backend NAME.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%select-stable-diffusion-backend.ps1" %PS_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
