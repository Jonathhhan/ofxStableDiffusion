@echo off
REM Run performance benchmarks on Windows

setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..

cd /d "%REPO_ROOT%"

echo Performance Benchmark Runner
echo ============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is required for benchmarking
    exit /b 1
)

REM Check if tests are built
if not exist "tests\build" (
    echo Building tests first...
    cmake -S tests -B tests\build -DCMAKE_BUILD_TYPE=Release
    cmake --build tests\build --config Release
)

REM Run benchmark script
python scripts\run_benchmarks.py %*

echo.
echo Benchmarks complete!
echo Results saved to: benchmarks\results\

endlocal
