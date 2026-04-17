@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run-tests.ps1" %*

endlocal
