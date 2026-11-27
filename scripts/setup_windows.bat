@echo off
REM Windows convenience wrapper around scripts\setup_windows.ps1
REM Bypasses the default PowerShell execution policy for this one invocation only.
setlocal
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
if errorlevel 1 (
    echo.
    echo Setup failed. See the error above.
    exit /b 1
)
endlocal
