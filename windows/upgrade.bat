@echo off
rem Auto-Tune Studio - Windows upgrade entry point (F1.2-D).
rem Double-click entry only: it locates the PowerShell script next to itself with
rem %~dp0 and returns that script's exit code. Run it from the newly extracted package: the program files come from this folder.
rem On failure the window stays open; set AUTO_TUNE_NO_PAUSE=1 to suppress that
rem in an automated run.
setlocal

set "SCRIPT=%~dp0upgrade.ps1"

if not exist "%SCRIPT%" (
    echo [ERROR] %SCRIPT% not found.
    echo Please extract the whole delivery package before running this file.
    set "EXITCODE=1"
    goto finish
)

if not exist "%~dp0package-manifest.json" (
    echo [ERROR] This folder is not a delivery package: package-manifest.json is missing.
    set "EXITCODE=1"
    goto finish
)

powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "EXITCODE=%ERRORLEVEL%"

:finish
if not "%EXITCODE%"=="0" if not defined AUTO_TUNE_NO_PAUSE pause
exit /b %EXITCODE%
