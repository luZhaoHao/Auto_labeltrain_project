@echo off
rem Auto-Tune Studio - Windows install entry point (F1.2-D).
rem Double-click entry only: it locates the PowerShell script next to itself with
rem %~dp0 and returns that script's exit code. All installation logic lives in install.ps1.
rem Deliberately *not* -NonInteractive: install.ps1 asks a first-time operator for
rem the install directory, and PowerShell refuses Read-Host in a non-interactive
rem host. Automation answers through -InstallRoot, -AcceptRecommended or
rem AUTO_TUNE_NONINTERACTIVE instead of the prompt.
rem On failure the window stays open; set AUTO_TUNE_NO_PAUSE=1 to suppress that
rem in an automated run.
setlocal

set "SCRIPT=%~dp0install.ps1"

if not exist "%SCRIPT%" (
    echo [ERROR] %SCRIPT% not found.
    echo Please extract the whole delivery package before running this file.
    set "EXITCODE=1"
    goto finish
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "EXITCODE=%ERRORLEVEL%"

:finish
if not "%EXITCODE%"=="0" if not defined AUTO_TUNE_NO_PAUSE pause
exit /b %EXITCODE%
