@echo off
rem Auto-Tune Studio - Windows uninstall entry point (F1.2-D).
rem Double-click entry only: it locates the PowerShell script next to itself with
rem %~dp0 and returns that script's exit code. Default: remove the program, the
rem private runtime, the launcher and the desktop shortcut, keep the user data.
rem On failure the window stays open; set AUTO_TUNE_NO_PAUSE=1 to suppress that
rem in an automated run.
setlocal

rem Deleting the user data must be asked for explicitly:
rem     uninstall.bat --remove-data --confirm
rem Every other argument is passed to the PowerShell layer untouched, exactly
rem like install / start / upgrade do.
set "SCRIPT=%~dp0uninstall.ps1"

if not exist "%SCRIPT%" (
    echo [ERROR] %SCRIPT% not found.
    echo Please extract the whole delivery package before running this file.
    set "EXITCODE=1"
    goto finish
)

set "REMOVE_DATA="
set "CONFIRM="
set "ARGS="

:parse_arguments
if "%~1"=="" goto run_uninstall
set "ARG=%~1"
if /i "%ARG%"=="--remove-data" set "REMOVE_DATA=-RemoveData"
if /i "%ARG%"=="-RemoveData" set "REMOVE_DATA=-RemoveData"
if /i "%ARG%"=="--confirm" set "CONFIRM=-Confirm"
if /i "%ARG%"=="-Confirm" set "CONFIRM=-Confirm"
if /i not "%ARG%"=="--remove-data" if /i not "%ARG%"=="-RemoveData" if /i not "%ARG%"=="--confirm" if /i not "%ARG%"=="-Confirm" set "ARGS=%ARGS% "%ARG%""
shift
goto parse_arguments

:run_uninstall
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%SCRIPT%" %REMOVE_DATA% %CONFIRM% %ARGS%
set "EXITCODE=%ERRORLEVEL%"

:finish
if not "%EXITCODE%"=="0" if not defined AUTO_TUNE_NO_PAUSE pause
exit /b %EXITCODE%
