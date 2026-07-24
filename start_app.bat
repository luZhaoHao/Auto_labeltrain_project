@echo off
chcp 65001 >nul
title Auto-Tune 服务器

echo ============================================
echo   YOLOv8 Auto-Tune Dashboard
echo ============================================
echo.

:: ---- 启动优先级: 便携环境 > conda activate > conda run ----
if exist "env\Scripts\python.exe" (
    echo [信息] 使用便携环境启动...
    env\Scripts\python.exe -m auto_tune.main
    goto :check_exit
)

call conda activate auto_tune 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [信息] 使用 conda activate 启动...
    python -m auto_tune.main
    goto :check_exit
)

echo [信息] 使用 conda run 启动...
conda run -n auto_tune python -m auto_tune.main

:check_exit
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 服务器异常退出，错误码: %ERRORLEVEL%
    echo.
    pause
)
