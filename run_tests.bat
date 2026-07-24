@echo off
chcp 65001 >nul
title Auto-Tune 测试运行

echo ============================================
echo   YOLOv8 Auto-Tune 测试
echo ============================================
echo.

:: 便携环境优先
set PYTHON=python
if exist "env\Scripts\python.exe" set PYTHON=env\Scripts\python.exe

:: 尝试 conda activate
call conda activate auto_tune 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [信息] 运行测试 (conda)...
    python -m pytest auto_tune\tests -v
    goto :done
)

:: conda run fallback
echo [信息] 运行测试 (conda run)...
conda run -n auto_tune python -m pytest auto_tune\tests -v 2>nul
if %ERRORLEVEL% EQU 0 goto :done

echo [信息] 尝试便携环境...
if exist "env\Scripts\python.exe" (
    env\Scripts\python.exe -m pytest auto_tune\tests -v
    goto :done
)

echo [错误] 未找到可用的 Python 环境
echo         请先运行 setup.bat 或 setup_from_package.bat

:done
echo.
if %ERRORLEVEL% EQU 0 (
    echo [OK] 测试通过
) else (
    echo [失败] 存在失败的测试
)
echo.
pause
