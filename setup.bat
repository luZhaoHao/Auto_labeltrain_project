@echo off
chcp 65001 >nul
title Auto-Tune 环境部署工具
echo ============================================
echo   YOLOv8 Auto-Tune 一键部署脚本
echo ============================================
echo.

:: ---- 检查 conda ----
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到 conda，请先安装 Miniconda 或 Anaconda。
    echo.
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    echo 安装完成后重新运行此脚本。
    pause
    exit /b 1
)
echo [OK] 检测到 conda

:: ---- 检查已有环境 ----
call conda env list | findstr /C:"auto_tune" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [提示] 环境 auto_tune 已存在，跳过创建。
    echo        如需重建请先运行: conda env remove -n auto_tune
    goto :config
)

:: ---- 创建环境 ----
echo.
echo [1/3] 正在创建 conda 环境 (auto_tune)...
echo       此步骤需要联网下载包，预计 5~15 分钟...
echo.
call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 环境创建失败，请检查网络连接后重试。
    pause
    exit /b 1
)
echo [OK] 环境创建完成

:config
:: ---- 配置 API Key ----
echo.
echo [2/3] 检查配置文件...
if exist auto_tune\config.yaml (
    echo [提示] auto_tune\config.yaml 已存在，跳过模板复制。
) else (
    copy auto_tune\config.template.yaml auto_tune\config.yaml >nul
    echo [提示] 已从模板创建 auto_tune\config.yaml
)
echo.
echo 请编辑 auto_tune\config.yaml，填入你的 API Key：
echo   - llm.api_key     : DeepSeek API Key（必填，用于诊断分析）
echo   - vision.api_key  : 通义千问 API Key（可选，用于视觉分析）
echo.
echo 如果先跳过，也可以以后手动修改 config.yaml。
pause

:: ---- 验证安装 ----
echo.
echo [3/3] 验证安装...
call conda run -n auto_tune python -c "import ultralytics; print('ultralytics', ultralytics.__version__); import torch; print('torch', torch.__version__); import cv2; print('opencv', cv2.__version__); print('环境验证通过！')"
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 部分包导入失败，请检查环境。
) else (
    echo [OK] 环境验证通过
)

echo.
echo ============================================
echo   部署完成！
echo.
echo   启动方式: 双击 start_app.bat
echo   测试方式: run_tests.bat
echo ============================================
echo.
pause
