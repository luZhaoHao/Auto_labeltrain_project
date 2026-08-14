@echo off
chcp 65001 >nul
title Auto-Tune 打包工具 (conda-pack)

echo ============================================
echo   YOLOv8 Auto-Tune 离线打包脚本
echo   方案：conda-pack (全量环境打包)
echo ============================================
echo.

set ROOT=%~dp0
set OUTPUT=%ROOT%build_output
set PACKAGE_DIR=%OUTPUT%\AutoTune_Package

:: ---- 自动检测 conda 路径 ----
set CONDA_CMD=conda
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [信息] conda 不在 PATH 中，正在搜索常见安装路径...
    set CONDA_CMD=
    if exist "D:\Program Files\anaconda3\Scripts\conda.exe" set "CONDA_CMD=D:\Program Files\anaconda3\Scripts\conda.exe"
    if exist "D:\Program Files\Anaconda3\Scripts\conda.exe" set "CONDA_CMD=D:\Program Files\Anaconda3\Scripts\conda.exe"
    if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramData%\anaconda3\Scripts\conda.exe"
    if exist "%ProgramFiles%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%ProgramFiles%\anaconda3\Scripts\conda.exe"
    if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%USERPROFILE%\anaconda3\Scripts\conda.exe"
    if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_CMD=%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"
    if not defined CONDA_CMD (
        echo [错误] 找不到 conda！
        echo.
        echo       请尝试以下方法之一：
        echo       1. 从开始菜单打开 "Anaconda Prompt"，然后运行此脚本
        echo       2. 手动设置 CONDA_CMD 环境变量指向 conda.exe
        echo.
        pause
        exit /b 1
    )
    echo [OK] 找到 conda: %CONDA_CMD%
)

:: 清理上次构建
if exist "%OUTPUT%" rmdir /s /q "%OUTPUT%"
mkdir "%OUTPUT%"
mkdir "%PACKAGE_DIR%"

:: ---- Step 1: 检查 conda-pack ----
echo [1/4] 检查 conda-pack...
"%CONDA_CMD%" run -n auto_tune pip show conda-pack >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [信息] 正在安装 conda-pack...
    "%CONDA_CMD%" install -n auto_tune conda-pack -y
    if %ERRORLEVEL% NEQ 0 (
        echo [错误] conda-pack 安装失败
        pause
        exit /b 1
    )
)
echo [OK] conda-pack 就绪

:: ---- Step 2: 打包 conda 环境 ----
echo [2/4] 正在打包 conda 环境 (auto_tune)...
echo       此步骤耗时 1~3 分钟，生成的包约 5~15 GB...
"%CONDA_CMD%" pack -n auto_tune -o "%PACKAGE_DIR%\auto_tune_env.tar.gz" --force
if %ERRORLEVEL% NEQ 0 (
    echo [错误] conda pack 失败
    pause
    exit /b 1
)
echo [OK] 环境打包完成

:: ---- Step 3: 复制项目源码 ----
echo [3/4] 正在复制项目源码...
xcopy "%ROOT%auto_tune" "%PACKAGE_DIR%\auto_tune\" /E /I /Y /EXCLUDE:"%OUTPUT%\exclude.txt" >nul 2>&1 || (
    :: 手动排除无需 xcopy /EXCLUDE 文件，先全量再删除
    xcopy "%ROOT%auto_tune" "%PACKAGE_DIR%\auto_tune\" /E /I /Y /Q >nul
)

:: 删除缓存和不需要的文件
if exist "%PACKAGE_DIR%\auto_tune\__pycache__" rmdir /s /q "%PACKAGE_DIR%\auto_tune\__pycache__"
for /d /r "%PACKAGE_DIR%\auto_tune" %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
for /r "%PACKAGE_DIR%\auto_tune" %%f in (*.pyc) do del "%%f" 2>nul

:: 复制环境配置和启动脚本
copy "%ROOT%environment.yml" "%PACKAGE_DIR%\" >nul
copy "%ROOT%setup.bat" "%PACKAGE_DIR%\" >nul
copy "%ROOT%start_app.bat" "%PACKAGE_DIR%\" >nul

echo [OK] 源码复制完成

:: ---- Step 4: 创建解包部署脚本 ----
echo [4/4] 创建部署脚本...
(
echo @echo off
echo chcp 65001 ^>nul
echo title Auto-Tune 部署工具
echo.
echo echo ============================================
echo echo   YOLOv8 Auto-Tune 离线部署
echo echo ============================================
echo echo.
echo :: ---- 1. 解压 conda 环境 ----
echo if exist "env\Scripts\python.exe" (
echo     echo [OK] 环境已解压，跳过
echo     goto :config
echo )
echo.
echo echo [1/3] 正在解压 conda 环境 ^(约 5~15 GB，需 3~10 分钟^)...
echo echo        请确保有足够的磁盘空间。
echo.
echo if not exist "auto_tune_env.tar.gz" (
echo     echo [错误] 找不到 auto_tune_env.tar.gz
echo     pause
echo     exit /b 1
echo )
echo.
echo :: Windows 10+ 自带 tar，否则需要 7-Zip
echo tar -xzf auto_tune_env.tar.gz 2^>nul
echo if exist "env\Scripts\python.exe" goto :relocate
echo.
echo :: 用 7-Zip 解压两次 (tar.gz -^> tar -^> dir)
echo if exist "env" rmdir /s /q "env"
echo mkdir env_temp
echo "%ProgramFiles%\7-Zip\7z.exe" x auto_tune_env.tar.gz -so 2^>nul ^| "%ProgramFiles%\7-Zip\7z.exe" x -aoa -si -tenv\ 2^>nul
echo if errorlevel 1 (
echo     echo [警告] 自动解压失败，请手动操作：
echo     echo   1. 用 7-Zip 解压 auto_tune_env.tar.gz 到 env\ 目录
echo     echo   2. 重新运行此脚本
echo     pause
echo     exit /b 1
echo )
echo.
echo :relocate
echo :: conda-pack 首次运行会修复路径
echo echo [信息] 正在修复环境路径 ^(首次运行需要^)...
echo env\Scripts\python.exe -c "print('路径修复完成')" 2^>nul
echo echo [OK] 环境解压完成
echo.
echo :config
echo :: ---- 2. 配置文件 ----
echo echo [2/3] 检查配置文件...
echo if exist "auto_tune\config.yaml" (
echo     echo [提示] auto_tune\config.yaml 已存在
echo ) else (
echo     copy auto_tune\config.template.yaml auto_tune\config.yaml ^>nul
echo     echo [提示] 已从模板创建 auto_tune\config.yaml
echo )
echo.
echo echo ============================================
echo echo   请编辑 auto_tune\config.yaml，填入：
echo echo   - llm.api_key     : DeepSeek API Key
echo echo   - vision.api_key  : 通义千问 API Key
echo echo ============================================
echo pause
echo.
echo :: ---- 3. 验证 ----
echo echo [3/3] 验证安装...
echo env\Scripts\python.exe -c "import ultralytics; print('ultralytics', ultralytics.__version__); import cv2; print('opencv', cv2.__version__); import torch; print('torch', torch.__version__); print('验证通过！')"
echo.
echo if errorlevel 1 (
echo     echo [警告] 验证失败，请检查环境
echo     pause
echo     exit /b 1
echo )
echo.
echo echo ============================================
echo echo   部署完成！
echo echo.
echo echo   启动方式: 双击 start_app.bat
echo echo ============================================
echo pause
) > "%PACKAGE_DIR%\setup_from_package.bat"

echo [OK] 部署脚本创建完成

:: ---- 创建 README ----
(
echo Auto-Tune 离线部署包使用说明
echo ===============================
echo.
echo 第一步：解压本包到目标电脑的任意目录（路径不要有中文和空格）。
echo.
echo 第二步：双击运行 setup_from_package.bat
echo          - 自动解压 conda 环境到 env\ 目录
echo          - 创建配置文件模板
echo.
echo 第三步：用记事本打开 auto_tune\config.yaml
echo          填入你的 API Key：
echo          - llm.api_key: DeepSeek API Key
echo          - vision.api_key: 通义千问 API Key
echo.
echo 第四步：双击 start_app.bat 启动 Web UI
echo          浏览器访问 http://127.0.0.1:8000
echo.
echo ===============================
echo 注意：
echo - 总大小约 5~15 GB，请确保目标磁盘有足够空间
echo - 首次解压需要 3~10 分钟
echo - 如果目标电脑有 NVIDIA GPU，会自动使用 CUDA 加速
echo - 启动后会自动在浏览器打开 Web 界面
) > "%PACKAGE_DIR%\README.txt"

echo [OK] README 创建完成

:: ---- 打包成 ZIP ----
echo.
echo [可选] 正在打包为 zip...
if exist "%OUTPUT%\AutoTune_Package.zip" del "%OUTPUT%\AutoTune_Package.zip"
powershell -Command "Add-Type -Assembly 'System.IO.Compression.FileSystem'; [System.IO.Compression.ZipFile]::CreateFromDirectory('%PACKAGE_DIR%', '%OUTPUT%\AutoTune_Package.zip', 'None', '')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] 打包完成: build_output\AutoTune_Package.zip
) else (
    echo [信息] zip 打包跳过，源码目录在: %PACKAGE_DIR%
)

echo.
echo ============================================
echo   打包完成！
echo.
echo   输出目录: %OUTPUT%
echo   包内容:
echo     - auto_tune_env.tar.gz  (conda 环境，约 5-15 GB)
echo     - auto_tune\            (项目源码)
echo     - setup_from_package.bat (目标电脑部署脚本)
echo     - start_app.bat         (启动脚本)
echo     - README.txt            (说明文档)
echo ============================================
echo.
pause
