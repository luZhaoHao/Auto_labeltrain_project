@echo off
setlocal
cd /d "%~dp0"
if not exist "log" mkdir "log"
python start_server.py > "log\server_log.txt" 2>&1
