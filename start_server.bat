@echo off
cd /d e:\dataprocess_modeltrain\Auto_labeltrain_project
"D:\Program Files\anaconda3\envs\auto_tune\python.exe" -c "import logging; logging.basicConfig(level=logging.INFO, format='%%(asctime)s [%%(name)s] %%(levelname)s: %%(message)s', force=True); from auto_tune.ui.app import start_server; start_server()" > log\server_log.txt 2>&1
