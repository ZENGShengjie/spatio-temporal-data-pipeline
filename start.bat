@echo off
chcp 65001 >nul
title 城市人流异常检测系统

echo ========================================
echo   城市人流时空异常检测系统
echo ========================================
echo.

:: 启动 API（后台新窗口）
echo [1/2] 启动 API 服务（端口 8000）...
start "API-后台" cmd /k "python -m week6.api.main --host 0.0.0.0 --port 8000"

:: 等待 3 秒
timeout /t 3 /nobreak >nul

:: 启动 Streamlit
echo [2/2] 启动可视化界面（端口 8501）...
start "Streamlit-后台" cmd /k "streamlit run week6/app.py --server.port 8501"

echo.
echo ========================================
echo   启动完成！
echo   API 文档:   http://localhost:8000/docs
echo   可视化界面: http://localhost:8501
echo ========================================
echo.
echo 注意：两个窗口会保持打开，请勿关闭。
echo 如需停止服务，在对应窗口按 Ctrl+C。
pause
