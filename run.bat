@echo off
echo ==========================================================
echo   Student Performance Prediction Platform (EduPredict AI)
echo   AI-Driven Academic Early Warning & Intervention System
echo   SDG 4: Quality Education
echo ==========================================================
echo.
echo Starting Web Application on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this window to stop the server.
echo.

py -3.14 app.py
if errorlevel 1 (
    python app.py
)
pause
