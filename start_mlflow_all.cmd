@echo off
echo ============================================
echo Demarrage MLflow
echo ============================================
echo.

cd /d "%~dp0"

echo Nettoyage des ports...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5001" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo OK
echo.

echo Demarrage MLflow Server (port 5000)...
start "MLflow Server" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && set MLFLOW_HOST=0.0.0.0 && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000"
timeout /t 12 /nobreak >nul
echo OK
echo.

echo Demarrage Model Server (port 5001)...
start "Model Server" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && mlflow models serve -m models:/kart_detector/2 -p 5001 --no-conda"
timeout /t 15 /nobreak >nul
echo OK
echo.

echo ============================================
echo Serveurs actifs:
echo   MLflow UI:  http://localhost:5000
echo   Model API:  http://localhost:5001/invocations
echo ============================================
echo.
echo Pour ngrok (IMPORTANT - utilisez cette commande exacte):
echo   ngrok http 5000 --host-header="localhost:5000"
echo.
pause
