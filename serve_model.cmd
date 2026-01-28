@echo off
REM Script pour servir un modèle MLflow
REM Usage: serve_model.cmd [model_uri]
REM Exemple: serve_model.cmd "models:/kart_detector/latest"

if "%~1"=="" (
    echo Usage: serve_model.cmd [model_uri]
    echo Exemple: serve_model.cmd "models:/kart_detector/latest"
    echo.
    echo Utilisation du modèle par défaut: models:/kart_detector/latest
    set MODEL_URI=models:/kart_detector/latest
) else (
    set MODEL_URI=%~1
)

echo Démarrage du serveur de modèle MLflow...
echo Modèle: %MODEL_URI%
echo Port: 5001
echo.
mlflow models serve -m "%MODEL_URI%" -p 5001 --no-conda
