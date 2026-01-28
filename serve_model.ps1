param(
    [string]$ModelUri = "models:/kart_detector/latest"
)

Write-Host "Démarrage du serveur de modèle MLflow..." -ForegroundColor Green
Write-Host "Modèle: $ModelUri" -ForegroundColor Cyan
Write-Host "Port: 5001" -ForegroundColor Cyan
Write-Host ""

mlflow models serve -m "$ModelUri" -p 5001 --no-conda
