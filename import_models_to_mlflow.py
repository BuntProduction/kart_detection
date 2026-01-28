"""
Script pour importer les modèles existants dans MLflow
"""
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def create_model():
    """Crée l'architecture du modèle (ResNet18 avec classification binaire)"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model

def import_model(model_path: str, model_name: str = "kart_detector"):
    """Importe un modèle .pth dans MLflow"""
    print(f"\nImport de {model_path}...")
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Créer et charger le modèle
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extraire les métadonnées
    epoch = checkpoint.get('epoch', 0)
    val_acc = checkpoint.get('val_acc', 0.0)
    
    # Créer un run MLflow
    with mlflow.start_run(run_name=f"import_{Path(model_path).stem}"):
        # Logger les paramètres
        mlflow.log_param("architecture", "ResNet18")
        mlflow.log_param("num_classes", 2)
        mlflow.log_param("dropout", 0.5)
        mlflow.log_param("imported_from", model_path)
        
        # Logger les métriques
        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("val_accuracy", val_acc)
        
        # Logger le modèle PyTorch
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        print(f"✓ Modèle importé: {model_name}")
        print(f"  - Epoch: {epoch}")
        print(f"  - Val Accuracy: {val_acc:.2%}")
        print(f"  - Run ID: {mlflow.active_run().info.run_id}")

def main():
    print("=" * 60)
    print("Import des modèles existants dans MLflow")
    print("=" * 60)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"Erreur: Le dossier {models_dir} n'existe pas")
        return
    
    # Chercher tous les fichiers .pth
    model_files = list(models_dir.glob("*.pth"))
    
    if not model_files:
        print(f"Aucun modèle .pth trouvé dans {models_dir}")
        return
    
    print(f"\n{len(model_files)} modèle(s) trouvé(s):")
    for f in model_files:
        print(f"  - {f.name}")
    
    # Importer chaque modèle
    for model_file in model_files:
        try:
            import_model(str(model_file))
        except Exception as e:
            print(f"Erreur lors de l'import de {model_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("Import terminé!")
    print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
    print("=" * 60)

if __name__ == "__main__":
    main()
