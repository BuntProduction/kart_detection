"""
Wrapper MLflow personnalisé pour le modèle Kart Detector
Permet d'accepter des images et de retourner des prédictions
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import mlflow
from mlflow.pyfunc import PythonModel
import numpy as np
from PIL import Image
import io
import base64


class KartDetectorWrapper(PythonModel):
    """Wrapper pour le modèle de détection de kart"""
    
    def load_context(self, context):
        """Charger le modèle PyTorch"""
        import torch
        import torch.nn as nn
        from torchvision import models
        
        # Charger le checkpoint
        model_path = context.artifacts["model_path"]
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Reconstruire le modèle
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Charger les poids
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Métadonnées
        self.class_to_idx = checkpoint.get('class_to_idx', {'go_kart': 0, 'no_kart': 1})
        self.go_kart_index = self.class_to_idx.get('go_kart', 0)
    
    def predict(self, context, model_input):
        """
        Prédiction sur des images
        
        Input peut être:
        - numpy array de shape (batch, 3, 224, 224) normalisé
        - pandas DataFrame avec colonne 'image' contenant des arrays
        """
        import pandas as pd
        
        # Convertir en tensor
        if isinstance(model_input, pd.DataFrame):
            # Si c'est un DataFrame, extraire les images
            if 'image' in model_input.columns:
                images = np.stack(model_input['image'].values)
            else:
                # Prendre toutes les colonnes comme une seule image
                images = model_input.values.reshape(-1, 3, 224, 224)
        else:
            # C'est déjà un numpy array
            images = model_input
        
        # S'assurer que c'est le bon shape
        if len(images.shape) == 3:
            images = images.reshape(1, 3, 224, 224)
        elif len(images.shape) == 2:
            # Cas où les données sont aplaties
            images = images.reshape(-1, 3, 224, 224)
        
        # Convertir en tensor
        images_tensor = torch.from_numpy(images).float().to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(images_tensor).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # Convertir en probabilité go_kart
            if self.go_kart_index == 1:
                predictions = outputs.cpu().numpy()
            else:
                predictions = (1 - outputs).cpu().numpy()
        
        return predictions


def register_model_with_wrapper(model_path: str, model_name: str = "kart_detector"):
    """
    Enregistrer le modèle avec le wrapper personnalisé dans MLflow
    
    Args:
        model_path: Chemin vers le fichier .pth
        model_name: Nom du modèle dans MLflow
    """
    print(f"Enregistrement du modèle avec wrapper personnalisé...")
    print(f"Modèle: {model_path}")
    
    # Configuration MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("kart_detector")
    
    # Charger le checkpoint pour obtenir les métadonnées
    checkpoint = torch.load(model_path, map_location='cpu')
    
    with mlflow.start_run(run_name=f"wrapper_{model_path.split('/')[-1]}"):
        # Logger les paramètres
        mlflow.log_params({
            "model_type": "kart_detector_wrapped",
            "model_architecture": "ResNet18",
            "wrapped": True,
            "source_model": model_path
        })
        
        # Logger les métriques si disponibles
        if 'val_acc' in checkpoint:
            mlflow.log_metrics({
                "val_acc": checkpoint['val_acc'],
                "epoch": checkpoint.get('epoch', 0)
            })
        
        # Enregistrer avec le wrapper
        artifacts = {
            "model_path": model_path
        }
        
        # Définir la signature (inputs: array de shape [batch, 3, 224, 224])
        from mlflow.models.signature import infer_signature
        import pandas as pd
        
        # Exemple d'input
        example_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        example_output = np.array([0.5])
        
        signature = infer_signature(example_input, example_output)
        
        # Log le modèle avec le wrapper
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=KartDetectorWrapper(),
            artifacts=artifacts,
            signature=signature,
            registered_model_name=model_name,
            pip_requirements=[
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "Pillow>=9.0.0",
                "numpy>=1.24.0",
            ]
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"✓ Modèle enregistré avec succès!")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: models:/{model_name}/latest")
    
    return run_id


if __name__ == "__main__":
    from pathlib import Path
    
    # Enregistrer le meilleur modèle
    models_dir = Path(__file__).parent / "models"
    best_model = models_dir / "kart_detector_epoch_7_acc_99.20.pth"
    
    if best_model.exists():
        print(f"Enregistrement du modèle: {best_model.name}")
        register_model_with_wrapper(str(best_model), "kart_detector")
    else:
        print(f"Modèle non trouvé: {best_model}")
        # Essayer de trouver n'importe quel modèle
        models = list(models_dir.glob("kart_detector_*.pth"))
        if models:
            best = max(models, key=lambda p: p.stat().st_mtime)
            print(f"Utilisation du modèle: {best.name}")
            register_model_with_wrapper(str(best), "kart_detector")
        else:
            print("Aucun modèle trouvé!")
