import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import glob
from typing import Optional, Tuple

import numpy as np

class KartDetector:
    def __init__(self, model_path, device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Charger le modèle entraîné"""
        # Créer l'architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Charger les poids
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.class_to_idx = checkpoint.get('class_to_idx')
        self.classes = checkpoint.get('classes')

        print(f"Modèle chargé depuis {model_path}")
        if 'val_acc' in checkpoint:
            print(f"Précision de validation : {checkpoint['val_acc']:.2f}%")
        if self.class_to_idx:
            print(f"Mapping classes: {self.class_to_idx}")
        else:
            print("⚠️ Mapping classes absent du checkpoint (fallback: go_kart=0, no_kart=1)")
        
        return model

    def _predict_pil(self, image: Image.Image) -> Tuple[str, float]:
        """Prédire si une image PIL (RGB) contient un kart."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor).squeeze()
            probability = float(output.item())

            # Le modèle est entraîné en binaire (sigmoid + BCELoss) sur des labels 0/1.
            # Donc la sortie est la probabilité de la classe avec index 1.
            # Pour retourner une probabilité "go_kart", on adapte selon le mapping.
            go_kart_index = None
            if isinstance(self.class_to_idx, dict) and 'go_kart' in self.class_to_idx:
                go_kart_index = int(self.class_to_idx['go_kart'])
            else:
                go_kart_index = 0

            kart_probability = probability if go_kart_index == 1 else (1.0 - probability)
            prediction = "KART DÉTECTÉ" if kart_probability > 0.5 else "PAS DE KART"

        return prediction, kart_probability

    def predict_frame(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        """Prédire depuis une frame OpenCV (BGR).

        Important: OpenCV charge les images en BGR. Si on passe directement une frame BGR
        au modèle entraîné sur des images RGB (PIL/ImageFolder), les couleurs sont inversées
        et les scores deviennent incohérents. On convertit donc BGR -> RGB ici.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame vide")

        # Import local pour éviter d'imposer cv2 si on n'utilise pas la vidéo
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        return self._predict_pil(image)
    
    def predict_image(self, image_path):
        """Prédire si une image contient un kart"""
        image = Image.open(image_path).convert('RGB')
        return self._predict_pil(image)
    
    def predict_batch(self, image_folder):
        """Prédire pour un dossier d'images"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        
        results = []
        for img_path in image_paths:
            prediction, probability = self.predict_image(img_path)
            results.append({
                'image': os.path.basename(img_path),
                'prediction': prediction,
                'probability': probability
            })
            print(f"{os.path.basename(img_path)}: {prediction} (prob: {probability:.4f})")
        
        return results

def find_latest_model(models_dir='models'):
    """Trouver le modèle le plus récent"""
    model_files = glob.glob(os.path.join(models_dir, '*.pth'))
    if not model_files:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {models_dir}")
    
    # Trier par date de modification
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        # Charger le dernier modèle entraîné
        model_path = find_latest_model()
        detector = KartDetector(model_path)
        
        # Exemples de prédiction
        test_images = [
            (r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data_binary\test\go_kart\07RZII7LXHF8.jpg", "Image avec KART"),
            (r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data_binary\test\no_kart\road_0000.jpg", "Image SANS kart (route)")
        ]
        
        for test_image, description in test_images:
            if os.path.exists(test_image):
                prediction, probability = detector.predict_image(test_image)
                print(f"\n{'='*60}")
                print(f"📸 {description}")
                print(f"Résultat: {prediction}")
                print(f"Probabilité kart: {probability:.4f}")
            else:
                print(f"Image non trouvée : {test_image}")
            
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Veuillez d'abord entraîner un modèle avec train.py")
