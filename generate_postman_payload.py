"""
Générer un payload JSON valide pour Postman à partir d'une image
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import sys
from pathlib import Path


def image_to_payload(image_path: str, output_path: str = "postman_payload.json"):
    """
    Convertir une image en payload JSON pour Postman
    
    Args:
        image_path: Chemin vers l'image
        output_path: Où sauvegarder le payload JSON
    """
    print(f"Traitement de l'image: {image_path}")
    
    # Charger l'image
    img = Image.open(image_path).convert("RGB")
    print(f"  Taille originale: {img.size}")
    
    # Transformation (même que le training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transformer l'image
    img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    print(f"  Shape après transformation: {img_tensor.shape}")
    
    # Convertir en numpy puis en liste Python
    img_array = img_tensor.numpy()
    
    # Créer le payload au format attendu par MLflow
    # IMPORTANT: Juste l'array, PAS de clé "inputs"
    payload = img_array.tolist()
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(payload, f)
    
    print(f"✓ Payload sauvegardé dans: {output_path}")
    print(f"  Taille du fichier: {Path(output_path).stat().st_size / 1024:.2f} KB")
    print(f"\nFormat du payload:")
    print(f"  - Type: List")
    print(f"  - Shape: {img_array.shape}")
    print(f"  - Min: {img_array.min():.3f}, Max: {img_array.max():.3f}")
    print(f"\n📋 Pour Postman:")
    print(f"  1. Méthode: POST")
    print(f"  2. URL: http://localhost:5001/invocations")
    print(f"  3. Headers: Content-Type: application/json")
    print(f"  4. Body: Copiez le contenu de {output_path}")
    
    return payload


def create_batch_payload(image_paths: list, output_path: str = "postman_batch_payload.json"):
    """
    Créer un payload avec plusieurs images
    
    Args:
        image_paths: Liste de chemins vers les images
        output_path: Où sauvegarder le payload JSON
    """
    print(f"Traitement de {len(image_paths)} images...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    batch_tensors = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        batch_tensors.append(img_tensor)
    
    # Stack en batch
    batch = torch.stack(batch_tensors)
    print(f"  Batch shape: {batch.shape}")
    
    # Convertir en liste
    payload = {
        "inputs": batch.numpy().tolist()
    }
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(payload, f)
    
    print(f"✓ Payload batch sauvegardé dans: {output_path}")
    return payload


def test_prediction_locally(image_path: str, model_path: str = None):
    """
    Tester la prédiction localement sans MLflow
    
    Args:
        image_path: Chemin vers l'image à tester
        model_path: Chemin vers le modèle .pth (optionnel)
    """
    from torchvision import models
    import torch.nn as nn
    
    print(f"\n🧪 Test de prédiction locale...")
    
    # Trouver le modèle
    if model_path is None:
        models_dir = Path("models")
        models_list = list(models_dir.glob("kart_detector_*.pth"))
        if not models_list:
            print("Aucun modèle trouvé!")
            return
        model_path = str(max(models_list, key=lambda p: p.stat().st_mtime))
    
    print(f"  Modèle: {model_path}")
    
    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Charger et transformer l'image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(img_tensor).squeeze().item()
    
    # Interpréter
    class_to_idx = checkpoint.get('class_to_idx', {'go_kart': 0, 'no_kart': 1})
    go_kart_index = class_to_idx.get('go_kart', 0)
    
    if go_kart_index == 1:
        prob_kart = output
    else:
        prob_kart = 1 - output
    
    print(f"\n  🏎️  Probabilité KART: {prob_kart*100:.2f}%")
    print(f"  ❌ Probabilité NO_KART: {(1-prob_kart)*100:.2f}%")
    
    if prob_kart > 0.5:
        print(f"\n  ✓ KART DÉTECTÉ! (confiance: {prob_kart*100:.1f}%)")
    else:
        print(f"\n  ✗ PAS DE KART (confiance: {(1-prob_kart)*100:.1f}%)")
    
    return prob_kart


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_postman_payload.py <image_path> [output.json]")
        print("\nExemples:")
        print("  python generate_postman_payload.py data/test/go_kart/image1.jpg")
        print("  python generate_postman_payload.py my_image.jpg my_payload.json")
        
        # Essayer de trouver une image de test
        test_paths = [
            Path("data_enhanced/test/go_kart"),
            Path("data/test/go_kart"),
            Path("extracted_empty_circuit")
        ]
        
        for test_dir in test_paths:
            if test_dir.exists():
                images = list(test_dir.glob("*.jpg"))
                if images:
                    print(f"\n🧪 Test avec une image d'exemple:")
                    img_path = str(images[0])
                    image_to_payload(img_path)
                    test_prediction_locally(img_path)
                    break
    else:
        img_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "postman_payload.json"
        
        # Générer le payload
        image_to_payload(img_path, output_path)
        
        # Test local
        test_prediction_locally(img_path)
