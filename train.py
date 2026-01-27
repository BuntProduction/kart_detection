import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
from tqdm import tqdm
import json
from pathlib import Path

# Configuration
class Config:
    # Chemins
    PROJECT_DIR = Path(__file__).resolve().parent
    # Par défaut: dataset généré localement via prepare_enhanced_dataset.py
    DATA_DIR = Path(os.environ.get('KART_DATA_DIR', PROJECT_DIR / 'data_enhanced'))
    MODEL_SAVE_PATH = str(PROJECT_DIR / 'models')
    
    # Hyperparamètres
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    NUM_CLASSES = 2  # kart / no_kart (binaire)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_transforms():
    """Définir les transformations pour l'augmentation de données"""
    train_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def load_datasets():
    """Charger les datasets"""
    train_transforms, val_transforms = get_data_transforms()

    data_dir = str(Config.DATA_DIR)
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_transforms
    )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset):
    """Créer les data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_model():
    """Créer le modèle (ResNet18 pré-entraîné)"""
    model = models.resnet18(pretrained=True)
    
    # Geler les premières couches
    for param in model.parameters():
        param.requires_grad = False
    
    # Remplacer la dernière couche pour la classification binaire
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    model = model.to(Config.DEVICE)
    return model

def train_epoch(model, train_loader, criterion, optimizer):
    """Entraîner pour une époque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(Config.DEVICE)
        labels = labels.float().to(Config.DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Valider le modèle"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(Config.DEVICE)
            labels = labels.float().to(Config.DEVICE)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def save_model(model, epoch, val_acc, optimizer, history, class_to_idx=None, classes=None):
    """Sauvegarder le modèle"""
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'class_to_idx': class_to_idx,
        'classes': classes,
    }
    
    save_path = os.path.join(Config.MODEL_SAVE_PATH, f'kart_detector_epoch_{epoch}_acc_{val_acc:.2f}.pth')
    torch.save(checkpoint, save_path)
    print(f"Modèle sauvegardé : {save_path}")

def train_model():
    """Fonction principale d'entraînement"""
    print(f"Device utilisé : {Config.DEVICE}")
    print(f"Chargement des données...")
    
    # Charger les données
    train_dataset, val_dataset, test_dataset = load_datasets()
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)
    
    print(f"Nombre d'images d'entraînement : {len(train_dataset)}")
    print(f"Nombre d'images de validation : {len(val_dataset)}")
    print(f"Nombre d'images de test : {len(test_dataset)}")
    print(f"Classes : {train_dataset.classes}")
    print(f"class_to_idx : {train_dataset.class_to_idx}")
    
    # Créer le modèle
    model = create_model()
    
    # Loss et optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\nDébut de l'entraînement...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nÉpoque {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Scheduler
        scheduler.step(val_acc)
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                epoch + 1,
                val_acc,
                optimizer,
                history,
                class_to_idx=train_dataset.class_to_idx,
                classes=train_dataset.classes,
            )
            print(f"✓ Nouveau meilleur modèle sauvegardé!")
    
    # Évaluation finale sur le test set
    print("\n" + "="*50)
    print("Évaluation finale sur le test set...")
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Sauvegarder l'historique
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\nEntraînement terminé!")
    return model, history

if __name__ == "__main__":
    model, history = train_model()
