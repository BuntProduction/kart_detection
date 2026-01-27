import json
import matplotlib.pyplot as plt
import os

def plot_training_history(history_path='models/training_history.json'):
    """Visualiser l'historique d'entraînement"""
    
    # Charger l'historique
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Époques', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Époques', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = 'models/training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphiques sauvegardés : {output_path}")
    
    # plt.show()  # Commenté pour éviter d'ouvrir la fenêtre
    plt.close()  # Fermer la figure pour libérer la mémoire
    
    # Afficher les stats finales
    print("\n" + "="*50)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*50)
    print(f"Meilleure validation accuracy : {max(history['val_acc']):.2f}%")
    print(f"Loss finale (train) : {history['train_loss'][-1]:.4f}")
    print(f"Loss finale (val) : {history['val_loss'][-1]:.4f}")
    print(f"Accuracy finale (train) : {history['train_acc'][-1]:.2f}%")
    print(f"Accuracy finale (val) : {history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    history_file = 'models/training_history.json'
    
    if os.path.exists(history_file):
        plot_training_history(history_file)
    else:
        print(f"Fichier d'historique non trouvé : {history_file}")
        print("Veuillez d'abord entraîner un modèle avec train.py")
