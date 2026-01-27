import os
import shutil
from pathlib import Path
import random

def prepare_enhanced_dataset():
    """
    Préparer un dataset enrichi avec:
    - Images de karts (classe: go_kart)
    - Images de routes génériques (classe: no_kart)
    - Images de circuit vide extraites (classe: no_kart)
    - Images de vélos (classe: no_kart)
    - Images no_kart ajoutées manuellement (classe: no_kart)
    """
    
    project_dir = Path(__file__).resolve().parent

    # Chemins source (relatifs au projet)
    kart_source = project_dir / 'data'
    road_broken = project_dir / 'Road Classification' / 'Broken'
    road_not_broken = project_dir / 'Road Classification' / 'Not Broken'
    circuit_empty = project_dir / 'extracted_empty_circuit'
    bicycle_dir = project_dir / 'bicycle'
    manual_no_kart_dir = project_dir / 'No kart'

    # Chemin destination (dans le projet)
    output_dir = project_dir / 'data_enhanced'
    
    print("🔧 PRÉPARATION DU DATASET ENRICHI")
    print("=" * 70)
    print("Sources:")
    print("  • Karts: Dataset original")
    print("  • Routes génériques: Broken + Not Broken")
    print("  • Circuit vide: Frames extraites de la vidéo")
    print("  • Vélos: Dataset complémentaire")
    print(f"  • No_kart manuel: {manual_no_kart_dir.name}")
    print("=" * 70)
    
    # Créer la structure de dossiers
    for split in ['train', 'val', 'test']:
        for class_name in ['go_kart', 'no_kart']:
            os.makedirs(output_dir / split / class_name, exist_ok=True)
    
    # 1. Copier les images de karts (déjà organisées)
    print("\n📦 1. Copie des images de KARTS...")
    for split in ['train', 'val', 'test']:
        src_dir = kart_source / split / 'go_kart'
        dst_dir = output_dir / split / 'go_kart'
        
        if src_dir.exists():
            files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
            for file in files:
                shutil.copy2(src_dir / file, dst_dir / file)
            print(f"  ✓ {split}: {len(files)} images de karts")
    
    # 2. Collecter toutes les images NO_KART
    print("\n📦 2. Collecte des images NO_KART...")
    no_kart_images = []

    def _is_image_file(filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Routes cassées
    if road_broken.exists():
        broken_files = [
            (str(road_broken / f), 'road_broken')
            for f in os.listdir(road_broken)
            if f.lower().endswith('.jpg')
        ]
        no_kart_images.extend(broken_files)
        print(f"  ✓ Routes cassées: {len(broken_files)} images")
    
    # Routes non cassées
    if road_not_broken.exists():
        not_broken_files = [
            (str(road_not_broken / f), 'road_ok')
            for f in os.listdir(road_not_broken)
            if f.lower().endswith('.jpg')
        ]
        no_kart_images.extend(not_broken_files)
        print(f"  ✓ Routes intactes: {len(not_broken_files)} images")
    
    # Circuit vide
    if circuit_empty.exists():
        circuit_files = [
            (str(circuit_empty / f), 'circuit')
            for f in os.listdir(circuit_empty)
            if f.lower().endswith('.jpg')
        ]
        no_kart_images.extend(circuit_files)
        print(f"  ✓ Circuit vide: {len(circuit_files)} images")
    
    # Vélos (prendre un échantillon)
    if bicycle_dir.exists():
        # On ignore bicycle/labels et on récupère les images à la racine
        bicycle_files = [
            f for f in os.listdir(bicycle_dir)
            if _is_image_file(f) and not (bicycle_dir / f).is_dir()
        ]
        random.seed(42)
        bicycle_sample = random.sample(bicycle_files, min(200, len(bicycle_files)))
        bicycle_paths = [(str(bicycle_dir / f), 'bicycle') for f in bicycle_sample]
        no_kart_images.extend(bicycle_paths)
        print(f"  ✓ Vélos: {len(bicycle_paths)} images (échantillon)")

    # No_kart manuel (dossier local du projet)
    if manual_no_kart_dir.exists():
        manual_files = [
            f for f in os.listdir(manual_no_kart_dir)
            if _is_image_file(f) and not (manual_no_kart_dir / f).is_dir()
        ]
        manual_paths = [(str(manual_no_kart_dir / f), 'manual') for f in manual_files]
        no_kart_images.extend(manual_paths)
        print(f"  ✓ No_kart manuel: {len(manual_paths)} images")
    
    print(f"\n  📊 TOTAL NO_KART: {len(no_kart_images)} images")
    
    # 3. Mélanger et diviser
    print("\n📂 3. Répartition des images NO_KART...")
    random.seed(42)
    random.shuffle(no_kart_images)
    
    # Split: 70% train, 15% val, 15% test
    total = len(no_kart_images)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    
    train_no_kart = no_kart_images[:train_size]
    val_no_kart = no_kart_images[train_size:train_size + val_size]
    test_no_kart = no_kart_images[train_size + val_size:]
    
    print(f"  • Train: {len(train_no_kart)} images")
    print(f"  • Val:   {len(val_no_kart)} images")
    print(f"  • Test:  {len(test_no_kart)} images")
    
    # 4. Copier les images NO_KART
    print("\n🚀 4. Copie des images NO_KART dans le dataset...")
    splits_data = {
        'train': train_no_kart,
        'val': val_no_kart,
        'test': test_no_kart
    }
    
    for split, images in splits_data.items():
        dst_dir = output_dir / split / 'no_kart'
        for i, (img_path, source_type) in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            dst_path = dst_dir / f'{source_type}_{i:04d}{ext}'
            shutil.copy2(img_path, dst_path)
        print(f"  ✓ {split}: {len(images)} images copiées")
    
    # 5. Résumé final
    print("\n" + "=" * 70)
    print("✅ DATASET ENRICHI PRÊT!")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        kart_count = len(os.listdir(output_dir / split / 'go_kart'))
        no_kart_count = len(os.listdir(output_dir / split / 'no_kart'))
        total_count = kart_count + no_kart_count
        
        print(f"\n{split.upper()}:")
        print(f"  • go_kart:  {kart_count:4d} images ({100*kart_count/total_count:.1f}%)")
        print(f"  • no_kart:  {no_kart_count:4d} images ({100*no_kart_count/total_count:.1f}%)")
        print(f"  • TOTAL:    {total_count:4d} images")
    
    print(f"\n📁 Dataset créé dans: {output_dir}")
    print("\n🎯 Prochaine étape:")
    print("  1. Lancez: python train.py")

    return str(output_dir)


if __name__ == "__main__":
    dataset_path = prepare_enhanced_dataset()
