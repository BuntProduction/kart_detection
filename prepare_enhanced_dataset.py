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
    
    # Chemins source
    kart_source = r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data"
    road_broken = r"c:\Users\gatie\Downloads\archive (1)\Road Classification\Broken"
    road_not_broken = r"c:\Users\gatie\Downloads\archive (1)\Road Classification\Not Broken"
    circuit_empty = "extracted_empty_circuit"
    bicycle_dir = r"c:\Users\gatie\Downloads\archive (2)\bicycle"
    manual_no_kart_dir = "No kart"
    
    # Chemin destination
    output_dir = r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data_enhanced"
    
    print("🔧 PRÉPARATION DU DATASET ENRICHI")
    print("=" * 70)
    print("Sources:")
    print("  • Karts: Dataset original")
    print("  • Routes génériques: Broken + Not Broken")
    print("  • Circuit vide: Frames extraites de la vidéo")
    print("  • Vélos: Dataset complémentaire")
    print(f"  • No_kart manuel: {manual_no_kart_dir}")
    print("=" * 70)
    
    # Créer la structure de dossiers
    for split in ['train', 'val', 'test']:
        for class_name in ['go_kart', 'no_kart']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # 1. Copier les images de karts (déjà organisées)
    print("\n📦 1. Copie des images de KARTS...")
    for split in ['train', 'val', 'test']:
        src_dir = os.path.join(kart_source, split, 'go_kart')
        dst_dir = os.path.join(output_dir, split, 'go_kart')
        
        if os.path.exists(src_dir):
            files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
            for file in files:
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
            print(f"  ✓ {split}: {len(files)} images de karts")
    
    # 2. Collecter toutes les images NO_KART
    print("\n📦 2. Collecte des images NO_KART...")
    no_kart_images = []

    def _is_image_file(filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Routes cassées
    if os.path.exists(road_broken):
        broken_files = [(os.path.join(road_broken, f), 'road_broken') 
                       for f in os.listdir(road_broken) if f.endswith('.jpg')]
        no_kart_images.extend(broken_files)
        print(f"  ✓ Routes cassées: {len(broken_files)} images")
    
    # Routes non cassées
    if os.path.exists(road_not_broken):
        not_broken_files = [(os.path.join(road_not_broken, f), 'road_ok') 
                           for f in os.listdir(road_not_broken) if f.endswith('.jpg')]
        no_kart_images.extend(not_broken_files)
        print(f"  ✓ Routes intactes: {len(not_broken_files)} images")
    
    # Circuit vide
    if os.path.exists(circuit_empty):
        circuit_files = [(os.path.join(circuit_empty, f), 'circuit') 
                        for f in os.listdir(circuit_empty) if f.endswith('.jpg')]
        no_kart_images.extend(circuit_files)
        print(f"  ✓ Circuit vide: {len(circuit_files)} images")
    
    # Vélos (prendre un échantillon)
    if os.path.exists(bicycle_dir):
        bicycle_files = [f for f in os.listdir(bicycle_dir) 
                        if f.endswith('.jpg') and not os.path.isdir(os.path.join(bicycle_dir, f))]
        # Prendre 200 vélos aléatoires pour équilibrer
        random.seed(42)
        bicycle_sample = random.sample(bicycle_files, min(200, len(bicycle_files)))
        bicycle_paths = [(os.path.join(bicycle_dir, f), 'bicycle') for f in bicycle_sample]
        no_kart_images.extend(bicycle_paths)
        print(f"  ✓ Vélos: {len(bicycle_paths)} images (échantillon)")

    # No_kart manuel (dossier local du projet)
    if os.path.exists(manual_no_kart_dir):
        manual_files = [
            f for f in os.listdir(manual_no_kart_dir)
            if _is_image_file(f) and not os.path.isdir(os.path.join(manual_no_kart_dir, f))
        ]
        manual_paths = [(os.path.join(manual_no_kart_dir, f), 'manual') for f in manual_files]
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
        dst_dir = os.path.join(output_dir, split, 'no_kart')
        for i, (img_path, source_type) in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            dst_path = os.path.join(dst_dir, f'{source_type}_{i:04d}{ext}')
            shutil.copy2(img_path, dst_path)
        print(f"  ✓ {split}: {len(images)} images copiées")
    
    # 5. Résumé final
    print("\n" + "=" * 70)
    print("✅ DATASET ENRICHI PRÊT!")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        kart_count = len(os.listdir(os.path.join(output_dir, split, 'go_kart')))
        no_kart_count = len(os.listdir(os.path.join(output_dir, split, 'no_kart')))
        total_count = kart_count + no_kart_count
        
        print(f"\n{split.upper()}:")
        print(f"  • go_kart:  {kart_count:4d} images ({100*kart_count/total_count:.1f}%)")
        print(f"  • no_kart:  {no_kart_count:4d} images ({100*no_kart_count/total_count:.1f}%)")
        print(f"  • TOTAL:    {total_count:4d} images")
    
    print(f"\n📁 Dataset créé dans: {output_dir}")
    print("\n🎯 Prochaine étape:")
    print("  1. Mettez à jour train.py avec le nouveau chemin:")
    print(f"     DATA_DIR = r'{output_dir}'")
    print("  2. Lancez: python train.py")
    
    return output_dir


if __name__ == "__main__":
    dataset_path = prepare_enhanced_dataset()
