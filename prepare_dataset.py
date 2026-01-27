import os
import shutil
import random

def prepare_binary_dataset():
    """
    Préparer le dataset avec 2 classes:
    - go_kart: images avec des karts
    - no_kart: images de routes (sans kart)
    """
    
    # Chemins source
    kart_source = r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data"
    road_broken = r"c:\Users\gatie\Downloads\archive (1)\Road Classification\Broken"
    road_not_broken = r"c:\Users\gatie\Downloads\archive (1)\Road Classification\Not Broken"
    
    # Chemin destination
    output_dir = r"c:\Users\gatie\Downloads\hqn7yuibkl9q5oufmmcwg\images.cv_hqn7yuibkl9q5oufmmcwg\data_binary"
    
    print("🔧 Préparation du dataset binaire...")
    print("="*60)
    
    # Créer la structure de dossiers
    for split in ['train', 'val', 'test']:
        for class_name in ['go_kart', 'no_kart']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # 1. Copier les images de karts (déjà organisées)
    print("\n📦 Copie des images de karts...")
    for split in ['train', 'val', 'test']:
        src_dir = os.path.join(kart_source, split, 'go_kart')
        dst_dir = os.path.join(output_dir, split, 'go_kart')
        
        if os.path.exists(src_dir):
            files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
            for file in files:
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
            print(f"  ✓ {split}: {len(files)} images de karts copiées")
    
    # 2. Collecter toutes les images de routes
    print("\n📦 Collecte des images de routes (no_kart)...")
    road_images = []
    
    # Images de routes cassées
    if os.path.exists(road_broken):
        broken_files = [os.path.join(road_broken, f) for f in os.listdir(road_broken) if f.endswith('.jpg')]
        road_images.extend(broken_files)
        print(f"  ✓ {len(broken_files)} images de routes cassées")
    
    # Images de routes non cassées
    if os.path.exists(road_not_broken):
        not_broken_files = [os.path.join(road_not_broken, f) for f in os.listdir(road_not_broken) if f.endswith('.jpg')]
        road_images.extend(not_broken_files)
        print(f"  ✓ {len(not_broken_files)} images de routes non cassées")
    
    print(f"  📊 Total images no_kart: {len(road_images)}")
    
    # 3. Mélanger et diviser les images de routes
    random.seed(42)
    random.shuffle(road_images)
    
    # Split: 70% train, 15% val, 15% test
    total = len(road_images)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    
    train_roads = road_images[:train_size]
    val_roads = road_images[train_size:train_size + val_size]
    test_roads = road_images[train_size + val_size:]
    
    print(f"\n📂 Répartition des images no_kart:")
    print(f"  • Train: {len(train_roads)} images")
    print(f"  • Val:   {len(val_roads)} images")
    print(f"  • Test:  {len(test_roads)} images")
    
    # 4. Copier les images de routes dans les bons dossiers
    print("\n🚀 Copie des images no_kart...")
    splits_data = {
        'train': train_roads,
        'val': val_roads,
        'test': test_roads
    }
    
    for split, images in splits_data.items():
        dst_dir = os.path.join(output_dir, split, 'no_kart')
        for i, img_path in enumerate(images):
            ext = os.path.splitext(img_path)[1]
            dst_path = os.path.join(dst_dir, f'road_{i:04d}{ext}')
            shutil.copy2(img_path, dst_path)
        print(f"  ✓ {split}: {len(images)} images no_kart copiées")
    
    # 5. Résumé final
    print("\n" + "="*60)
    print("✅ DATASET PRÊT!")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        kart_count = len(os.listdir(os.path.join(output_dir, split, 'go_kart')))
        no_kart_count = len(os.listdir(os.path.join(output_dir, split, 'no_kart')))
        total_count = kart_count + no_kart_count
        
        print(f"\n{split.upper()}:")
        print(f"  • go_kart:  {kart_count:4d} images ({100*kart_count/total_count:.1f}%)")
        print(f"  • no_kart:  {no_kart_count:4d} images ({100*no_kart_count/total_count:.1f}%)")
        print(f"  • TOTAL:    {total_count:4d} images")
    
    print(f"\n📁 Dataset créé dans: {output_dir}")
    print("\n🎯 Prochaine étape: Lancez 'python train.py' pour entraîner le modèle!")
    
    return output_dir

if __name__ == "__main__":
    dataset_path = prepare_binary_dataset()
