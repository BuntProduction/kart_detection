import cv2
import os
from pathlib import Path

def select_roi_from_video(video_path):
    """Sélectionner la zone d'intérêt sur la vidéo"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Impossible de lire la vidéo")
    
    # Dimensions originales
    orig_height, orig_width = frame.shape[:2]
    
    # Redimensionner pour l'affichage
    max_display_width = 960
    max_display_height = 540
    
    scale = 1.0
    if orig_width > max_display_width or orig_height > max_display_height:
        scale_w = max_display_width / orig_width
        scale_h = max_display_height / orig_height
        scale = min(scale_w, scale_h)
        
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        print(f"\n📐 Vidéo redimensionnée pour affichage:")
        print(f"  Original: {orig_width}x{orig_height}")
        print(f"  Affichage: {display_width}x{display_height}")
    else:
        display_frame = frame
    
    print("\n🎯 SÉLECTION DE LA ZONE À EXTRAIRE")
    print("=" * 60)
    print("Sélectionnez la zone de chronométrage/détection")
    print("  1. Cliquez et glissez pour dessiner un rectangle")
    print("  2. Appuyez sur ENTRÉE pour valider")
    print("=" * 60)
    
    # Créer la fenêtre et la positionner
    window_name = "Selectionnez la zone - Appuyez ENTREE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
    cv2.moveWindow(window_name, 50, 50)
    
    # Sélection ROI
    roi = cv2.selectROI(window_name, display_frame, False, False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        # Remettre à l'échelle si nécessaire
        if scale != 1.0:
            roi = tuple(int(coord / scale) for coord in roi)
        
        print(f"\n✓ Zone sélectionnée: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        return roi
    else:
        print("\n✗ Aucune zone sélectionnée")
        return None

def extract_frames_for_training(video_path, output_dir, roi=None, num_frames=200, frame_skip=30):
    """
    Extraire des frames de la vidéo pour enrichir le dataset 'no_kart'
    avec des images du circuit vide (zone ROI uniquement)
    
    Args:
        video_path: Chemin vers la vidéo
        output_dir: Dossier de sortie
        roi: Zone d'intérêt (x, y, width, height) ou None pour l'image complète
        num_frames: Nombre de frames à extraire
        frame_skip: Extraire 1 frame tous les N frames
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\n🎬 EXTRACTION DE FRAMES POUR ENTRAÎNEMENT")
    print("=" * 60)
    print(f"Vidéo: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Zone: {roi if roi else 'Image complète'}")
    print(f"Extraction: {num_frames} frames (1 tous les {frame_skip} frames)")
    print("=" * 60)
    
    extracted = 0
    frame_count = 0
    
    while extracted < num_frames and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Extraire la ROI si définie
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            # Sauvegarder la frame (zoomée sur la zone)
            filename = output_path / f'circuit_empty_{extracted:04d}.jpg'
            cv2.imwrite(str(filename), frame)
            extracted += 1
            
            if extracted % 10 == 0:
                print(f"Extrait: {extracted}/{num_frames} frames")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n✅ {extracted} frames extraites dans: {output_dir}")
    print("\n📝 PROCHAINES ÉTAPES:")
    print("1. Vérifiez les images extraites et supprimez celles avec des karts")
    print("2. Copiez les images restantes dans:")
    print(f"   data_binary/train/no_kart/ (~140 images)")
    print(f"   data_binary/val/no_kart/ (~30 images)")
    print("3. Relancez l'entraînement avec: python train.py")
    print("4. Testez avec le nouveau modèle")


if __name__ == "__main__":
    video_path = "kart_video.mp4"
    output_dir = "extracted_empty_circuit"
    
    if not os.path.exists(video_path):
        print(f"❌ Vidéo non trouvée: {video_path}")
        print("Assurez-vous que la vidéo est dans le dossier courant")
    else:
        # Sélectionner la zone d'intérêt
        roi = select_roi_from_video(video_path)
        
        if roi is None:
            print("\n⚠️ Aucune zone sélectionnée. Abandon.")
        else:
            # Extraire 50 frames zoomées sur cette zone
            extract_frames_for_training(video_path, output_dir, roi=roi, num_frames=50, frame_skip=30)
