import cv2
import torch
import numpy as np
from predict import KartDetector
import json
import os
from datetime import datetime
from pathlib import Path
import argparse
import sys

class VideoKartDetector:
    def __init__(self, model_path, video_path, roi=None):
        """
        Détecteur de karts dans une vidéo avec zone d'intérêt (ROI)
        
        Args:
            model_path: Chemin vers le modèle entraîné
            video_path: Chemin vers la vidéo MP4
            roi: Region of Interest (x, y, width, height) ou None pour toute l'image
        """
        self.detector = KartDetector(model_path)
        self.video_path = video_path
        self.roi = roi
        self.results = []
        
    def set_roi_interactive(self):
        """Définir la ROI de manière interactive (une seule fois au début)"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Impossible de lire la vidéo")
        
        # Dimensions originales
        orig_height, orig_width = frame.shape[:2]
        
        # Redimensionner pour l'affichage - taille adaptée à l'écran
        max_display_width = 960  # Réduit pour s'assurer que ça tient à l'écran
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
            print(f"  Échelle: {scale:.2f}")
        else:
            display_frame = frame
        
        print("\n🎯 SÉLECTION DE LA ZONE D'INTÉRÊT (une seule fois)")
        print("=" * 60)
        print("Instructions:")
        print("  1. Cliquez et glissez pour dessiner un rectangle")
        print("  2. Appuyez sur ENTRÉE pour valider")
        print("  3. Appuyez sur 'c' pour annuler et recommencer")
        print("  4. Appuyez sur ESC pour quitter sans sauvegarder")
        print("=" * 60)
        
        # Créer la fenêtre nommée et la positionner
        window_name = "Zone de detection - Appuyez ENTREE pour valider"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_frame.shape[1], display_frame.shape[0])
        cv2.moveWindow(window_name, 50, 50)  # Positionner en haut à gauche de l'écran
        
        # Sélection ROI avec OpenCV sur l'image redimensionnée
        roi = cv2.selectROI(window_name, display_frame, False, False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # Si une zone a été sélectionnée
            # Remettre à l'échelle si nécessaire
            if scale != 1.0:
                roi = tuple(int(coord / scale) for coord in roi)
            
            self.roi = roi
            print(f"\n✓ Zone sélectionnée (coordonnées réelles):")
            print(f"  x={roi[0]}, y={roi[1]}, largeur={roi[2]}, hauteur={roi[3]}")
            return roi
        else:
            print("\n✗ Aucune zone sélectionnée, utilisation de l'image complète")
            self.roi = None
            return None
    
    def extract_roi(self, frame):
        """Extraire la région d'intérêt d'une frame"""
        if self.roi is None:
            return frame
        
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]
    
    def detect_in_video(
        self,
        output_dir='output',
        confidence_threshold=0.5,
        save_frames=True,
        frame_skip=1,
        min_consecutive=1,
        dump_roi_samples=0,
        progress_callback=None,
    ):
        """
        Analyser la vidéo et détecter les karts
        
        Args:
            output_dir: Dossier pour sauvegarder les résultats
            confidence_threshold: Seuil de confiance pour la détection
            save_frames: Sauvegarder les frames avec détections
            frame_skip: Analyser 1 frame sur N (pour accélérer)
            min_consecutive: Nombre minimum de frames consécutives >= seuil pour valider une détection
        """
        # Créer le dossier de sortie
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        frames_path = output_path / 'frames'
        if save_frames:
            frames_path.mkdir(exist_ok=True)

        roi_samples_path = output_path / 'roi_samples'
        if dump_roi_samples and dump_roi_samples > 0:
            roi_samples_path.mkdir(exist_ok=True)
        
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {self.video_path}")
        
        # Informations vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print("\n🎬 ANALYSE DE LA VIDÉO")
        print("=" * 60)
        print(f"Vidéo: {self.video_path}")
        print(f"FPS: {fps:.2f}")
        print(f"Frames totales: {total_frames}")
        print(f"Durée: {duration:.2f} secondes")
        print(f"Zone d'intérêt: {'Oui' if self.roi else 'Image complète'}")
        print(f"Frame skip: 1/{frame_skip}")
        print("=" * 60)
        
        frame_count = 0
        kart_detections = []
        consecutive_hits = 0
        dumped = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyser seulement certaines frames
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Extraire la ROI
                roi_frame = self.extract_roi(frame)

                # Détection directe sur la frame OpenCV (corrige le piège BGR/RGB)
                prediction, probability = self.detector.predict_frame(roi_frame)

                # Debug: sauvegarder quelques crops ROI pour inspection
                if dump_roi_samples and dump_roi_samples > 0 and dumped < dump_roi_samples:
                    sample_name = roi_samples_path / f'roi_frame_{frame_count:06d}_p_{probability:.3f}.jpg'
                    cv2.imwrite(str(sample_name), roi_frame)
                    dumped += 1
                
                # Calculer le timestamp
                timestamp = frame_count / fps
                
                # Appliquer un filtre "N frames consécutives" pour éviter les faux positifs ponctuels
                is_hit = probability >= confidence_threshold
                consecutive_hits = consecutive_hits + 1 if is_hit else 0
                has_kart = consecutive_hits >= max(1, int(min_consecutive))

                # Enregistrer le résultat
                result = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'has_kart': has_kart,
                    'raw_hit': is_hit,
                    'consecutive_hits': consecutive_hits,
                    'probability': probability,
                    'prediction': prediction
                }
                self.results.append(result)

                if callable(progress_callback):
                    try:
                        progress_callback(frame_count, total_frames)
                    except Exception:
                        pass
                
                # Si kart détecté (après filtre)
                if has_kart:
                    kart_detections.append(result)
                    
                    # Sauvegarder la frame si demandé
                    if save_frames:
                        # Annoter l'image
                        annotated = frame.copy()
                        if self.roi:
                            x, y, w, h = self.roi
                            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Ajouter le texte
                        text = f"KART DETECTE - {probability:.2%}"
                        cv2.putText(annotated, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Timestamp
                        time_text = f"t={timestamp:.2f}s"
                        cv2.putText(annotated, time_text, (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Sauvegarder
                        frame_filename = frames_path / f'frame_{frame_count:06d}_kart.jpg'
                        cv2.imwrite(str(frame_filename), annotated)
                
                # Afficher la progression
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames}", end='\r')
                
                frame_count += 1
                
        finally:
            cap.release()
        
        print(f"\n\n✓ Analyse terminée!")
        print(f"  • Frames analysées: {len(self.results)}")
        print(f"  • Karts détectés: {len(kart_detections)} fois")
        
        return self.results, kart_detections
    
    def analyze_passages(self, min_gap_seconds=2.0):
        """
        Analyser les passages de karts à partir des détections
        Un passage = transition no_kart -> kart -> no_kart
        
        Args:
            min_gap_seconds: Temps minimum entre deux passages (pour éviter les doublons)
        """
        if not self.results:
            print("Aucune détection à analyser")
            return []
        
        passages = []
        in_passage = False
        passage_start = None
        passage_frames = []
        last_passage_time = -999
        
        for result in self.results:
            if result['has_kart']:
                if not in_passage:
                    # Début d'un nouveau passage
                    if result['timestamp'] - last_passage_time >= min_gap_seconds:
                        in_passage = True
                        passage_start = result['timestamp']
                        passage_frames = [result]
                else:
                    passage_frames.append(result)
            else:
                if in_passage:
                    # Fin du passage
                    passage_end = passage_frames[-1]['timestamp']
                    max_prob = max(f['probability'] for f in passage_frames)
                    
                    passage_info = {
                        'passage_id': len(passages) + 1,
                        'start_time': passage_start,
                        'end_time': passage_end,
                        'duration': passage_end - passage_start,
                        'max_confidence': max_prob,
                        'frame_count': len(passage_frames)
                    }
                    passages.append(passage_info)
                    
                    last_passage_time = passage_end
                    in_passage = False
                    passage_frames = []
        
        print("\n📊 ANALYSE DES PASSAGES")
        print("=" * 60)
        print(f"Nombre de passages détectés: {len(passages)}")
        print("=" * 60)
        
        for p in passages:
            print(f"\nPassage #{p['passage_id']}:")
            print(f"  • Temps: {p['start_time']:.2f}s - {p['end_time']:.2f}s")
            print(f"  • Durée: {p['duration']:.2f}s")
            print(f"  • Confiance max: {p['max_confidence']:.2%}")
            print(f"  • Frames: {p['frame_count']}")
        
        return passages
    
    def save_results(self, output_dir='output'):
        """Sauvegarder les résultats en JSON"""
        output_path = Path(output_dir)
        
        # Analyser les passages
        passages = self.analyze_passages()
        
        # Préparer les données
        data = {
            'video_path': str(self.video_path),
            'roi': self.roi,
            'analysis_date': datetime.now().isoformat(),
            'total_frames_analyzed': len(self.results),
            'total_passages': len(passages),
            'passages': passages,
            'all_detections': self.results
        }
        
        # Sauvegarder
        results_file = output_path / 'detection_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {results_file}")
        
        # Créer un résumé texte
        summary_file = output_path / 'summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RÉSUMÉ DE L'ANALYSE VIDÉO\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Vidéo: {self.video_path}\n")
            f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Zone d'intérêt: {self.roi if self.roi else 'Image complète'}\n\n")
            f.write(f"Frames analysées: {len(self.results)}\n")
            f.write(f"Passages détectés: {len(passages)}\n\n")
            
            f.write("PASSAGES:\n")
            f.write("-" * 60 + "\n")
            for p in passages:
                f.write(f"\nPassage #{p['passage_id']}:\n")
                f.write(f"  Temps: {p['start_time']:.2f}s - {p['end_time']:.2f}s\n")
                f.write(f"  Durée: {p['duration']:.2f}s\n")
                f.write(f"  Confiance: {p['max_confidence']:.2%}\n")
        
        print(f"📄 Résumé sauvegardé: {summary_file}")
        
        return results_file, summary_file


def main():
    """Fonction principale pour analyser une vidéo"""

    interactive_mode = len(sys.argv) == 1

    parser = argparse.ArgumentParser(description="Détection de passages de karts dans une vidéo (classification binaire + ROI)")
    parser.add_argument('--video', default='kart_video.mp4', help="Chemin vers la vidéo")
    parser.add_argument('--model', default=None, help="Chemin vers le modèle .pth (sinon: modèle le plus récent dans models/)")
    parser.add_argument('--model-dir', default='models', help="Dossier contenant les modèles")
    parser.add_argument('--output', default='output_video_analysis', help="Dossier de sortie")
    parser.add_argument('--roi', default=None, help="ROI x,y,w,h (ex: 3144,1184,168,112). Si absent: interactif si --interactive-roi")
    parser.add_argument('--interactive-roi', action='store_true', help="Ouvre une fenêtre pour sélectionner la ROI")
    parser.add_argument('--frame-skip', type=int, default=5, help="Analyser 1 frame sur N")
    parser.add_argument('--threshold', type=float, default=0.55, help="Seuil de confiance (probabilité kart) pour valider un hit")
    parser.add_argument('--min-consecutive', type=int, default=1, help="Nombre minimum de frames consécutives >= seuil")
    parser.add_argument('--save-frames', action='store_true', help="Sauvegarder les frames avec détections")
    parser.add_argument('--debug-first', type=int, default=0, help="Afficher les scores des N premières frames analysées")
    parser.add_argument('--dump-roi-samples', type=int, default=0, help="Sauvegarder les N premiers crops ROI pour vérifier la zone")
    parser.add_argument('--report-top', type=int, default=0, help="Afficher les N meilleures frames (probabilité kart)")
    args = parser.parse_args()

    video_path = args.video
    model_dir = args.model_dir
    output_dir = args.output
    
    # Vérifier que la vidéo existe
    if not os.path.exists(video_path):
        print(f"❌ Erreur: Vidéo non trouvée '{video_path}'")
        print("\nAjoutez votre vidéo dans le dossier et nommez-la 'kart_video.mp4'")
        return

    # Choisir le modèle
    if args.model:
        best_model = Path(args.model)
        if not best_model.exists():
            print(f"❌ Modèle introuvable: {best_model}")
            return
    else:
        model_files = list(Path(model_dir).glob('*.pth'))
        if not model_files:
            print("❌ Aucun modèle trouvé. Lancez d'abord train.py")
            return

        # Par défaut, prendre le plus récent
        best_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"✓ Modèle sélectionné: {best_model.name}")
    
    detector = VideoKartDetector(str(best_model), video_path)

    # ROI
    if interactive_mode:
        print("\n" + "=" * 60)
        choice = input("Voulez-vous sélectionner une zone (ROI) pour ZOOMER sur le circuit ? (o/n) [o]: ").lower() or 'o'
        if choice != 'n':
            detector.set_roi_interactive()
        else:
            print("✓ ROI: image complète")
    elif args.roi:
        try:
            x, y, w, h = (int(p.strip()) for p in args.roi.split(','))
            detector.roi = (x, y, w, h)
            print(f"✓ ROI définie via CLI: {detector.roi}")
        except Exception:
            print(f"❌ ROI invalide: {args.roi} (attendu: x,y,w,h)")
            return
    elif args.interactive_roi:
        detector.set_roi_interactive()
    else:
        print("✓ ROI: image complète (aucun crop)")

    # Paramètres interactifs (quand lancé sans args)
    if interactive_mode:
        print("\n" + "=" * 60)
        print("PARAMÈTRES D'ANALYSE")
        print("=" * 60)
        args.frame_skip = int(input("Analyser 1 frame sur combien? (1=toutes, 2=une sur deux, etc.) [2]: ") or "2")
        args.threshold = float(input("Seuil de confiance (0.0-1.0) [0.55]: ") or "0.55")
        args.min_consecutive = int(input("Nombre min. de frames consécutives >= seuil [1]: ") or "1")
        args.save_frames = (input("Sauvegarder les frames avec détections? (o/n) [o]: ").lower() or 'o') != 'n'
        args.debug_first = int(input("Debug: afficher les scores des N premières frames analysées [30]: ") or "30")
        args.dump_roi_samples = int(input("Sauvegarder N crops ROI pour vérifier la zone [30]: ") or "30")
        args.report_top = int(input("Afficher les N meilleures frames (probabilité kart) [10]: ") or "10")
    
    # Analyser la vidéo
    results, detections = detector.detect_in_video(
        output_dir=output_dir,
        confidence_threshold=args.threshold,
        save_frames=args.save_frames,
        frame_skip=args.frame_skip,
        min_consecutive=args.min_consecutive,
        dump_roi_samples=args.dump_roi_samples,
    )

    if args.debug_first > 0:
        print("\n" + "=" * 60)
        print("DEBUG SCORES (premières frames analysées)")
        print("=" * 60)
        for r in results[:args.debug_first]:
            print(
                f"frame={r['frame']:<6d} t={r['timestamp']:<8.2f}s kart_prob={r['probability']:.4f} "
                f"raw_hit={r.get('raw_hit', False)} cons={r.get('consecutive_hits', 0)} has_kart={r['has_kart']}"
            )
    
    # Sauvegarder les résultats
    detector.save_results(output_dir)

    # Rapport top-N (utile quand on a "0 détection")
    if args.report_top and args.report_top > 0 and results:
        print("\n" + "=" * 60)
        print(f"TOP {args.report_top} PROBAS (kart_prob)")
        print("=" * 60)
        top = sorted(results, key=lambda r: r.get('probability', 0.0), reverse=True)[:args.report_top]
        for r in top:
            print(f"frame={r['frame']:<6d} t={r['timestamp']:<8.2f}s kart_prob={r['probability']:.4f} pred={r['prediction']}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSE TERMINÉE!")
    print("=" * 60)
    print(f"Les résultats sont dans le dossier: {output_dir}/")


if __name__ == "__main__":
    main()
