import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2

from predict import KartDetector


ROI = Tuple[int, int, int, int]


@dataclass
class CollectedSample:
    frame: int
    timestamp_s: float
    probability: float
    roi: Optional[ROI]
    roi_path: str
    full_path: Optional[str] = None


def _extract_roi(frame_bgr, roi: Optional[ROI]):
    if roi is None:
        return frame_bgr
    x, y, w, h = roi
    return frame_bgr[y : y + h, x : x + w]


def collect_false_positive_candidates(
    *,
    model_path: str,
    video_path: str,
    roi: Optional[ROI],
    out_dir: str,
    min_score: float = 0.5,
    max_score: float = 0.8,
    frame_skip: int = 2,
    max_samples: int = 500,
    min_gap_frames: int = 10,
    save_full_frames: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[CollectedSample]:
    """Sauvegarde des *candidats* faux-positifs.

    Comme on n'a pas les labels, on récupère simplement les frames dont le score
    P(go_kart) est dans [min_score, max_score].

    L'idée est de collecter la zone grise (0.5-0.8) sur des vidéos plutôt "no_kart",
    puis tu trieras manuellement.
    """

    if frame_skip < 1:
        raise ValueError("frame_skip doit être >= 1")
    if not (0.0 <= min_score <= 1.0 and 0.0 <= max_score <= 1.0):
        raise ValueError("min_score/max_score doivent être dans [0, 1]")
    if min_score > max_score:
        raise ValueError("min_score doit être <= max_score")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    crops_dir = out_path / "roi"
    crops_dir.mkdir(parents=True, exist_ok=True)

    full_dir = out_path / "frames"
    if save_full_frames:
        full_dir.mkdir(parents=True, exist_ok=True)

    detector = KartDetector(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    samples: list[CollectedSample] = []
    last_saved_frame = -10**9
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            roi_frame = _extract_roi(frame, roi)
            _, prob = detector.predict_frame(roi_frame)

            if min_score <= prob <= max_score and (frame_idx - last_saved_frame) >= min_gap_frames:
                ts = float(frame_idx) / float(fps)

                roi_name = crops_dir / f"fp_frame_{frame_idx:06d}_t_{ts:.2f}_p_{prob:.3f}.jpg"
                cv2.imwrite(str(roi_name), roi_frame)

                full_path: Optional[str] = None
                if save_full_frames:
                    annotated = frame.copy()
                    if roi is not None:
                        x, y, w, h = roi
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"CANDIDAT FP p={prob:.3f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"frame={frame_idx} t={ts:.2f}s",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    full_name = full_dir / f"fp_frame_{frame_idx:06d}_p_{prob:.3f}.jpg"
                    cv2.imwrite(str(full_name), annotated)
                    full_path = str(full_name)

                samples.append(
                    CollectedSample(
                        frame=int(frame_idx),
                        timestamp_s=ts,
                        probability=float(prob),
                        roi=roi,
                        roi_path=str(roi_name),
                        full_path=full_path,
                    )
                )
                last_saved_frame = frame_idx

                if max_samples and len(samples) >= int(max_samples):
                    break

            if callable(progress_callback) and total_frames > 0:
                try:
                    progress_callback(frame_idx, total_frames)
                except Exception:
                    pass

            frame_idx += 1

    finally:
        cap.release()

    manifest = {
        "video": str(video_path),
        "model": str(model_path),
        "roi": list(roi) if roi is not None else None,
        "min_score": float(min_score),
        "max_score": float(max_score),
        "frame_skip": int(frame_skip),
        "max_samples": int(max_samples),
        "min_gap_frames": int(min_gap_frames),
        "count": len(samples),
        "samples": [s.__dict__ for s in samples],
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return samples


def _parse_roi(roi_str: Optional[str]) -> Optional[ROI]:
    if not roi_str:
        return None
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI invalide: attendu 'x,y,w,h'")
    x, y, w, h = (int(float(p)) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("ROI invalide: w/h doivent être > 0")
    return (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Collecte des candidats faux-positifs (zone grise)")
    parser.add_argument("--model", required=True, help="Chemin vers le modèle .pth")
    parser.add_argument("--video", required=True, help="Chemin vers la vidéo")
    parser.add_argument("--out", default="faux_positifs", help="Dossier de sortie")
    parser.add_argument("--roi", default=None, help="ROI au format x,y,w,h (optionnel)")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--max-score", type=float, default=0.8)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--min-gap-frames", type=int, default=10)
    parser.add_argument("--save-full-frames", action="store_true")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"run_{run_id}"

    roi = _parse_roi(args.roi)

    samples = collect_false_positive_candidates(
        model_path=args.model,
        video_path=args.video,
        roi=roi,
        out_dir=str(out_dir),
        min_score=float(args.min_score),
        max_score=float(args.max_score),
        frame_skip=int(args.frame_skip),
        max_samples=int(args.max_samples),
        min_gap_frames=int(args.min_gap_frames),
        save_full_frames=bool(args.save_full_frames),
    )

    print(f"OK: {len(samples)} candidats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    main()
