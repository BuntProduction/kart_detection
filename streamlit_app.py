import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import streamlit.elements.image as st_image

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

# streamlit-drawable-canvas 0.9.3 dépend d'une API interne supprimée dans Streamlit récents.
# Si l'API n'existe pas, on désactive le canvas et on propose une ROI manuelle avec aperçu.
if st_canvas is not None and not hasattr(st_image, "image_to_url"):
    st_canvas = None

from video_detector import VideoKartDetector


APP_TITLE = "Kart detector (vidéo)"


CACHE_DIR = Path(".streamlit_cache")
CACHE_UPLOADS_DIR = CACHE_DIR / "uploads"


def list_models(models_dir: str = "models"):
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    return sorted(models_path.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)


def read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Impossible de lire la première frame de la vidéo")
    return frame


@st.cache_data(show_spinner=False)
def read_first_frame_cached(video_path: str) -> np.ndarray:
    return read_first_frame(video_path)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def ensure_run_dir(base_dir: str = "output_video_analysis") -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cache_uploaded_video(uploaded) -> Path:
    """Sauvegarde l'upload dans un cache stable (une seule fois) pour éviter
    de recopier la vidéo à chaque rerun Streamlit.
    """
    CACHE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    meta = (uploaded.name, int(getattr(uploaded, "size", 0) or 0))
    cached_meta = st.session_state.get("uploaded_video_meta")
    cached_path = st.session_state.get("uploaded_video_path")

    if cached_meta == meta and cached_path and Path(cached_path).exists():
        return Path(cached_path)

    # Nouvelle vidéo (ou cache manquant): recopier en streaming
    safe_name = Path(uploaded.name).name
    dst = CACHE_UPLOADS_DIR / safe_name

    uploaded.seek(0)
    with open(dst, "wb") as out:
        shutil.copyfileobj(uploaded, out, length=1024 * 1024)

    st.session_state["uploaded_video_meta"] = meta
    st.session_state["uploaded_video_path"] = str(dst)
    # Invalide le dernier run si on change de vidéo
    st.session_state.pop("last_run_dir", None)
    st.session_state.pop("last_run_summary", None)
    return dst


def clamp_roi(roi: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (x, y, w, h)


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)

st.markdown(
    """
Cette app utilise ton modèle **ResNet18** (classification binaire) pour analyser une vidéo.

Étapes:
1) Upload la vidéo
2) Choisis le modèle `.pth`
3) Dessine une ROI (zone à analyser) sur la 1ère frame
4) Lance l'analyse et regarde les frames/ROI samples générés
"""
)

models = list_models("models")
if not models:
    st.error("Aucun modèle trouvé dans le dossier `models/`. Lance d'abord l'entraînement.")
    st.stop()

model_choice = st.selectbox(
    "Modèle à utiliser",
    options=models,
    format_func=lambda p: p.name,
)

uploaded = st.file_uploader("Dépose ta vidéo (mp4)", type=["mp4", "mov", "avi", "mkv"])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Paramètres")
    frame_skip = st.number_input("Frame skip (1 = toutes)", min_value=1, max_value=60, value=2, step=1)
    threshold = st.slider("Seuil de confiance (probabilité kart)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    min_consecutive = st.number_input("Min frames consécutives >= seuil", min_value=1, max_value=30, value=1, step=1)
    save_frames = st.checkbox("Sauvegarder les frames détectées", value=True)
    dump_roi_samples = st.number_input("Sauver N crops ROI (debug)", min_value=0, max_value=200, value=30, step=10)

with col2:
    st.subheader("Sorties")
    output_base = st.text_input("Dossier de sortie", value="output_video_analysis")

if not uploaded:
    st.info("Upload une vidéo pour commencer.")
    st.stop()

video_path = cache_uploaded_video(uploaded)
st.success(f"Vidéo prête: {video_path}")

first_frame_bgr = read_first_frame_cached(str(video_path))
first_frame_rgb = bgr_to_rgb(first_frame_bgr)
height, width = first_frame_rgb.shape[:2]

st.subheader("Sélection de la ROI")
st.caption("Dessine un rectangle sur la 1ère frame (fond vidéo). Seule cette zone sera analysée.")

canvas_scale = 0.6 if width > 1280 else 1.0
canvas_w = int(width * canvas_scale)
canvas_h = int(height * canvas_scale)

roi = None

if st_canvas is None:
    st.warning(
        "Sélection ROI par dessin indisponible (incompatibilité Streamlit récente). "
        "Utilise la ROI manuelle ci-dessous (avec aperçu)."
    )

# ROI manuelle + aperçu (fonctionne partout)
default_roi = st.session_state.get("roi_manual") or (0, 0, min(300, width), min(200, height))
default_roi = clamp_roi(tuple(map(int, default_roi)), width=width, height=height)

r1, r2, r3, r4 = st.columns(4)
with r1:
    x = st.slider("x", min_value=0, max_value=max(0, width - 1), value=int(default_roi[0]), step=1)
with r2:
    y = st.slider("y", min_value=0, max_value=max(0, height - 1), value=int(default_roi[1]), step=1)
with r3:
    rw = st.number_input("w", min_value=1, max_value=width, value=int(default_roi[2]), step=1)
with r4:
    rh = st.number_input("h", min_value=1, max_value=height, value=int(default_roi[3]), step=1)

roi = clamp_roi((int(x), int(y), int(rw), int(rh)), width=width, height=height)
st.session_state["roi_manual"] = roi

# Aperçu
preview = first_frame_rgb.copy()
cv2.rectangle(preview, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 3)
st.image(preview, caption="Aperçu ROI (rectangle vert)", use_container_width=True)

if roi is None:
    st.warning("Aucune ROI dessinée: l'analyse se fera sur l'image complète (moins fiable).")
else:
    st.caption(f"ROI actuelle: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")

with st.expander("Aide / conseils ROI", expanded=False):
    st.markdown(
        """
- Choisis une zone où les karts passent **toujours** (ligne, virage, etc.).
- Plus le kart est gros dans la ROI, plus le modèle est fiable.
- Si tu ne détectes rien, baisse le seuil (ex: 0.45–0.60) et mets `min_consecutive=2`.
"""
    )

run_btn = st.button("Lancer l'analyse", type="primary")

if run_btn:
    # Créer un dossier de run uniquement au click (évite d'en créer plein au moindre changement de widget)
    run_dir = ensure_run_dir(output_base)
    st.session_state["last_run_dir"] = str(run_dir)

    detector = VideoKartDetector(model_path=str(model_choice), video_path=str(video_path), roi=roi)

    progress = st.progress(0)
    status = st.empty()

    def on_progress(frame_idx, total):
        if total and total > 0:
            p = min(1.0, max(0.0, frame_idx / float(total)))
            progress.progress(int(p * 100))
            status.text(f"Analyse... frame {frame_idx}/{total}")

    t0 = time.time()
    results, detections = detector.detect_in_video(
        output_dir=str(run_dir),
        confidence_threshold=float(threshold),
        save_frames=bool(save_frames),
        frame_skip=int(frame_skip),
        min_consecutive=int(min_consecutive),
        dump_roi_samples=int(dump_roi_samples),
        progress_callback=on_progress,
    )
    detector.save_results(str(run_dir))
    dt = time.time() - t0

    progress.progress(100)
    status.text(f"Terminé en {dt:.1f}s")

    st.session_state["last_run_summary"] = {
        "frames_analysées": len(results),
        "detections": len(detections),
        "roi": roi,
        "output_dir": str(run_dir),
    }


# Affichage des résultats du dernier run (évite de recalculer/relister à chaque rerun)
last_run_dir = st.session_state.get("last_run_dir")
last_summary = st.session_state.get("last_run_summary")

if last_run_dir and Path(last_run_dir).exists() and last_summary:
    run_dir = Path(last_run_dir)
    st.divider()
    st.subheader("Dernier résultat")
    st.write(last_summary)

    # Top scores: recharge depuis le JSON si présent
    results_json = run_dir / "detection_results.json"
    if results_json.exists():
        import json

        data = json.loads(results_json.read_text(encoding="utf-8"))
        detections_all = data.get("all_detections", [])
        if detections_all:
            top = sorted(detections_all, key=lambda r: r.get("probability", 0.0), reverse=True)[:10]
            st.subheader("Top 10 scores")
            st.dataframe(top, use_container_width=True)

    frames_dir = run_dir / "frames"
    roi_samples_dir = run_dir / "roi_samples"

    st.subheader("Images générées")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frames détectées**")
        if frames_dir.exists():
            imgs = sorted(frames_dir.glob("*.jpg"))
            st.image([str(p) for p in imgs[:30]], use_container_width=True)
        else:
            st.info("Aucune frame détectée sauvegardée.")
    with c2:
        st.markdown("**ROI samples (debug)**")
        if roi_samples_dir.exists():
            imgs = sorted(roi_samples_dir.glob("*.jpg"))
            st.image([str(p) for p in imgs[:30]], use_container_width=True)
        else:
            st.info("Aucun ROI sample.")

    st.subheader("Fichiers de sortie")
    st.write(
        {
            "detection_results": str(run_dir / "detection_results.json"),
            "summary": str(run_dir / "summary.txt"),
        }
    )
