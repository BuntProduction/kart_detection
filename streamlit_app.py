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

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from video_detector import VideoKartDetector
from collect_false_positives import collect_false_positive_candidates


APP_TITLE = "Kart detector (vidéo)"


CACHE_DIR = Path(".streamlit_cache")
CACHE_UPLOADS_DIR = CACHE_DIR / "uploads"

# Configuration MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')


def list_mlflow_models():
    """Lister les modèles disponibles dans MLflow"""
    if not MLFLOW_AVAILABLE:
        return []
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        
        # Récupérer les versions du modèle enregistré
        model_name = "kart_detector"
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            return [(f"MLflow: {model_name}/v{v.version}", f"models:/{model_name}/{v.version}") 
                    for v in sorted(versions, key=lambda x: int(x.version), reverse=True)]
        except Exception:
            return []
    except Exception:
        return []


def list_models(models_dir: str = "models"):
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    local_models = sorted(models_path.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Combiner avec les modèles MLflow
    all_models = [(f"Local: {p.name}", str(p)) for p in local_models]
    mlflow_models = list_mlflow_models()
    all_models.extend(mlflow_models)
    
    return all_models


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


def ensure_false_positives_dir(base_dir: str = "faux_positifs") -> Path:
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


def _load_image_uint8_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _apply_roi_uint8(img_rgb: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x, y, rw, rh = clamp_roi(tuple(map(int, roi)), width=w, height=h)
    return img_rgb[y : y + rh, x : x + rw]


@st.cache_resource(show_spinner=False)
def _load_shap_gokart_model(model_path: str):
    """Charge le modèle et expose une sortie scalaire P(go_kart).

    Utilise la même convention que predict.py:
    - La tête Sigmoid produit P(classe index 1)
    - On convertit vers P(go_kart) selon class_to_idx (si présent)
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    base = models.resnet18(pretrained=False)
    num_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    base.load_state_dict(checkpoint["model_state_dict"])
    base.to(device)
    base.eval()

    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict) and "go_kart" in class_to_idx:
        go_kart_index = int(class_to_idx["go_kart"])
    else:
        go_kart_index = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess_uint8_batch(images_nhwc_uint8: np.ndarray) -> "torch.Tensor":
        x = np.clip(images_nhwc_uint8, 0, 255).astype(np.float32) / 255.0
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        return (tensor - mean) / std

    class GoKartProbabilityModel(nn.Module):
        def __init__(self, base_model: nn.Module, go_idx: int):
            super().__init__()
            self.base = base_model
            self.go_idx = int(go_idx)

        def forward(self, x):
            p_class1 = self.base(x).squeeze(1)
            p_go = p_class1 if self.go_idx == 1 else (1.0 - p_class1)
            return p_go.unsqueeze(1)

    gokart_model = GoKartProbabilityModel(base, go_kart_index)
    gokart_model.eval()
    return gokart_model, preprocess_uint8_batch, device


@st.cache_resource(show_spinner=False)
def _load_shap_background_uint8(limit: int = 8) -> list[np.ndarray]:
    """Background stable pour SHAP (images du circuit vide si disponibles)."""
    candidates: list[np.ndarray] = []
    empty_dir = Path("extracted_empty_circuit")
    if empty_dir.exists():
        for p in sorted(empty_dir.glob("*.jpg"))[:limit]:
            try:
                candidates.append(_load_image_uint8_rgb(p))
            except Exception:
                continue
    return candidates


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
    format_func=lambda p: p[0],  # p est un tuple (nom, chemin)
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
st.image(preview, caption="Aperçu ROI (rectangle vert)", width="stretch")

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

st.divider()
st.subheader("Récupérer les faux positifs (à trier)")
st.caption(
    "Collecte automatiquement les frames/ROI dont le score est dans une zone grise (ex: 0.50–0.80). "
    "Tu pourras ensuite trier manuellement et réutiliser ces images comme négatifs."
)

fp_c1, fp_c2, fp_c3 = st.columns([1, 1, 1])
with fp_c1:
    fp_min = st.slider("Seuil min (candidats)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
with fp_c2:
    fp_max = st.slider("Seuil max (candidats)", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
with fp_c3:
    fp_frame_skip = st.number_input("Frame skip (FP)", min_value=1, max_value=60, value=int(frame_skip), step=1)

fp_max_samples = st.number_input("Max candidats à sauvegarder", min_value=10, max_value=5000, value=500, step=50)
fp_min_gap = st.number_input("Écart min entre sauvegardes (frames)", min_value=0, max_value=300, value=10, step=1)
fp_save_full = st.checkbox("Sauver aussi la frame complète annotée", value=False)

fp_btn = st.button("Recuperer les faux positifs", type="secondary")

if run_btn:
    # Créer un dossier de run uniquement au click (évite d'en créer plein au moindre changement de widget)
    run_dir = ensure_run_dir(output_base)
    st.session_state["last_run_dir"] = str(run_dir)

    # model_choice est un tuple (nom, chemin)
    model_path = model_choice[1]
    detector = VideoKartDetector(model_path=str(model_path), video_path=str(video_path), roi=roi)

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


if fp_btn:
    if fp_min > fp_max:
        st.error("Le seuil min doit être <= au seuil max.")
    else:
        fp_run_dir = ensure_false_positives_dir("faux_positifs")
        st.session_state["last_fp_dir"] = str(fp_run_dir)

        progress = st.progress(0)
        status = st.empty()

        def on_progress_fp(frame_idx, total):
            if total and total > 0:
                p = min(1.0, max(0.0, frame_idx / float(total)))
                progress.progress(int(p * 100))
                status.text(f"Collecte FP... frame {frame_idx}/{total}")

        with st.spinner("Collecte des candidats faux-positifs (ça peut prendre un peu de temps)..."):
            # model_choice est un tuple (nom, chemin)
            model_path = model_choice[1]
            samples = collect_false_positive_candidates(
                model_path=str(model_path),
                video_path=str(video_path),
                roi=roi,
                out_dir=str(fp_run_dir),
                min_score=float(fp_min),
                max_score=float(fp_max),
                frame_skip=int(fp_frame_skip),
                max_samples=int(fp_max_samples),
                min_gap_frames=int(fp_min_gap),
                save_full_frames=bool(fp_save_full),
                progress_callback=on_progress_fp,
            )

        progress.progress(100)
        status.text(f"Terminé: {len(samples)} candidats sauvegardés dans {fp_run_dir}")
        st.success(f"OK: {len(samples)} candidats sauvegardés")

        # Aperçu
        roi_dir = fp_run_dir / "roi"
        if roi_dir.exists():
            imgs = sorted(roi_dir.glob("*.jpg"))
            if imgs:
                st.image([str(p) for p in imgs[:30]], width="stretch")
        st.write({"manifest": str(fp_run_dir / "manifest.json")})


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
            st.dataframe(top, width="stretch")

    frames_dir = run_dir / "frames"
    roi_samples_dir = run_dir / "roi_samples"

    st.subheader("Images générées")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frames détectées**")
        if frames_dir.exists():
            imgs = sorted(frames_dir.glob("*.jpg"))
            st.image([str(p) for p in imgs[:30]], width="stretch")
        else:
            st.info("Aucune frame détectée sauvegardée.")
    with c2:
        st.markdown("**ROI samples (debug)**")
        if roi_samples_dir.exists():
            imgs = sorted(roi_samples_dir.glob("*.jpg"))
            st.image([str(p) for p in imgs[:30]], width="stretch")
        else:
            st.info("Aucun ROI sample.")

    st.divider()
    st.subheader("Explainability (SHAP)")
    st.caption("Génère une heatmap SHAP pour voir quelles zones poussent la prédiction vers 'go_kart'.")

    # Sources possibles pour l'image à expliquer
    available_frame_imgs = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    available_roi_imgs = sorted(roi_samples_dir.glob("*.jpg")) if roi_samples_dir.exists() else []

    src = st.radio(
        "Choisir l'image à expliquer",
        options=["Frame détectée", "ROI sample", "Upload image"],
        horizontal=True,
    )

    chosen_path: Optional[Path] = None
    uploaded_img = None
    apply_roi = False

    if src == "Frame détectée":
        if not available_frame_imgs:
            st.info("Aucune frame détectée disponible. Active 'Sauvegarder les frames détectées' puis relance l'analyse.")
        else:
            chosen_path = st.selectbox(
                "Frame",
                options=available_frame_imgs,
                format_func=lambda p: p.name,
            )
            apply_roi = st.checkbox("Appliquer la ROI avant SHAP", value=True)

    elif src == "ROI sample":
        if not available_roi_imgs:
            st.info("Aucun ROI sample disponible. Mets 'Sauver N crops ROI (debug)' > 0 puis relance l'analyse.")
        else:
            chosen_path = st.selectbox(
                "ROI sample",
                options=available_roi_imgs,
                format_func=lambda p: p.name,
            )
            apply_roi = st.checkbox("Appliquer la ROI avant SHAP", value=False, disabled=True)

    else:
        uploaded_img = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png", "bmp"], key="shap_upload")
        apply_roi = st.checkbox("Appliquer la ROI avant SHAP", value=False)

    explain_btn = st.button("Afficher SHAP", type="secondary")

    if explain_btn:
        if uploaded_img is None and chosen_path is None:
            st.warning("Choisis une image (ou upload) avant de lancer SHAP.")
        else:
            with st.spinner("Calcul SHAP (ça peut prendre ~10-60s sur CPU)..."):
                import shap
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Charger l'image
                if uploaded_img is not None:
                    img = Image.open(uploaded_img).convert("RGB")
                    img_rgb = np.asarray(img, dtype=np.uint8)
                    img_name = Path(uploaded_img.name).stem
                else:
                    img_rgb = _load_image_uint8_rgb(chosen_path)  # type: ignore[arg-type]
                    img_name = chosen_path.stem  # type: ignore[union-attr]

                # Appliquer ROI si demandé
                roi_for_shap = last_summary.get("roi") if isinstance(last_summary, dict) else None
                if apply_roi and roi_for_shap is not None:
                    img_rgb = _apply_roi_uint8(img_rgb, roi_for_shap)

                # Redimensionner à 224x224
                img_resized = np.asarray(Image.fromarray(img_rgb).resize((224, 224)), dtype=np.uint8)
                image_batch_uint8 = np.expand_dims(img_resized, axis=0)

                # model_choice est un tuple (nom, chemin)
                model_path = model_choice[1]
                gokart_model, preprocess_uint8_batch, device = _load_shap_gokart_model(str(model_path))

                # Background: circuit vide si dispo, sinon image courante répétée
                bg_imgs = _load_shap_background_uint8(limit=8)
                bg_ready: list[np.ndarray] = []
                for b in bg_imgs:
                    b2 = b
                    if apply_roi and roi_for_shap is not None:
                        try:
                            b2 = _apply_roi_uint8(b2, roi_for_shap)
                        except Exception:
                            b2 = b
                    b2 = np.asarray(Image.fromarray(b2).resize((224, 224)), dtype=np.uint8)
                    bg_ready.append(b2)

                if not bg_ready:
                    bg_ready = [img_resized for _ in range(4)]

                import torch

                background_tensor = preprocess_uint8_batch(np.stack(bg_ready, axis=0))
                image_tensor = preprocess_uint8_batch(image_batch_uint8)

                with torch.no_grad():
                    p_go = float(gokart_model(image_tensor).squeeze().detach().cpu().item())

                explainer = shap.GradientExplainer(gokart_model, background_tensor)
                shap_values = explainer.shap_values(image_tensor)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap_values = np.asarray(shap_values)
                if shap_values.ndim >= 4 and shap_values.shape[-1] == 1:
                    shap_values = np.squeeze(shap_values, axis=-1)
                if shap_values.ndim == 5:
                    if shap_values.shape[1] == 1:
                        shap_values = shap_values[:, 0]
                    elif shap_values.shape[0] == 1:
                        shap_values = shap_values[0]
                elif shap_values.ndim == 3:
                    shap_values = shap_values[None, ...]

                if shap_values.ndim != 4:
                    raise RuntimeError(f"Forme SHAP inattendue: {shap_values.shape}")

                if shap_values.shape[1] == 3:
                    shap_values_nhwc = np.transpose(shap_values, (0, 2, 3, 1))
                elif shap_values.shape[-1] == 3:
                    shap_values_nhwc = shap_values
                else:
                    raise RuntimeError(f"Canaux SHAP inattendus: {shap_values.shape}")

                st.write({"P(go_kart)": round(p_go, 4), "device": str(device)})

                plt.figure(figsize=(10, 4))
                shap.image_plot([shap_values_nhwc], image_batch_uint8, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)

                # Sauvegarde optionnelle dans le run_dir
                shap_dir = run_dir / "shap"
                shap_dir.mkdir(exist_ok=True)
                out_path = shap_dir / f"{img_name}_shap_go_kart.png"
                plt.savefig(out_path, bbox_inches="tight", dpi=180)
                plt.close("all")
                st.success(f"SHAP sauvegardé: {out_path}")

    st.subheader("Fichiers de sortie")
    st.write(
        {
            "detection_results": str(run_dir / "detection_results.json"),
            "summary": str(run_dir / "summary.txt"),
        }
    )
