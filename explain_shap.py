import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def _parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI invalide. Format attendu: x,y,w,h")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("ROI invalide: w et h doivent être > 0")
    return x, y, w, h


def _list_images(paths: List[str]) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in exts:
                    out.append(child)
        elif path.is_file() and path.suffix.lower() in exts:
            out.append(path)
    return out


def _load_image_as_uint8_rgb(image_path: Path, size: int, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")

    if roi is not None:
        x, y, w, h = roi
        # Clamp ROI to image bounds
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img.width, x + w)
        y1 = min(img.height, y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"ROI hors image pour {image_path.name}: {roi} sur {img.width}x{img.height}")
        img = img.crop((x0, y0, x1, y1))

    if img.size != (size, size):
        img = img.resize((size, size))

    arr = np.asarray(img, dtype=np.uint8)  # (H,W,3) 0-255
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainability SHAP pour le modèle kart/no_kart (PyTorch).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin vers le checkpoint .pth (par défaut: dernier modèle dans models/)",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Images ou dossier(s) d'images (jpg/png/bmp).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="shap_outputs",
        help="Dossier de sortie (png).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Nombre max d'images à expliquer (évite des runs très longs).",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optionnel: ROI à appliquer avant explication (format x,y,w,h).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=0,
        help="(Non utilisé avec GradientExplainer; gardé pour compat CLI).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch interne SHAP.",
    )
    args = parser.parse_args()

    # Imports lourds après parsing
    import torch
    import torch.nn as nn
    from torchvision import models
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_dir = Path(__file__).resolve().parent

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        models_dir = project_dir / "models"
        candidates = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("Aucun modèle .pth trouvé dans models/. Fournis --model.")
        model_path = candidates[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint + model
    checkpoint = torch.load(str(model_path), map_location=device)

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_to_idx = checkpoint.get("class_to_idx")
    # By design, output sigmoid = P(class index 1).
    # We expose outputs as [P(no_kart), P(go_kart)] to make SHAP easier to read.
    if isinstance(class_to_idx, dict) and "go_kart" in class_to_idx:
        go_kart_index = int(class_to_idx["go_kart"])
    else:
        # Fallback consistent with previous assumptions in predict.py
        go_kart_index = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess_uint8_batch_to_tensor(images_nhwc_uint8: np.ndarray) -> torch.Tensor:
        """Convertit (N,224,224,3) uint8 RGB -> tensor float normalisé (N,3,224,224)."""
        if images_nhwc_uint8.ndim != 4 or images_nhwc_uint8.shape[-1] != 3:
            raise ValueError("Entrée attendue: (N,H,W,3)")
        x = images_nhwc_uint8.astype(np.float32) / 255.0
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        return (tensor - mean) / std

    class GoKartProbabilityModel(nn.Module):
        def __init__(self, base: nn.Module, go_idx: int):
            super().__init__()
            self.base = base
            self.go_idx = int(go_idx)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            p_class1 = self.base(x).squeeze(1)  # (N,)
            p_go = p_class1 if self.go_idx == 1 else (1.0 - p_class1)
            return p_go.unsqueeze(1)  # (N,1)

    gokart_model = GoKartProbabilityModel(model, go_kart_index)
    gokart_model.eval()

    roi = _parse_roi(args.roi)
    image_paths = _list_images(args.inputs)
    if not image_paths:
        raise FileNotFoundError("Aucune image trouvée dans --inputs")

    if args.limit is not None and args.limit > 0:
        image_paths = image_paths[: int(args.limit)]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SHAP setup
    # IMPORTANT:
    # - shap.maskers.Image utilise numba et casse avec numpy 2.4 (np.trapz supprimé).
    # - On utilise donc GradientExplainer (basé gradients PyTorch), compatible ici.
    img_size = 224

    # Background: quelques images (par défaut, premières images d'entrée)
    bg_count = min(8, len(image_paths))
    bg_imgs: List[np.ndarray] = []
    for p in image_paths[:bg_count]:
        try:
            bg_imgs.append(_load_image_as_uint8_rgb(p, size=img_size, roi=roi))
        except Exception:
            continue
    if not bg_imgs:
        raise RuntimeError("Impossible de construire le background SHAP (aucune image valide).")

    background_tensor = preprocess_uint8_batch_to_tensor(np.stack(bg_imgs, axis=0))
    explainer = shap.GradientExplainer(gokart_model, background_tensor)

    print(f"Modèle: {model_path}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")
    if roi is not None:
        print(f"ROI appliquée avant explication: {roi}")
    print(f"Sortie: {out_dir}")

    for img_path in image_paths:
        try:
            img_uint8 = _load_image_as_uint8_rgb(img_path, size=img_size, roi=roi)
        except Exception as e:
            print(f"[SKIP] {img_path.name}: {e}")
            continue

        images_uint8 = np.expand_dims(img_uint8, axis=0)
        image_tensor = preprocess_uint8_batch_to_tensor(images_uint8)

        with torch.no_grad():
            p_go = float(gokart_model(image_tensor).squeeze().detach().cpu().item())
        print(f"{img_path.name}: P(go_kart)={p_go:.3f}")

        # shap_values retourne souvent une liste (une entrée par output).
        shap_values = explainer.shap_values(image_tensor)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Normaliser la forme en (N, 3, H, W)
        shap_values = np.asarray(shap_values)

        # Certains retours incluent une dimension de sortie singleton (… , 1)
        if shap_values.ndim >= 4 and shap_values.shape[-1] == 1:
            shap_values = np.squeeze(shap_values, axis=-1)

        if shap_values.ndim == 5:
            # Cas fréquents:
            # - (N, 1, 3, H, W)
            # - (1, N, 3, H, W)
            if shap_values.shape[1] == 1:
                shap_values = shap_values[:, 0]
            elif shap_values.shape[0] == 1:
                shap_values = shap_values[0]
        elif shap_values.ndim == 3:
            # (3, H, W) pour une seule image
            shap_values = shap_values[None, ...]

        if shap_values.ndim != 4:
            raise RuntimeError(f"Forme SHAP inattendue pour {img_path.name}: {shap_values.shape}")

        # Convert (N,3,H,W) -> (N,H,W,3) pour l'affichage
        if shap_values.shape[1] == 3:
            shap_values_nhwc = np.transpose(shap_values, (0, 2, 3, 1))
        elif shap_values.shape[-1] == 3:
            shap_values_nhwc = shap_values
        else:
            raise RuntimeError(f"Canaux inattendus pour {img_path.name}: {shap_values.shape}")

        plt.figure(figsize=(10, 4))
        shap.image_plot([shap_values_nhwc], images_uint8, show=False)

        out_path = out_dir / f"{img_path.stem}_shap_go_kart.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=180)
        plt.close("all")

    print("Terminé.")


if __name__ == "__main__":
    main()
