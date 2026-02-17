"""
run_depth_paysagea.py - Depth Anything conforme au contrat Paysagea

Règle #1: Utilise UNIQUEMENT *_preprocessed.jpg et *_preprocessed.json
Règle #2: Depth alignée pixel à pixel (même taille que resized_size)
Règle #3: Documente la normalisation (near_is_one, normalized [0..1])

Usage:
  python run_depth_paysagea.py --img Inputs/XXX_preprocessed.jpg --meta Inputs/XXX_preprocessed.json --outdir Outputs --near-is-one
"""

import argparse
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to *_preprocessed.jpg (ou .png)")
    parser.add_argument("--meta", required=True, help="Path to *_preprocessed.json")
    parser.add_argument("--outdir", default="Outputs", help="Dossier de sortie")
    parser.add_argument("--near-is-one", action="store_true", default=True,
                        help="1=proche, 0=loin (convention Paysagea)")
    args = parser.parse_args()

    img_path = Path(args.img)
    meta_path = Path(args.meta)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Charger meta (source de vérité)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Lire image pré-traitée
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    h, w = img_bgr.shape[:2]
    print(f"Taille image chargée: {w}×{h} (largeur×hauteur)")

    # Vérifier taille vs resized_size et redimensionner si nécessaire
    preprocess = meta.get("preprocess", {})
    expected = meta.get("resized_size") or preprocess.get("resized_size") or meta.get("image_size") or meta.get("size")
    if isinstance(expected, list) and len(expected) == 2:
        ew, eh = expected[0], expected[1]
        print(f"Taille attendue (resized_size): {ew}×{eh}")
        if (w, h) != (ew, eh):
            print(f"⚠️  Taille différente → redimensionnement {w}×{h} → {ew}×{eh}")
            img_bgr = cv2.resize(img_bgr, (ew, eh), interpolation=cv2.INTER_CUBIC)
            w, h = ew, eh
            print(f"✓  Image redimensionnée à {w}×{h}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to(device).eval()

    transform = Compose([
        Resize(
            width=w,
            height=h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    sample = transform({"image": img_rgb})["image"]
    sample = torch.from_numpy(sample).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(sample)
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

    # Normaliser [0, 1]
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - dmin) / (dmax - dmin)

    # Convention: near_is_one => 1=proche, 0=loin
    if args.near_is_one:
        depth_norm = 1.0 - depth_norm

    # Noms de sortie
    base = img_path.stem.replace("_preprocessed", "").replace("_01", "")
    depth_npy = outdir / f"{base}_depth.npy"
    depth_preview = outdir / f"{base}_depth_preview.png"
    depth_json = outdir / f"{base}_depth.json"

    np.save(depth_npy, depth_norm)

    preview8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(depth_preview), preview8)

    image_id = meta.get("image_id") or sha256_of_file(img_path)
    preprocessed_filename = meta.get("preprocessed_filename") or img_path.name

    payload = {
        "version": "depth_output_v1",
        "image_id": image_id,
        "preprocessed_filename": preprocessed_filename,
        "image_size": [w, h],
        "depth_file": depth_npy.name,
        "depth_preview": depth_preview.name,
        "depth_range": [0.0, 1.0],
        "normalized": True,
        "near_is_one": bool(args.near_is_one),
        "model": "LiheYoung/depth_anything_vitl14",
        "notes": "Depth alignée pixel à pixel sur image pré-traitée."
    }

    with open(depth_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:")
    print(" ", depth_npy)
    print(" ", depth_json)
    print(" ", depth_preview)


if __name__ == "__main__":
    main()
