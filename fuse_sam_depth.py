"""
fuse_sam_depth.py - Profondeur de chaque masque SAM

Prend:
  - depth.npy + depth.json (sortie de run_depth_paysagea.py)
  - sam_output.json

Produit VisionOutput.json avec pour CHAQUE segment:
  - mean_depth, depth_std, depth_band (front/mid/back)

Modifier les chemins en haut du fichier selon ton image.
"""

import json
import numpy as np
from pycocotools import mask as mask_utils
from pathlib import Path

# ============ À ADAPTER selon ton image ============
# Base du nom (sans _depth, _preprocessed, etc.)
BASE = "WhatsApp Image 2026-02-10 at 14.05.13-400x225"
DEPTH_NPY = Path(f"Outputs/{BASE}_depth.npy")
DEPTH_JSON = Path(f"Outputs/{BASE}_depth.json")
SAM_JSON = Path(f"../Inputs/{BASE}_sam_output.json")
PREPROCESS_JSON = Path(f"../Inputs/{BASE}_preprocessed.json")
OUT_JSON = Path("VisionOutput.json")
OUT_MASKS_DIR = Path("Outputs/masks")  # Un JSON par masque
# =================================================

print("Loading depth map...")
depth = np.load(DEPTH_NPY)

print("Loading depth metadata...")
with open(DEPTH_JSON, "r", encoding="utf-8") as f:
    depth_meta = json.load(f)

print("Loading SAM output...")
with open(SAM_JSON, "r", encoding="utf-8") as f:
    sam_data = json.load(f)

print("Loading preprocess metadata...")
preprocess_meta = {}
if PREPROCESS_JSON.exists():
    with open(PREPROCESS_JSON, "r", encoding="utf-8") as f:
        preprocess_meta = json.load(f)
else:
    print(f"⚠️ Preprocess metadata not found: {PREPROCESS_JSON}")

print("Depth shape:", depth.shape)
print("SAM segments:", len(sam_data["sam_output"]["segments"]))

H, W = depth.shape

# Vérif alignement
sam_size = sam_data["sam_output"].get("image_size")
if sam_size and sam_size != [W, H]:
    raise ValueError(f"SAM image_size {sam_size} != depth {[W,H]}")

near_is_one = bool(depth_meta.get("near_is_one", True))


def depth_band(x: float) -> str:
    """front=proche, mid=milieu, back=loin"""
    if x >= 0.66:
        return "front"
    if x >= 0.33:
        return "mid"
    return "back"


print("\nComputing depth for each mask...")
segments_out = []

for seg in sam_data["sam_output"]["segments"]:
    seg_id = seg["segment_id"]
    rle = seg["mask_rle"]

    # RLE peut être dict avec "counts" string ou list
    if isinstance(rle, dict) and "size" in rle:
        mask = mask_utils.decode(rle).astype(bool)
    else:
        mask = mask_utils.decode({"size": [H, W], "counts": rle}).astype(bool)

    if mask.shape != (H, W):
        mask = np.ascontiguousarray(mask.transpose() if mask.shape == (W, H) else mask)
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} != depth {depth.shape}")

    vals = depth[mask]
    if vals.size == 0:
        mean_depth = None
        depth_std = None
        band = None
        min_depth = max_depth = None
    else:
        mean_depth = float(vals.mean())
        depth_std = float(vals.std())
        band = depth_band(mean_depth)
        min_depth = float(vals.min())
        max_depth = float(vals.max())

    seg_enriched = dict(seg)
    seg_enriched["mean_depth"] = mean_depth
    seg_enriched["depth_std"] = depth_std
    seg_enriched["depth_band"] = band
    seg_enriched["min_depth"] = min_depth
    seg_enriched["max_depth"] = max_depth
    seg_enriched["num_pixels"] = int(mask.sum())
    segments_out.append(seg_enriched)

vision_output = {
    "version": "vision_segments_v1",
    "image_id": preprocess_meta.get("image_id") or sam_data.get("image_id") or depth_meta.get("image_id"),
    "image_size": [W, H],
    "preprocess": preprocess_meta,
    "depth_meta": {
        "model": depth_meta.get("model", "LiheYoung/depth_anything_vitl14"),
        "near_is_one": near_is_one,
        "depth_range": depth_meta.get("depth_range", [0.0, 1.0]),
        "normalized": depth_meta.get("normalized", True),
        "depth_file": str(DEPTH_NPY),
    },
    "sam_meta": {
        "sam_file": str(SAM_JSON),
        "segments_count": len(segments_out),
    },
    "segments": segments_out,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(vision_output, f, indent=2)

# Un JSON par masque
OUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
for seg in segments_out:
    seg_id = seg["segment_id"]
    mask_data = {
        "segment_id": seg_id,
        "mean_depth": seg.get("mean_depth"),
        "depth_std": seg.get("depth_std"),
        "depth_band": seg.get("depth_band"),
        "min_depth": seg.get("min_depth"),
        "max_depth": seg.get("max_depth"),
        "num_pixels": seg.get("num_pixels"),
        "area_ratio": seg.get("area_ratio"),
        "centroid": seg.get("centroid"),
        "bbox": seg.get("bbox"),
    }
    mask_file = OUT_MASKS_DIR / f"mask_{seg_id}.json"
    with open(mask_file, "w", encoding="utf-8") as f:
        json.dump(mask_data, f, indent=2)

print(f"\n✅ Saved {OUT_JSON} with {len(segments_out)} segments.")
print(f"✅ Saved {len(segments_out)} fichiers JSON individuels dans {OUT_MASKS_DIR}/ (mask_0.json, mask_1.json, ...)")
print("\nExemple segment 0:", {
    k: vision_output["segments"][0].get(k)
    for k in ["segment_id", "mean_depth", "depth_std", "depth_band"]
})
