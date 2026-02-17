"""
export_vision_v0.py - Génère main.json (VisionOutput v0) pour le LLM

Lit VisionOutput.json (segments + depth) et produit le format simplifié v0 :
- zones (lower_area, mid_area, upper_area)
- anchors (points sûrs pour placement)
- constraints (zones interdites)

Le LLM reçoit UNIQUEMENT ce fichier main.json.
"""

import json
import argparse
from pathlib import Path

# Zones verticales (y normalisé 0=haut, 1=bas)
LOWER_Y = (0.65, 1.0)   # lower_area
MID_Y = (0.35, 0.65)    # mid_area
UPPER_Y = (0.0, 0.35)   # upper_area

MIN_ANCHOR_DISTANCE = 0.12  # distance min entre 2 anchors (normalisée)
MAX_ANCHORS = 15


def segment_in_zone(centroid_y: float, zone: tuple) -> bool:
    return zone[0] <= centroid_y <= zone[1]


def compute_zone_coverage(segments: list, zone: tuple) -> float:
    """Coverage = somme des area_ratio des segments dont le centroid.y est dans la zone"""
    total = 0.0
    for seg in segments:
        cy = seg.get("centroid", [0.5, 0.5])[1]
        if segment_in_zone(cy, zone):
            total += seg.get("area_ratio", 0)
    return round(total, 2)


def anchors_from_segments(segments: list, image_size: list) -> list:
    """
    Option B : Anchors depuis segments SAM
    - centroid.y > 0.65 (lower_area ou mid_area bas)
    - depth_band = front ou mid
    - distance min entre anchors
    """
    w, h = image_size[0], image_size[1]
    candidates = []

    for seg in segments:
        centroid = seg.get("centroid", [0.5, 0.5])
        cx, cy = centroid[0], centroid[1]
        depth_band = seg.get("depth_band", "back")
        mean_depth = seg.get("mean_depth", 0)
        area_ratio = seg.get("area_ratio", 0)

        # Candidat : dans lower/mid bas, pas back
        if cy < 0.65:
            continue
        if depth_band not in ("front", "mid"):
            continue
        if area_ratio < 0.005:  # trop petit
            continue

        # Déterminer area
        if cy >= 0.65:
            area = "lower_area"
        elif cy >= 0.35:
            area = "mid_area"
        else:
            area = "upper_area"

        # Score : front > mid, plus proche = meilleur
        score = 0.92 if depth_band == "front" else 0.78
        score += min(mean_depth * 0.1, 0.08)  # légère bonus si proche
        score = round(min(score, 0.98), 2)

        candidates.append({
            "x": round(cx, 2),
            "y": round(cy, 2),
            "area": area,
            "depth_band": depth_band,
            "score": score,
            "area_ratio": area_ratio,
        })

    # Trier par score décroissant
    candidates.sort(key=lambda c: (c["score"], c["area_ratio"]), reverse=True)

    # Filtrer par distance min
    selected = []
    for c in candidates:
        too_close = False
        for s in selected:
            dx = c["x"] - s["x"]
            dy = c["y"] - s["y"]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < MIN_ANCHOR_DISTANCE:
                too_close = True
                break
        if not too_close:
            selected.append(c)
            if len(selected) >= MAX_ANCHORS:
                break

    # Formater avec id
    anchors = []
    for i, a in enumerate(selected, 1):
        anchors.append({
            "id": f"a{i}",
            "x": a["x"],
            "y": a["y"],
            "area": a["area"],
            "depth_band": a["depth_band"],
            "score": a["score"],
        })
    return anchors


def export_v0(vision_input: Path, main_output: Path) -> dict:
    """Produit main.json (VisionOutput v0) depuis VisionOutput.json"""
    with open(vision_input, "r", encoding="utf-8") as f:
        vision = json.load(f)

    segments = vision.get("segments", [])
    w, h = vision.get("image_size", [400, 225])
    image_id = vision.get("image_id", "unknown")
    near_is_one = vision.get("depth_meta", {}).get("near_is_one", True)

    # Coverage par zone
    cov_lower = compute_zone_coverage(segments, LOWER_Y)
    cov_mid = compute_zone_coverage(segments, MID_Y)
    cov_upper = compute_zone_coverage(segments, UPPER_Y)

    # Normaliser si besoin (total ~1)
    total = cov_lower + cov_mid + cov_upper
    if total > 0:
        cov_lower = round(cov_lower / total, 2)
        cov_mid = round(cov_mid / total, 2)
        cov_upper = round(cov_upper / total, 2)

    main_data = {
        "version": "vision_output_v0",
        "image_id": image_id,
        "image_info": {
            "width": w,
            "height": h,
        },
        "depth": {
            "near_is_one": near_is_one,
            "bands": {
                "front": [0.66, 1.0],
                "mid": [0.33, 0.66],
                "back": [0.0, 0.33],
            },
        },
        "zones": {
            "lower_area": {
                "coverage": cov_lower,
                "confidence": 0.70,
                "meaning": "Zone basse de l'image (souvent sol).",
            },
            "mid_area": {
                "coverage": cov_mid,
                "confidence": 0.60,
                "meaning": "Zone centrale (souvent fond / éléments verticaux).",
            },
            "upper_area": {
                "coverage": cov_upper,
                "confidence": 0.90,
                "meaning": "Zone haute (souvent ciel / haut de végétation).",
            },
        },
        "constraints": {
            "forbidden_areas": ["upper_area"],
            "notes": [
                "Ne pas placer d'éléments dans la zone haute.",
                "Préférer les anchors de lower_area pour des objets au sol.",
            ],
        },
        "anchors": anchors_from_segments(segments, [w, h]),
    }

    with open(main_output, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2)

    print(f"✅ main.json (VisionOutput v0) sauvegardé : {main_output}")
    print(f"   - {len(main_data['anchors'])} anchors")
    print(f"   - zones: lower={cov_lower}, mid={cov_mid}, upper={cov_upper}")
    return main_data


def main():
    parser = argparse.ArgumentParser(description="Export VisionOutput v0 (main.json) pour LLM")
    parser.add_argument("--input", default="VisionOutput.json", help="VisionOutput.json (segments+depth)")
    parser.add_argument("--output", default="main.json", help="Fichier de sortie pour LLM")
    args = parser.parse_args()

    vision_path = Path(args.input)
    main_path = Path(args.output)
    if not vision_path.exists():
        print(f"❌ Fichier introuvable : {vision_path}")
        print("   Lance d'abord : python fuse_sam_depth.py")
        return 1
    export_v0(vision_path, main_path)
    return 0


if __name__ == "__main__":
    exit(main())
