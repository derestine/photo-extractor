#!/usr/bin/env python3
"""Diagnostic for the refined multi-photo extractor."""
from pathlib import Path
import cv2
import numpy as np
from photo_extractor import load_image, find_photo_quads, _destination_size

RAW = Path("raw")
OUT = Path("diag_out")
OUT.mkdir(exist_ok=True)

for f in sorted(RAW.glob("*.HEIC")):
    img = load_image(f)
    if img is None:
        print(f"{f.name}: LOAD FAILED")
        continue
    ih, iw = img.shape[:2]
    quads = find_photo_quads(img)
    # Viz input with all detected quads
    viz = img.copy()
    for q in quads:
        cv2.polylines(viz, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 10)
    cv2.imwrite(str(OUT / f"{f.stem}_viz.jpg"), cv2.resize(viz, (iw // 5, ih // 5)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    sizes = []
    covs = []
    for q in quads:
        w, h = _destination_size(q)
        sizes.append(f"{w}x{h}")
        covs.append(f"{100 * cv2.contourArea(q.reshape(-1,1,2)) / (iw*ih):.0f}%")
    if quads:
        print(f"{f.name}: {len(quads)} photo(s)  cov=[{', '.join(covs)}]  out=[{', '.join(sizes)}]")
    else:
        print(f"{f.name}: NO QUAD (fallback)")
