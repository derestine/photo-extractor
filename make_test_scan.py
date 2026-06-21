#!/usr/bin/env python3
"""Generate a synthetic 'phone scan of a photo' for testing the extractor.

Creates a photo with recognisable content, places it rotated/perspective-skewed
on a textured-ish background (a 'floor'), and adds a white matte border around
the photo. This exercises the detection + perspective correction + white-crop.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make_photo(w: int = 900, h: int = 1200) -> np.ndarray:
    """A faux landscape photo with a sky, sun, ground, and some 'people'."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(h // 2):
        t = y / (h / 2)
        img[y, :, 0] = int(220 - 60 * t)   # B
        img[y, :, 1] = int(210 - 80 * t)   # G
        img[y, :, 2] = int(255 - 30 * t)   # R
    # Ground
    img[h // 2 :, :, :] = (60, 110, 80)  # greenish
    # Sun
    cv2.circle(img, (int(w * 0.75), int(h * 0.2)), 60, (60, 200, 240), -1)
    # Tree trunks / people (vertical bars)
    for x in (120, 300, 480, 660):
        cv2.rectangle(img, (x, int(h * 0.45)), (x + 24, int(h * 0.85)), (40, 70, 50), -1)
    # Text so we can see if the warp is upright / readable
    cv2.putText(img, "SAMPLE PHOTO", (60, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return img


def add_white_matte(photo: np.ndarray, border: int = 40) -> np.ndarray:
    h, w = photo.shape[:2]
    matte = np.full((h + 2 * border, w + 2 * border, 3), 255, dtype=np.uint8)
    matte[border : border + h, border : border + w] = photo
    return matte


def place_on_floor(photo: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    """Warp the photo with a perspective tilt and drop it on a dark floor."""
    ch, cw = canvas_hw
    floor = np.full((ch, cw, 3), 50, dtype=np.uint8)
    # Speckle the floor so it has texture for edge detection to ignore.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 25, size=floor.shape, dtype=np.uint8)
    floor = cv2.add(floor, noise)

    ph, pw = photo.shape[:2]
    # Destination quad — slightly trapezoidal to simulate camera tilt.
    margin_x, top_y, bot_y = 180, 220, ch - 180
    dst = np.array(
        [[margin_x + 40, top_y], [cw - margin_x - 40, top_y],
         [cw - margin_x, bot_y], [margin_x, bot_y]],
        dtype="float32",
    )
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(photo, matrix, (cw, ch))

    # Composite with a HARD polygon mask so the photo's edge against the floor
    # is crisp (like a real scan), not softened by warp interpolation.
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(dst).astype(np.int32)], 255)
    floor = np.where(mask[..., None] > 0, warped, floor)
    return floor.astype(np.uint8)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_scan.png")
    photo = make_photo()
    matte = add_white_matte(photo, border=50)
    scan = place_on_floor(matte, canvas_hw=(1600, 1400))
    cv2.imwrite(str(out), scan)
    print(f"Wrote synthetic scan to {out} ({scan.shape[1]}x{scan.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
