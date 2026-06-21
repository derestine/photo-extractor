#!/usr/bin/env python3
"""Extract the core photograph from a phone scan of a physical photo.

Pipeline:
  1. Find the largest quadrilateral in the image (the photo sitting on a surface).
  2. Perspective-warp it into a clean rectangle.
  3. Trim any white border the matte left behind.
  4. Lightly enhance color/contrast so the result looks like the original print.
  5. Auto-orient to upright using face detection.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Register HEIC support on import so PIL can open iPhone scans.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - HEIC support is optional
    pass


SUPPORTED = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff")

# Work on a downscaled copy for detection — fast, and edge detection behaves
# better at a fixed resolution. Corners are scaled back to full res afterward.
WORK_EDGE = 1024


def load_image(path: Path) -> np.ndarray | None:
    """Open an image as a BGR uint8 array. Falls back to PIL for HEIC etc."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # HEIC and a few others: route through PIL → numpy → OpenCV BGR.
    try:
        from PIL import Image

        with Image.open(path) as pil:
            rgb = pil.convert("RGB")
            arr = np.array(rgb)[:, :, ::-1].copy()  # RGB → BGR
            return arr
    except Exception as exc:
        print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
        return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Uses the classic sum/difference trick: TL has the smallest x+y, BR the
    largest, TR the smallest y-x (i.e. largest x-y), BL the largest y-x.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)             # x + y
    diff = np.diff(pts, axis=1).ravel()  # y - x
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _closed_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edges with lowered thresholds + morphological closing.

    Real phone scans rarely give a clean closed boundary from plain Canny —
    lighting gradients, matte texture, and the photo's own detail fragment the
    edge along the boundary. A large closing kernel fuses those fragments into
    a single closed contour that findContours can grab.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(blurred)
    edges = cv2.Canny(blurred, int(max(0, 0.5 * med)), int(min(255, 1.0 * med)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Adaptive-threshold mask that isolates the photo as a solid region.

    This is the key tool for low-contrast scans (e.g. a white-bordered photo on
    a white surface): Canny sees no border-vs-background edge, but the photo's
    content still reads as a distinct region against its local background.
    Adaptive (not global) threshold handles uneven lighting and works whether
    the surface is lighter or darker than the photo.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5
    )
    # Collect the thresholded bits into solid blobs that match photo scale.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _fit_quad(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to an ordered 4-corner quad.

    Try Douglas-Peucker at increasing tolerances until it collapses to 4
    convex points. Many real photo contours are slightly non-convex or have
    extra vertices from texture; if no clean 4-point fit emerges, fall back to
    the minimum-area rotated rectangle, which assumes the subject is rectangular
    (a safe assumption for a photograph).
    """
    peri = cv2.arcLength(contour, True)
    for eps_frac in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx.reshape(4, 2).astype("float32"))
    rect = cv2.minAreaRect(contour)
    return _order_corners(cv2.boxPoints(rect).astype("float32"))


def _quad_area(q: np.ndarray) -> float:
    return float(cv2.contourArea(q.reshape(-1, 1, 2)))


def _quads_overlap(a: np.ndarray, b: np.ndarray, iou_thresh: float = 0.2) -> bool:
    """True if two quads share more than iou_thresh of the smaller one's area.

    Used to de-duplicate the same photo caught by both the edge and threshold
    methods, without merging genuinely separate photos in a multi-photo scan.
    """
    a_poly = a.reshape(-1, 1, 2).astype(np.int32)
    b_poly = b.reshape(-1, 1, 2).astype(np.int32)
    inter, _ = cv2.intersectConvexConvex(a_poly.reshape(4, 2), b_poly.reshape(4, 2))
    if inter <= 0:
        return False
    return inter / min(_quad_area(a), _quad_area(b)) > iou_thresh


def find_photo_quads(img: np.ndarray, max_photos: int = 8) -> list[np.ndarray]:
    """Find all photographs in the image; return ordered corner quads (orig res).

    Combines closed-edge and adaptive-threshold contour sources so it works on
    dark surfaces (strong border edge) and light/white surfaces (border blends
    in, only content is visible). Each large contour is fit to a quad; duplicate
    detections of the same photo are collapsed.
    """
    h, w = img.shape[:2]
    scale = WORK_EDGE / float(max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    min_area = 0.04 * (small.shape[0] * small.shape[1])

    candidates: list[np.ndarray] = []
    for src in (_closed_edges(gray), _threshold_mask(gray)):
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > min_area:
                q = _fit_quad(c)
                if q is not None and _quad_area(q) > min_area:
                    candidates.append(q)

    # Largest first; keep non-overlapping ones.
    candidates.sort(key=_quad_area, reverse=True)
    quads: list[np.ndarray] = []
    for q in candidates:
        if all(not _quads_overlap(q, kept) for kept in quads):
            quads.append(q)
        if len(quads) >= max_photos:
            break
    return [_order_corners(q) / scale for q in quads]


def find_photo_corners(img: np.ndarray) -> np.ndarray | None:
    """Single-photo convenience wrapper around find_photo_quads."""
    quads = find_photo_quads(img)
    return quads[0] if quads else None


def _destination_size(corners: np.ndarray) -> tuple[int, int]:
    """Compute the warped image's (width, height) from corner distances."""
    (tl, tr, br, bl) = corners
    width_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    height_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = np.hypot(br[0] - tr[0], br[1] - tr[1])
    w = int(round(max(width_top, width_bot)))
    h = int(round(max(height_left, height_right)))
    return w, h


def _inset_corners(corners: np.ndarray, px: float = 3.0) -> np.ndarray:
    """Pull each corner toward the quad's centroid by `px` pixels.

    Warping the *exact* detected boundary samples the very edge, where bilinear
    interpolation blends the subject with its background (floor). Insetting a
    hair lands on clean subject pixels and avoids a softened halo.
    """
    centroid = corners.mean(axis=0)
    vec = corners - centroid
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return corners - px * (vec / norm)


def crop_white_border(img: np.ndarray, threshold: int = 240, edge_frac: float = 0.95) -> np.ndarray:
    """Trim a near-white matte margin from each edge independently.

    A row/column is considered "border" only if it is almost entirely white
    (>= edge_frac). This avoids biting into a photo that happens to have light
    content (sky, snow) along one edge — those rows have real detail mixed in.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_white = gray >= threshold
    row_white = is_white.mean(axis=1)  # per-row fraction of white pixels
    col_white = is_white.mean(axis=0)

    content_rows = np.where(row_white < edge_frac)[0]
    content_cols = np.where(col_white < edge_frac)[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return img  # whole frame is white (or all content) — leave it.

    y0, y1 = int(content_rows[0]), int(content_rows[-1]) + 1
    x0, x1 = int(content_cols[0]), int(content_cols[-1]) + 1
    pad = max(2, int(0.003 * max(img.shape[:2])))
    return img[max(0, y0 - pad) : min(img.shape[0], y1 + pad), max(0, x0 - pad) : min(img.shape[1], x1 + pad)]


def enhance(img: np.ndarray) -> np.ndarray:
    """Batch-consistent tone normalization for a cohesive montage.

    Order matters and is deterministic (no per-image tuning) so every photo
    in a batch gets the same treatment:
      1. Gray-world white balance — neutralize the yellow/aging casts and scan
         color shifts that make old prints look patchy next to each other.
      2. Luminance percentile stretch — restore contrast lost to fading.
      3. CLAHE on lightness — local contrast.
      4. Gentle saturation lift.
    """
    out = img.astype(np.float32)

    # 1. Gray-world white balance: force the average pixel to neutral gray.
    #    Subtract each channel's mean offset, then re-center on the image mean
    #    (keeps overall brightness instead of dragging it toward 128).
    mean_per_channel = out.reshape(-1, 3).mean(axis=0)
    gray = mean_per_channel.mean()
    out += (gray - mean_per_channel)  # broadcast (3,) over (H,W,3)

    # 2. Luminance percentile stretch (2nd–98th → 0–255) on the L channel.
    lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lo, hi = np.percentile(l, 2), np.percentile(l, 98)
    if hi - lo >= 8:  # skip if already high-contrast (avoid amplifying noise)
        l = np.clip((l.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    # 3. CLAHE for local contrast.
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # 4. Gentle saturation lift (1.1x).
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# --- Auto-orientation -------------------------------------------------------
# Photos scanned at an angle come out sideways or upside-down. We pick the
# upright rotation by running a face detector on all four 0/90/180/270 rotations
# and keeping the one with the most confident face. Faces are a strong signal
# for personal/family photos; if none is detected confidently we leave the
# image untouched rather than guess (a wrong rotation is worse than none).

_YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yunet.onnx"
_FACE_DETECTOR: cv2.FaceDetectorYN | None = None
_FACE_DETECTOR_FAILED = False

_ROT_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _get_face_detector() -> cv2.FaceDetectorYN | None:
    """Lazily load the YuNet face detector, downloading it once on first use."""
    global _FACE_DETECTOR, _FACE_DETECTOR_FAILED
    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR
    if _FACE_DETECTOR_FAILED:
        return None
    if not _MODEL_PATH.exists():
        try:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"  downloading face model → {_MODEL_PATH.name}", file=sys.stderr)
            urllib.request.urlretrieve(_YUNET_URL, _MODEL_PATH)
        except Exception as exc:
            print(f"  ! could not download face model; skipping auto-orientation ({exc})", file=sys.stderr)
            _FACE_DETECTOR_FAILED = True
            return None
    _FACE_DETECTOR = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (320, 320), score_threshold=0.5, nms_threshold=0.3)
    return _FACE_DETECTOR


def _best_face_score(detector: cv2.FaceDetectorYN, img: np.ndarray) -> float:
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return 0.0
    return float(max(f[-1] for f in faces))


def orient(img: np.ndarray, min_confidence: float = 0.6) -> tuple[np.ndarray, int]:
    """Rotate `img` so faces are upright. Returns (image, applied_degrees).

    Detection runs on a downscaled copy for speed; the chosen rotation is
    applied losslessly to the full-resolution image.
    """
    detector = _get_face_detector()
    if detector is None:
        return img, 0

    longest = max(img.shape[:2])
    if longest > 900:
        s = longest / 900
        small = cv2.resize(img, (int(img.shape[1] / s), int(img.shape[0] / s)))
    else:
        small = img

    scores = {}
    for deg, code in _ROT_CODES.items():
        rotated = small if code is None else cv2.rotate(small, code)
        scores[deg] = _best_face_score(detector, rotated)

    best_deg = max(scores, key=scores.get)
    if scores[best_deg] < min_confidence or best_deg == 0:
        return img, 0  # no confident face, or already upright — don't risk a guess.
    return cv2.rotate(img, _ROT_CODES[best_deg]), best_deg


# --- Deskew (level residual tilt) -------------------------------------------
# Perspective correction gets the photo axis-aligned to within a few degrees,
# but an imperfect quad or a tilted print leaves a small residual rotation.
# We estimate it from dominant near-axis edges (Hough lines) and rotate to
# level — but only when the evidence is strong, so we never add a guess-rotation
# on top of the 90/180/270 flips handled by orient().
def deskew(img: np.ndarray, min_deg: float = 0.4, max_deg: float = 6.0, min_lines: int = 12, agree_deg: float = 0.6) -> np.ndarray:
    """Return a small-rotation-corrected copy of `img`, or the original if the
    skew can't be determined confidently.

    Safety matters more than coverage: a photo full of interior lines (window
    lattice, picture frames, fences) produces spurious "dominant" angles that
    have nothing to do with the true horizontal. So we require the near-horizontal
    AND near-vertical edge populations to each be tightly clustered AND to agree
    on the same skew — only a genuinely tilted photo's edges all tilt together.
    Otherwise we leave the image alone.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = max(40, int(0.15 * max(img.shape[:2])))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return img

    h_dev: list[float] = []  # deviations of near-horizontal lines from 0°
    v_dev: list[float] = []  # deviations of near-vertical lines from 90°
    for x1, y1, x2, y2 in lines[:, 0, :]:
        length = float(np.hypot(x2 - x1, y2 - y1))
        a = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
        weight = max(1, int(length / 20))
        if abs(a) <= 12:
            h_dev.extend([a] * weight)
        elif abs(abs(a) - 90) <= 12:
            v_dev.extend([a - (90 if a > 0 else -90)] * weight)

    if len(h_dev) < min_lines or len(v_dev) < min_lines:
        return img

    h_med, v_med = float(np.median(h_dev)), float(np.median(v_dev))
    # Each group must be tightly clustered (lines actually agree)...
    if float(np.median(np.abs(np.array(h_dev) - h_med))) > 1.0:
        return img
    if float(np.median(np.abs(np.array(v_dev) - v_med))) > 1.0:
        return img
    # ...and the two groups must agree on the same skew.
    if abs(h_med - v_med) > agree_deg:
        return img

    skew = (h_med + v_med) / 2.0
    if abs(skew) < min_deg or abs(skew) > max_deg:
        return img
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, skew, 1.0)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# --- Date-stamp removal (conservative) --------------------------------------
# Printed date stamps ("11 96", "11 11 96") are distracting on a montage. They
# are almost always small, digit-shaped, and in the border — so we restrict the
# search to the outer band and require a cluster of digit-sized marks (a date
# has several digits) before inpainting. Better to leave a stamp than to eat
# part of a photo.
def remove_date_stamps(img: np.ndarray, min_digits: int = 3) -> np.ndarray:
    """Inpaint printed date stamps. Deliberately conservative.

    A date stamp is a short, tight horizontal row of small digits sitting in a
    *corner* of the photo. We detect digit-sized marks via local contrast
    (color-agnostic), group them into rows, and only inpaint a row that:
      - lives inside one of the four corner boxes, and
      - spans less than ~18% of the photo width (a date is short), and
      - has tight, even spacing between digits.
    Anything that doesn't meet all of that is left alone — we never risk
    erasing photo content (clothing patterns, signs, fences) that happens to
    form a horizontal line of small marks.
    """
    h, w = img.shape[:2]

    # Four corner search regions (each ~32% wide x 18% tall).
    cw, ch = int(0.32 * w), int(0.18 * h)
    corners = np.zeros((h, w), dtype=bool)
    for oy in (0, h - ch):
        for ox in (0, w - cw):
            corners[oy : oy + ch, ox : ox + cw] = True

    # Local-contrast text mask: small features standing out from a blurred bg.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    text = cv2.bitwise_or(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se))
    _, text = cv2.threshold(text, 28, 255, cv2.THRESH_BINARY)
    text[~corners] = 0

    # Digit-sized components.
    min_h, max_h = 0.006 * h, 0.045 * h
    n, _, stats, _ = cv2.connectedComponentsWithStats(text, 8)
    boxes = []  # (cx, cy, x, y, bw, bh)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bh > 0 and min_h <= bh <= max_h and 0.2 <= bw / bh <= 1.2:
            boxes.append((x + bw / 2, y + bh / 2, x, y, bw, bh))
    if len(boxes) < min_digits:
        return img

    # Group into horizontal rows (digit centers aligned in y).
    boxes.sort(key=lambda b: b[1])
    candidates: list[list[tuple]] = []
    used = [False] * len(boxes)
    for i, seed in enumerate(boxes):
        if used[i]:
            continue
        row = [seed]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if abs(boxes[j][1] - seed[1]) < 0.8 * seed[5]:
                row.append(boxes[j])
                used[j] = True
        if len(row) >= min_digits:
            candidates.append(sorted(row, key=lambda b: b[0]))

    mask = np.zeros((h, w), dtype=np.uint8)
    for row in candidates:
        xs = [b[0] for b in row]
        span = xs[-1] - xs[0]
        med_h = float(np.median([b[5] for b in row]))
        gaps = [row[k + 1][0] - row[k][0] for k in range(len(row) - 1)]
        # Date string: short, tight, evenly spaced.
        if span > 0.18 * w:
            continue
        if gaps and (max(gaps) > 3.5 * med_h or min(gaps) < 0.3 * med_h):
            continue
        for _, _, x, y, bw, bh in row:
            mask[y : y + bh, x : x + bw] = 255

    if mask.sum() == 0:
        return img
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def warp_quad(
    img: np.ndarray,
    corners: np.ndarray,
    do_crop_white: bool,
    do_enhance: bool,
    do_orient: bool,
    do_deskew: bool,
    do_remove_dates: bool,
) -> np.ndarray | None:
    """Perspective-flatten one quad into a clean, montage-ready photo.

    Order: warp → deskew → trim matte → remove date stamps → tone normalize → orient.
    """
    w, h = _destination_size(corners)
    if w < 64 or h < 64:
        return None
    src = _inset_corners(corners)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h))
    if do_deskew:
        warped = deskew(warped)
    if do_crop_white:
        warped = crop_white_border(warped)
    if do_remove_dates:
        warped = remove_date_stamps(warped)
    if do_enhance:
        warped = enhance(warped)
    if do_orient:
        warped, _ = orient(warped)
    return warped


def extract_all(
    img: np.ndarray,
    do_enhance: bool = True,
    do_crop_white: bool = True,
    do_orient: bool = True,
    do_deskew: bool = True,
    do_remove_dates: bool = True,
) -> list[np.ndarray]:
    """Full pipeline: detect every photo in the scan and return each flattened."""
    out: list[np.ndarray] = []
    for corners in find_photo_quads(img):
        warped = warp_quad(img, corners, do_crop_white, do_enhance, do_orient, do_deskew, do_remove_dates)
        if warped is not None:
            out.append(warped)
    return out


def extract(
    img: np.ndarray,
    do_enhance: bool = True,
    do_crop_white: bool = True,
    do_orient: bool = True,
    do_deskew: bool = True,
    do_remove_dates: bool = True,
) -> np.ndarray | None:
    """Single-photo convenience wrapper. Returns the first (largest) photo."""
    results = extract_all(
        img, do_enhance=do_enhance, do_crop_white=do_crop_white, do_orient=do_orient, do_deskew=do_deskew, do_remove_dates=do_remove_dates
    )
    return results[0] if results else None


def process_file(path: Path, out_dir: Path, enhance_photo: bool, crop_white: bool, orient_photo: bool, deskew_photo: bool, remove_dates: bool, quality: int) -> int:
    """Extract every photo from one scan; write one JPG per photo. Returns count."""
    img = load_image(path)
    if img is None:
        return 0
    results = extract_all(
        img,
        do_enhance=enhance_photo,
        do_crop_white=crop_white,
        do_orient=orient_photo,
        do_deskew=deskew_photo,
        do_remove_dates=remove_dates,
    )
    if not results:
        # Last-resort fallback: save the original so the user isn't left empty-handed.
        print(f"  ~ {path.name}: no clear photo boundary found, saving original", file=sys.stderr)
        results = [img]
    written = 0
    multi = len(results) > 1
    for i, res in enumerate(results):
        suffix = f"_{i + 1}" if multi else ""
        out_path = out_dir / f"{path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_path), res, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"  ✓ {path.name} → {out_path.name}  ({res.shape[1]}×{res.shape[0]})")
        written += 1
    return written


def gather_inputs(paths: list[Path]) -> list[Path]:
    """Expand directories and filter to supported image types."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED))
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"  ! skipping {p} (not a supported image or directory)", file=sys.stderr)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract a photographed photo from a phone scan.",
        epilog="Examples:\n"
        "  uv run photo_extractor.py scan.jpg -o out/\n"
        "  uv run photo_extractor.py scans/ -o out/ --no-enhance\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="image file(s) or folder(s)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output directory")
    ap.add_argument("--no-enhance", action="store_true", help="skip tone normalization")
    ap.add_argument("--no-crop-white", action="store_true", help="skip white-border trimming")
    ap.add_argument("--no-orient", action="store_true", help="skip auto-orientation (face-based)")
    ap.add_argument("--no-deskew", action="store_true", help="skip residual-tilt leveling")
    ap.add_argument("--remove-dates", action="store_true", help="remove printed date stamps (experimental; review outputs)")
    ap.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100, default 95)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    files = gather_inputs(args.inputs)
    if not files:
        print("No supported images found.", file=sys.stderr)
        return 1

    photos = 0
    for f in files:
        photos += process_file(
            f,
            args.output,
            not args.no_enhance,
            not args.no_crop_white,
            not args.no_orient,
            not args.no_deskew,
            args.remove_dates,
            args.quality,
        )

    print(f"\nDone: {photos} photo(s) from {len(files)} scan(s) → {args.output}")
    return 0 if photos else 2


if __name__ == "__main__":
    raise SystemExit(main())
