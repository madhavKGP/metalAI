"""Shape features that separate a scratch (mechanical) from a crack (material failure). See decisions.md D-030.

Idea in one line: find the thin line-like marks in the image, thin them to a 1-pixel
skeleton, then measure HOW the lines look (straight? branched? even width? all
pointing the same way?) instead of what the pixels look like.

Pipeline per image
  1. Flatten uneven lighting (subtract a heavy blur), boost contrast (CLAHE).
  2. Ridge filter (Sato) enhances thin lines. Scratches in NEU are mostly bright and
     cracks mostly dark, so we run it for BOTH polarities and keep the stronger one.
     That way the features describe shape, not brightness.
  3. Hysteresis threshold -> binary mask of line pixels; drop tiny specks.
  4. Skeletonize (1-pixel-wide centre lines) and measure.

Features (all scale-free, so image size does not matter)
  straightness      length-weighted mean of (end-to-end span / path length) per line. 1 = ruler-straight.
  branch_density    junction pixels (3+ neighbours) per 100 skeleton pixels.
  endpoint_density  line ends per 100 skeleton pixels (many short pieces -> high).
  width_cv          coefficient of variation of line width along the skeleton (tapering -> high).
  orient_coherence  0..1, 1 = all lines point the same way (parallel scratches).
  longest_frac      longest single line / image diagonal (one long scratch -> high).
  fragments_per_100 separate line pieces per 100 skeleton pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

FEATURES = ["straightness", "branch_density", "endpoint_density", "width_cv",
            "orient_coherence", "longest_frac", "fragments_per_100"]

MIN_OBJ = 30      # px, drop specks in the mask
MIN_SKEL = 15     # px, drop skeleton pieces shorter than this
_K8 = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])


def preprocess(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), 15)
    flat = cv2.normalize(f - bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat).astype(np.float32) / 255.0


def line_mask(img: np.ndarray) -> tuple[np.ndarray, str]:
    p = preprocess(img)
    best = None
    for dark in (True, False):
        r = sato(p, sigmas=(1, 2, 3), black_ridges=dark)
        score = r[r >= np.percentile(r, 99)].mean()          # strength of the strongest ridges
        if best is None or score > best[0]:
            best = (score, r, "dark" if dark else "bright")
    _, r, pol = best
    lo, hi = np.percentile(r, 93), np.percentile(r, 98.5)
    m = apply_hysteresis_threshold(r, lo, hi)
    m = remove_small_objects(m, max_size=MIN_OBJ - 1)
    return m, pol


def skeleton_features(mask: np.ndarray) -> dict:
    H, W = mask.shape
    sk = skeletonize(mask)
    lab = label(sk, connectivity=2)
    for rp in regionprops(lab):
        if rp.area < MIN_SKEL:
            sk[lab == rp.label] = False
    lab = label(sk, connectivity=2)
    n_sk = int(sk.sum())
    if n_sk == 0:
        return {f: 0.0 for f in FEATURES}

    nb = ndi.convolve(sk.astype(int), _K8, mode="constant")
    branch = int(((nb >= 13) & sk).sum())          # centre(10) + 3 or more neighbours
    ends = int((nb == 11).sum())                   # centre + exactly 1 neighbour

    dist = ndi.distance_transform_edt(mask)
    w = 2 * dist[sk]
    width_cv = float(w.std() / (w.mean() + 1e-6))

    spans, lengths, angles = [], [], []
    for rp in regionprops(lab):
        L = rp.area
        span = max(rp.feret_diameter_max, 1.0)
        spans.append(min(span / L, 1.0))
        lengths.append(L)
        angles.append(rp.orientation)
    lengths = np.array(lengths, float)
    wts = lengths / lengths.sum()
    straight = float(np.sum(np.array(spans) * wts))
    ang = np.array(angles)
    coh = float(np.abs(np.sum(wts * np.exp(2j * ang))))   # resultant length of doubled angles

    return {
        "straightness": straight,
        "branch_density": 100 * branch / n_sk,
        "endpoint_density": 100 * ends / n_sk,
        "width_cv": width_cv,
        "orient_coherence": coh,
        "longest_frac": float(lengths.max() / np.hypot(H, W)),
        "fragments_per_100": float(100 * lab.max() / n_sk),
    }


def extract(img: np.ndarray) -> tuple[dict, np.ndarray, str]:
    m, pol = line_mask(img)
    return skeleton_features(m), m, pol
