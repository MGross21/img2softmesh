"""Foreground segmentation: image alpha > rembg (optional extra) > GrabCut."""

from importlib.util import find_spec

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _has_alpha(img):
    return img.shape[2] == 4 and (img[:, :, 3] < 250).mean() > 0.01


def _rembg(bgr):
    from rembg import remove
    out = np.asarray(remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), only_mask=True))
    if out.ndim == 3:
        out = out[:, :, -1]
    return out > 127


def _grabcut(bgr):
    """GrabCut seeded from 'differs from the border color' -- good for product shots on plain backgrounds."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, min(h, w) // 100)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    diff = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    gc = np.where(diff > 18, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[diff > 45] = cv2.GC_FGD
    gc[:b], gc[-b:], gc[:, :b], gc[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def clean_mask(mask):
    """Largest component, holes filled, small speckle removed. Returns uint8 0/255."""
    m = mask.astype(np.uint8) * 255
    k = max(3, int(min(m.shape) * 0.006) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise ValueError("segmentation found no foreground object")
    m = np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled


def segment(img, method="auto"):
    """Return (bgr, mask) with mask uint8 0/255.

    method: auto | alpha | rembg | grabcut
    """
    bgr = np.ascontiguousarray(img[:, :, :3])
    if method == "auto":
        method = "alpha" if _has_alpha(img) else "rembg" if find_spec("rembg") else "grabcut"
    match method:
        case "alpha":
            if not _has_alpha(img):
                raise ValueError("image has no transparency to use as a mask")
            mask = img[:, :, 3] > 127
        case "rembg":
            if not find_spec("rembg"):
                raise ImportError("rembg not installed: pip install 'img2softmesh[rembg]'")
            mask = _rembg(bgr)
        case "grabcut":
            mask = _grabcut(bgr)
        case _:
            raise ValueError(f"unknown segmentation method: {method}")
    return bgr, clean_mask(mask)
