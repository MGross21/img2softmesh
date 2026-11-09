"""Foreground segmentation: image alpha > U^2-Net-p (optional extra) > GrabCut."""

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


U2NETP = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"


def _u2netp(bgr):
    """Salient-object mask from U^2-Net-p (4.7 MB ONNX, ~0.3 s on CPU), pre/post-processed as rembg does."""
    import onnxruntime as ort

    from .trellis import download, home
    sess = ort.InferenceSession(download(U2NETP, home() / "u2netp.onnx"), providers=ort.get_available_providers())
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_AREA)
    x = (x / max(x.max(), 1e-6) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    p = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    p = (p - p.min()) / (p.max() - p.min())
    return cv2.resize(p, bgr.shape[1::-1], interpolation=cv2.INTER_LANCZOS4) > 0.5


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

    method: auto | alpha | u2net | grabcut
    """
    bgr = np.ascontiguousarray(img[:, :, :3])
    if method == "auto":
        method = "alpha" if _has_alpha(img) else "u2net" if find_spec("onnxruntime") else "grabcut"
    match method:
        case "alpha":
            if not _has_alpha(img):
                raise ValueError("image has no transparency to use as a mask")
            mask = img[:, :, 3] > 127
        case "u2net":
            if not find_spec("onnxruntime"):
                raise ImportError("onnxruntime not installed: pip install 'img2softmesh[u2net]'")
            mask = _u2netp(bgr)
        case "grabcut":
            mask = _grabcut(bgr)
        case _:
            raise ValueError(f"unknown segmentation method: {method}")
    return bgr, clean_mask(mask)
