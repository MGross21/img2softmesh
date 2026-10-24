"""Crop / rectify the photo and prepare it as a texture."""

import cv2
import numpy as np


def crop_to_mask(bgr, mask, pad_frac=0.03, max_side=2048):
    ys, xs = np.nonzero(mask)
    pad = int(max(mask.shape) * pad_frac) + 2
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    bgr, mask = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    # zero border so contours / distance transforms never touch the image edge
    bgr = cv2.copyMakeBorder(bgr, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    s = max_side / max(mask.shape)
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = (cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
    return bgr, mask


def _order_quad(q):
    s, d = q.sum(1), np.diff(q, axis=1)[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


def rectify_if_quad(bgr, mask, min_fill=0.95):
    """If the silhouette is a (perspective) quadrilateral, warp it to a flat rectangle.

    Returns (bgr, mask) or None. Fixes keystone on photos of towels/blankets laid flat.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) != 4 or cv2.contourArea(c) / cv2.contourArea(approx) < min_fill:
        return None
    tl, tr, br, bl = q = _order_quad(approx[:, 0, :].astype(np.float32))
    w = round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    h = round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    pad = max(4, int(max(w, h) * 0.02))
    dst = np.array([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    size = (w + 2 * pad, h + 2 * pad)
    out_bgr = cv2.warpPerspective(bgr, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out_mask = np.zeros((size[1], size[0]), np.uint8)
    out_mask[pad:pad + h, pad:pad + w] = 255
    return out_bgr, out_mask


def bleed(bgr, mask, levels=8):
    """Push-pull fill of the background with nearby foreground color.

    Stops background color leaking onto the mesh edges through texture filtering/mipmaps.
    """
    m = (mask > 0).astype(np.float32)
    acc = bgr.astype(np.float32) * m[..., None]
    pyr = [(acc, m)]
    for _ in range(levels):
        a, w = pyr[-1]
        if min(w.shape) < 4:
            break
        pyr.append((cv2.pyrDown(a), cv2.pyrDown(w)))
    filled = None
    for a, w in reversed(pyr):
        col = a / np.maximum(w, 1e-6)[..., None]
        if filled is None:
            filled = col
        else:
            up = cv2.resize(filled, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
            t = np.clip(w * 4, 0, 1)[..., None]
            filled = col * t + up * (1 - t)
    out = np.where(m[..., None] > 0, bgr.astype(np.float32), filled)
    return np.clip(out, 0, 255).astype(np.uint8)
