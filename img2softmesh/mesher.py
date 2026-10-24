"""Silhouette -> triangle mesh (single sheet or welded two-layer shell)."""

import cv2
import numpy as np
import triangle


def _smooth_closed(c, k):
    if k < 1:
        return c
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.vstack([c[-k:], c, c[:k]])
    return np.stack([np.convolve(pad[:, i], kern, "valid") for i in range(2)], 1)


def _resample_closed(c, step):
    closed = np.vstack([c, c[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(12, round(s[-1] / step))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], 1)


def sample(img, pts):
    h, w = img.shape[:2]
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    return img[y, x]


def boundary_edges(tris):
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


def boundary_mask(n, tris):
    b = np.zeros(n, bool)
    b[boundary_edges(tris).ravel()] = True
    return b


def silhouette_mesh(mask, n_target):
    """Quality-triangulate the mask's outer silhouette with ~n_target vertices.

    Returns (pts [N,2] in pixel coords, tris [M,3]) wound counter-clockwise in a y-up frame.
    """
    h = np.sqrt(np.count_nonzero(mask) / (n_target * np.sqrt(3) / 2))  # edge length of the target lattice
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    boundary = _resample_closed(_smooth_closed(c, max(1, int(h / 4))), h)
    seg = np.column_stack([np.arange(len(boundary)), np.roll(np.arange(len(boundary)), -1)])
    # p: respect the outline, q30: min angle 30deg, a: max triangle area (refined triangles average ~1/1.7
    # of the cap, so the cap is 1.7x an equilateral triangle of side h),
    # Y: no extra points on the outline (it is already sampled at spacing h)
    t = triangle.triangulate({"vertices": boundary, "segments": seg}, f"pq30a{1.7 * h * h * np.sqrt(3) / 4:.6f}YQ")
    pts, tris = t["vertices"], t["triangles"]

    # consistent CCW winding with y up = clockwise in image coords (y down)
    d1, d2 = pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]
    ccw_img = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] > 0
    tris[ccw_img] = tris[ccw_img, ::-1]
    return pts, tris


def split_chords(pts, tris):
    """Split interior edges whose endpoints both lie on the boundary.

    Needed before welding two layers: such an edge would otherwise be shared by
    four triangles (non-manifold) once front and back share boundary vertices.
    """
    while True:
        bnd = boundary_mask(len(pts), tris)
        occ = {}
        for ti, t in enumerate(tris):
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                if bnd[a] and bnd[b]:
                    occ.setdefault((min(a, b), max(a, b)), []).append((ti, k))
        chords = [(e, o) for e, o in occ.items() if len(o) == 2]
        if not chords:
            return pts, tris
        new_pts, repl, used = list(pts), {}, set()
        for (a, b), o in chords:
            if any(ti in used for ti, _ in o):
                continue  # handled next pass
            m = len(new_pts)
            new_pts.append((pts[a] + pts[b]) / 2)
            for ti, k in o:
                used.add(ti)
                t = tris[ti]
                x, y, z = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                repl[ti] = [(x, m, z), (m, y, z)]
        out = []
        for ti, t in enumerate(tris):
            out.extend(repl.get(ti, [tuple(t)]))
        pts, tris = np.array(new_pts), np.array(out)


def two_layer(pts, tris):
    """Duplicate interior vertices into a back layer welded at the silhouette.

    Returns (pts, tris, is_back). Back faces are reversed so normals point outward.
    """
    n = len(pts)
    interior = ~boundary_mask(n, tris)
    back = np.arange(n)
    back[interior] = n + np.arange(interior.sum())
    pts2 = np.vstack([pts, pts[interior]])
    tris2 = np.vstack([tris, back[tris][:, ::-1]])
    is_back = np.r_[np.zeros(n, bool), np.ones(interior.sum(), bool)]
    return pts2, tris2, is_back
