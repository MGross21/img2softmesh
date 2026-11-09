"""Generated 3D asset (GLB) -> shape + back-side texture.

Generated meshes are rarely manifold, so they are never simulated or drawn directly. Instead:
  physics: the clean silhouette cloth mesh, lifted onto the asset's top/bottom surfaces (MuJoCo flex)
  render:  that same mesh, photo-aligned uv on both layers: the front layer shows the photo, the back
           layer the asset seen from below (one skin per layer)
"""

import cv2
import numpy as np


def load_glb(path):
    """Return (verts, faces, uv, PIL texture image) of a single-texture GLB."""
    import trimesh
    mesh = trimesh.load(str(path), force="mesh")  # flattens the scene graph, applying node transforms
    mat = mesh.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces), np.asarray(mesh.visual.uv, float), img


def rasterize(p2, z, faces, shape):
    """Top-view painter's-algorithm rasterization of a mesh.

    p2: vertex pixel coords, z: vertex heights. Returns (top height, bottom height, bottom face id) maps;
    pixels the mesh misses take the nearest covered value.
    """
    from scipy.ndimage import distance_transform_edt
    tri_z = z[faces].mean(1).astype(np.float32)
    polys = np.rint(p2[faces]).astype(np.int32)
    out = []
    for order in (np.argsort(tri_z), np.argsort(-tri_z)):  # highest painted last = top; lowest last = bottom
        hm = np.full(shape, np.nan, np.float32)
        ids = np.full(shape, -1, np.int32)
        for i in order:
            cv2.fillConvexPoly(hm, polys[i], float(tri_z[i]))
            cv2.fillConvexPoly(ids, polys[i], int(i))
        idx = tuple(distance_transform_edt(np.isnan(hm), return_distances=False, return_indices=True))
        out.append((hm[idx], ids[idx]))
    (top, _), (bottom, bottom_ids) = out
    return top, bottom, bottom_ids


def back_view(p2, uv, faces, tex, ids, photo, mask):
    """Color the mask region with the asset's underside (its back), color-matched to the photo."""
    H, W = tex.shape[:2]
    ys, xs = np.nonzero(ids >= 0)
    f = faces[ids[ys, xs]]
    a, b, c = p2[f[:, 0]], p2[f[:, 1]], p2[f[:, 2]]
    q = np.column_stack([xs, ys]).astype(float)
    v0, v1, v2 = b - a, c - a, q - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den[np.abs(den) < 1e-12] = 1e-12
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bary = np.clip(np.column_stack([1 - l1 - l2, l1, l2]), 0, 1)
    bary /= bary.sum(1, keepdims=True)
    t = (bary[:, :, None] * uv[f]).sum(1)  # trimesh uv: v up
    mx, my = np.zeros(ids.shape, np.float32), np.zeros(ids.shape, np.float32)
    mx[ys, xs], my[ys, xs] = t[:, 0] * W, (1 - t[:, 1]) * H
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img[ids < 0] = 0
    return _match_color(img, mask > 0, photo, mask)


def _match_color(img, used, photo, mask):
    """Shift LAB mean/std of img[used] onto the photo's foreground (generated albedo runs dark)."""
    t = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    p = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)[mask > 0]
    src = t[used]
    t = (t - src.mean(0)) / (src.std(0) + 1e-6) * p.std(0) + p.mean(0)
    return cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
