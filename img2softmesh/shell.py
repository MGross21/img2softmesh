"""Render skins over the cloth, and generated 3D asset (GLB) -> shape + back-side texture.

Generated meshes are rarely manifold, so they are never simulated or skinned directly. Instead:
  physics: the clean silhouette cloth mesh, lifted onto the asset's top/bottom surfaces (MuJoCo flex)
  render:  that same mesh as a MuJoCo <skin> with an atlas: photo on the front, the asset seen from
           below on the back (seam vertices are duplicated so each side keeps its own UVs)
"""

import struct

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


def layer_skin(pts, tris, is_back, verts, shape):
    """Skin over the two-layer cloth with a side-by-side atlas (front | back).

    Returns (skin verts, uv, faces, bone per skin vertex). Seam vertices appear twice, once per side.
    """
    h, w = shape
    front = np.flatnonzero(~is_back)
    back_tris = is_back[tris].any(1)
    seam = np.unique(tris[back_tris][~is_back[tris[back_tris]]])
    back = np.r_[np.flatnonzero(is_back), seam]
    bone = np.r_[front, back]
    uv = np.r_[np.column_stack([pts[front, 0] / (2 * w), pts[front, 1] / h]),
               np.column_stack([0.5 + pts[back, 0] / (2 * w), pts[back, 1] / h])]
    fmap = np.full(len(pts), -1)
    fmap[front] = np.arange(len(front))
    bmap = np.full(len(pts), -1)
    bmap[back] = len(front) + np.arange(len(back))
    faces = np.where(back_tris[:, None], bmap[tris], fmap[tris])
    return verts[bone], uv, faces, bone


def slab_skin(pts, tris, verts, shape):
    """Skin over a single-layer cloth: front, back copy and rim. Skin inflate pushes the sides apart
    along their normals, so the cloth renders as a slab as thick as its contact shell.

    Both sides use the photo uv (the back reads mirrored, like the reverse of a woven cloth).
    """
    n, (h, w) = len(pts), shape
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    a, b = e[~np.isin(e[:, 0] * n + e[:, 1], e[:, 1] * n + e[:, 0])].T  # outline edges, CCW
    faces = np.vstack([tris, tris[:, ::-1] + n, np.column_stack([a, a + n, b + n]), np.column_stack([a, b + n, b])])
    uv = np.tile(pts / [w, h], (2, 1))
    return np.vstack([verts, verts]), uv, faces, np.tile(np.arange(n), 2)


def write_skn(path, verts, uv, faces, bone, bind_pos, body_prefix="cloth_"):
    """MuJoCo binary skin; skin vertex i follows sim vertex body bone[i] rigidly (uv: v down)."""
    order = np.argsort(bone, kind="stable")
    starts = np.searchsorted(bone[order], np.arange(len(bind_pos) + 1))
    used = [b for b in range(len(bind_pos)) if starts[b + 1] > starts[b]]  # MuJoCo rejects empty bones
    out = [struct.pack("4i", len(verts), len(uv), len(faces), len(used)),
           verts.astype("<f4").tobytes(), uv.astype("<f4").tobytes(), faces.astype("<i4").tobytes()]
    for b in used:
        ids = order[starts[b]:starts[b + 1]]
        out.append(f"{body_prefix}{b}".encode().ljust(40, b"\0"))
        out.append(struct.pack("<3f4f", *bind_pos[b], 1, 0, 0, 0))
        out.append(struct.pack("<i", len(ids)))
        out.append(ids.astype("<i4").tobytes())
        out.append(np.ones(len(ids), "<f4").tobytes())
    with open(path, "wb") as fh:
        fh.write(b"".join(out))
