"""image -> segmented texture -> cloth mesh (silhouette or TRELLIS.2) -> OBJ + MuJoCo scene."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from .export import write_obj, write_scene
from .mesher import sample, silhouette_mesh, split_chords, two_layer
from .presets import PRESETS
from .segment import load_image, segment
from .texture import bleed, crop_to_mask, rectify_if_quad

TABLE_H = 0.5


def build(image, kind="auto", size=None, verts=None, radius=None, segment_method="auto", geometry="silhouette",
          out_dir=None, max_texture=2048, selfcollide="none", trellis_res=512):
    """Run the full pipeline. Returns a dict describing the outputs.

    geometry: "silhouette" -- flat/two-layer mesh from the outline, texture = photo pixels (fast, CPU)
              "trellis"    -- TRELLIS.2 generates full 3D + 360-degree texture (local GPU, ~4 GB weights)
    """
    image = Path(image)
    out = Path(out_dir) if out_dir else Path("out") / image.stem
    out.mkdir(parents=True, exist_ok=True)

    bgr, mask = segment(load_image(image), segment_method)
    bgr, mask = crop_to_mask(bgr, mask, max_side=max_texture)
    cv2.imwrite(str(out / "mask.png"), mask)

    rect = rectify_if_quad(bgr, mask)
    if kind == "auto":
        kind = "towel" if rect is not None else "garment"
    if kind not in PRESETS:
        raise ValueError(f"unknown kind {kind!r}; choose from: auto, {', '.join(PRESETS)}")
    p = PRESETS[kind] if radius is None else replace(PRESETS[kind], radius=radius)
    rectified = p.layers == 1 and rect is not None
    if rectified:
        bgr, mask = rect

    if geometry == "silhouette":
        m = _silhouette(out, bgr, mask, p, size, verts)
    elif geometry == "trellis":
        m = _trellis(out, bgr, mask, p, size, verts, trellis_res)
    else:
        raise ValueError(f"unknown geometry backend {geometry!r}")

    v3 = m["verts"]
    mass = m["area"] * p.density * p.layers

    grasp = _grasp_patches(v3, m["top"], m["edge"], p.grasp)
    ext = np.abs(v3[:, :2]).max(0)
    write_scene(out / "scene.xml", name=image.stem, kind=kind, preset=p, mass=mass,
                table=(ext[0] * 1.3 + 0.1, ext[1] * 1.3 + 0.1, TABLE_H / 2),
                grippers=[(v3[g[0]], g) for g in grasp], selfcollide=selfcollide)

    info = dict(image=str(image), kind=kind, geometry=geometry, preset=asdict(p), rectified=rectified,
                vertices=len(v3), triangles=len(m["tris"]), size_m=(2 * ext).tolist(),
                area_m2=float(m["area"]), mass_kg=float(mass), radius_m=p.radius, grasp_vertices=grasp,
                out_dir=str(out), scene=str(out / "scene.xml"))
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    return info


def _grasp_patches(v3, top, edge, points):
    """Vertex patch per grip point on the upper surface ("left"/"right" top corner, "collar" top center).

    A patch, not one vertex: pulling a single vertex just stretches it into a spike.
    """
    t = np.flatnonzero(top)
    x, y = v3[t, 0], v3[t, 1]
    score = {"left": y - x, "right": y + x, "collar": np.where(np.abs(x) < 0.1 * np.ptp(x), y, -np.inf)}
    patches = []
    for name in points:
        near = np.linalg.norm(v3 - v3[t[np.argmax(score[name])]], axis=1)
        patches.append([i for i in np.argsort(near)[:7].tolist() if near[i] < 1.6 * edge])
    return patches


def _outline(mask, p, size, verts):
    """Clean cloth topology from the silhouette, in the metric frame.

    Frame: x right, y up (image top = +y), z up off the table; pixel (x, y) -> ((x - cx) * scale, -(y - cy) * scale).
    """
    n = verts or p.verts
    pts, tris = silhouette_mesh(mask, n)
    if p.layers == 2:
        pts, tris = split_chords(pts, tris)
        pts, tris, is_back = two_layer(pts, tris)
    else:
        is_back = np.zeros(len(pts), bool)
    ys, xs = np.nonzero(mask)
    scale = (size or p.size) / max(np.ptp(xs) + 1, np.ptp(ys) + 1)  # m per px
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    xy = np.column_stack([(pts[:, 0] - cx) * scale, -(pts[:, 1] - cy) * scale])
    return dict(pts=pts, tris=tris, is_back=is_back, xy=xy, scale=scale, area=len(xs) * scale ** 2,
                edge=np.sqrt(len(xs) / (n * np.sqrt(3) / 2)) * scale)


def _silhouette(out, bgr, mask, p, size, verts):
    from .shell import layer_skin, slab_skin, write_skn
    o = _outline(mask, p, size, verts)
    pts, is_back, radius = o["pts"], o["is_back"], p.radius

    # small initial puff separates the two layers (fades to 0 at the seam)
    dist_m = sample(cv2.distanceTransform(mask, cv2.DIST_L2, 5), pts) * o["scale"]
    puff = np.clip(dist_m / (3 * o["edge"]), 0, 1) * max(3 * radius, 0.006)
    z_layer = np.where(is_back, -puff, puff) if p.layers == 2 else np.zeros(len(pts))

    z0 = TABLE_H + 0.02 + (puff.max() if p.layers == 2 else radius)
    v3 = np.column_stack([o["xy"], z0 + z_layer])
    # render: skin with visible thickness. Two layers: front | back atlas, the back shows the photo mirrored
    tex = bleed(bgr, mask)
    if p.layers == 2:
        tex = np.hstack([tex, tex])
        skin = layer_skin(pts, o["tris"], is_back, v3, mask.shape)
    else:
        skin = slab_skin(pts, o["tris"], v3, mask.shape)
    cv2.imwrite(str(out / "texture.png"), tex)
    write_skn(out / "cloth.skn", *skin, v3)
    h, w = mask.shape
    uv = np.stack([pts[:, 0] / (w * p.layers), 1 - pts[:, 1] / h], 1)  # obj (v up), front of the texture
    write_obj(out / "cloth.obj", v3, o["tris"], uv)
    return dict(verts=v3, tris=o["tris"], edge=o["edge"], area=o["area"], top=~is_back)


def _trellis(out, bgr, mask, p, size, verts, res):
    from . import trellis
    from .shell import back_view, layer_skin, load_glb, rasterize, write_skn

    cv2.imwrite(str(out / "cutout.png"), np.dstack([bgr, mask]))
    glb = trellis.generate(out / "cutout.png", out / "trellis.glb", res=res)
    rv, rf, ruv, img = load_glb(glb)

    # glTF is y-up with the subject facing +z (lying face-up: glTF z becomes world up).
    # Register the generated mesh onto the photo: its x/y bbox <-> the mask bbox.
    o = _outline(mask, p, size, verts)
    my, mx = np.nonzero(mask)
    lo, hi = rv.min(0), rv.max(0)
    kx = (mx.max() - mx.min()) / (hi[0] - lo[0])  # px per glTF unit
    p2 = np.column_stack([mx.min() + (rv[:, 0] - lo[0]) * kx,
                          my.min() + (hi[1] - rv[:, 1]) / (hi[1] - lo[1]) * (my.max() - my.min())])
    zr = (rv[:, 2] - lo[2]) * kx * o["scale"]  # meters above the lowest point
    top_h, bot_h, bot_ids = rasterize(p2, zr, rf, mask.shape)

    # physics: clean outline topology, lifted onto the generated top/bottom surfaces
    pts, tris, is_back = o["pts"], o["tris"], o["is_back"]
    zt, zb = sample(top_h, pts), sample(bot_h, pts)
    zb = np.minimum(zb, zt - max(2.5 * p.radius, 0.005))
    z = np.where(is_back, zb, zt)
    if p.layers == 2:  # welded seam (front vertices also used by back faces) sits between the surfaces
        seam = np.zeros(len(pts), bool)
        seam[tris[is_back[tris].any(1)].ravel()] = True
        seam &= ~is_back
        z[seam] = (zt[seam] + zb[seam]) / 2
    base = TABLE_H + 0.02
    sv = np.column_stack([o["xy"], base + z])
    sv[:, 2] += max(0.0, base + p.radius - sv[:, 2].min())  # nothing starts inside the table

    # render: photo on the front, the generated underside on the back
    gen_tex = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
    back = back_view(p2, ruv, rf, gen_tex, bot_ids, bgr, mask)
    cv2.imwrite(str(out / "texture.png"), np.hstack([bleed(bgr, mask), bleed(back, mask)]))
    write_skn(out / "cloth.skn", *layer_skin(pts, tris, is_back, sv, mask.shape), sv)
    write_obj(out / "cloth.obj", sv, tris)
    return dict(verts=sv, tris=tris, edge=o["edge"], area=o["area"], top=~is_back)
