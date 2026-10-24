"""img2softmesh command line."""

import argparse
import json
from pathlib import Path

from .presets import PRESETS


def _add_build_args(p):
    p.add_argument("image", help="photo of the item (PNG with transparency skips segmentation)")
    p.add_argument("-k", "--kind", default="auto", choices=["auto", *PRESETS],
                   help="item preset (default: auto = towel if rectangular, else garment)")
    p.add_argument("-s", "--size", type=float, help="longest side in meters (default: preset)")
    p.add_argument("-n", "--verts", type=int, help="target vertices per layer (default: preset)")
    p.add_argument("-r", "--radius", type=float, help="contact thickness in meters, shell = 2x (default: preset)")
    p.add_argument("--segment", default="auto", choices=["auto", "alpha", "rembg", "grabcut"])
    p.add_argument("-g", "--geometry", default="silhouette", choices=["silhouette", "trellis"],
                   help="silhouette: mesh from the outline, photo as texture (fast, CPU). "
                        "trellis: TRELLIS.2 3D + 360-degree texture (local GPU, downloads ~4 GB once)")
    p.add_argument("--selfcollide", action="store_const", const="auto", default="none",
                   help="cloth-cloth collision (realistic folds, much slower)")
    p.add_argument("-o", "--out", help="output directory (default: out/<image name>)")


def _build(a):
    from .pipeline import build
    info = build(a.image, kind=a.kind, size=a.size, verts=a.verts, radius=a.radius, segment_method=a.segment,
                 geometry=a.geometry, out_dir=a.out, selfcollide=a.selfcollide)
    print(json.dumps({k: info[k] for k in ("kind", "vertices", "triangles", "size_m", "mass_kg", "scene")}, indent=2))
    return info


def _sim(scene, video):
    from . import demo
    if video:
        rt = demo.record(scene, video)
        print(f"wrote {video}  ({rt:.2f}x real time)")
    else:
        demo.view(scene)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="img2softmesh", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    _add_build_args(sub.add_parser("make", help="image -> textured cloth mesh + MuJoCo scene"))

    s = sub.add_parser("sim", help="run the scripted grasp demo on a generated scene")
    s.add_argument("scene", help="scene.xml (or its directory)")
    s.add_argument("--video", help="render to mp4 instead of opening the viewer")

    d = sub.add_parser("demo", help="make + sim in one go")
    _add_build_args(d)
    d.add_argument("--video", help="render to mp4 instead of opening the viewer")

    a = ap.parse_args(argv)
    if a.cmd == "make":
        _build(a)
    elif a.cmd == "sim":
        scene = Path(a.scene)
        _sim(scene / "scene.xml" if scene.is_dir() else scene, a.video)
    else:
        _sim(_build(a)["scene"], a.video)


if __name__ == "__main__":
    main()
