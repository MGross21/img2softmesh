"""Writers: OBJ (+MTL) and a MuJoCo MJCF scene with the cloth as a <flexcomp>."""

from pathlib import Path


def write_obj(path, verts, tris, uvs=None, texture="texture.png"):
    path = Path(path)
    lines = ["o cloth"]
    if uvs is not None:
        mtl = path.with_suffix(".mtl")
        mtl.write_text(f"newmtl cloth\nKd 1 1 1\nKs 0 0 0\nmap_Kd {texture}\n")
        lines = [f"mtllib {mtl.name}", "o cloth", "usemtl cloth"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    if uvs is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
        # one uv per vertex, so v and vt share indices
        lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in t) for t in tris]
    else:
        lines += ["f " + " ".join(str(i + 1) for i in t) for t in tris]
    path.write_text("\n".join(lines) + "\n")


SCENE = Path(__file__).with_name("scene.xml")
GRIPPER = '    <body name="gripper{g}" mocap="true" pos="{pos}"><geom class="gripper"/></body>'
GRASP = '    <connect name="grasp{g}_{i}" body1="cloth_{v}" body2="gripper{g}" anchor="0 0 0"/>'


def write_scene(path, *, name, kind, preset, mass, table, grippers, selfcollide="none"):
    """Fill scene.xml: table (half-extents x, y, z), cloth flexcomp from the fabric preset, scripted grippers.

    grippers: [(position, cloth vertex ids it holds), ...]
    """
    p = preset
    xml = SCENE.read_text().format(
        name=name, kind=kind, radius=p.radius, mass=mass, selfcollide=selfcollide, grip_r=max(2.5 * p.radius, 0.01),
        table_hx=table[0], table_hy=table[1], table_hz=table[2],
        edge_eq=str(p.edge_equality).lower(), young=p.young, poisson=p.poisson, thickness=p.thickness,
        damping=p.damping, elastic2d=p.elastic2d,
        grippers="\n".join(GRIPPER.format(g=g, pos=" ".join(f"{v:.5f}" for v in pos))
                           for g, (pos, _) in enumerate(grippers)),
        grasp="\n".join(GRASP.format(g=g, i=i, v=v) for g, (_, ids) in enumerate(grippers) for i, v in enumerate(ids)))
    Path(path).write_text(xml)
