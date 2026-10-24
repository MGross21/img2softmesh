"""Per-item geometry + fabric presets.

layers=1 -> single cloth sheet (towel, blanket...).
layers=2 -> front/back sheets welded along the silhouette (t-shirt, sock...).

Physics values map onto MuJoCo <flexcomp> attributes. They are hand-tuned
starting points, not measured fabric data.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Preset:
    layers: int
    size: float          # longest side of the item, meters
    density: float       # areal density, kg/m^2 (per layer)
    young: float         # Young's modulus, Pa
    poisson: float
    thickness: float     # shell thickness used for bending stiffness, m
    edge_equality: bool  # near-inextensible edges (woven) vs stretchy membrane (knit)
    elastic2d: Literal["bend", "stretch", "both"]
    damping: float
    verts: int = 300     # target vertex count for one layer (MuJoCo flex cost grows fast)
    radius: float = 0.002  # contact thickness: cloth collides as a shell 2 * radius thick, m
    grasp: tuple[str, ...] = ("left", "right")  # grip points: top-left / top-right corner, "collar" = top center


SHIRT = ("left", "collar", "right")  # both sleeves + neckline

PRESETS = {
    # Preset(layers, size, density, young, poisson, thickness, edge_equality, elastic2d, damping, ...)
    # flat / single layer
    "towel":      Preset(1, 0.70, 0.45, 5e4, 0.0, 3.0e-3, True, "bend", 2e-3, radius=0.003),
    "blanket":    Preset(1, 1.50, 0.30, 3e4, 0.0, 4.0e-3, True, "bend", 2e-3, verts=400, radius=0.004),
    "sheet":      Preset(1, 2.00, 0.12, 3e4, 0.0, 0.8e-3, True, "bend", 1e-3, verts=400, radius=0.0015),
    "napkin":     Preset(1, 0.45, 0.15, 3e4, 0.0, 1.0e-3, True, "bend", 1e-3, radius=0.0015),
    "flat":       Preset(1, 0.70, 0.20, 3e4, 0.0, 1.5e-3, True, "bend", 1e-3),
    # sewn / two layer
    "tshirt":     Preset(2, 0.70, 0.15, 4e4, 0.3, 1.2e-3, True, "bend", 1e-3, grasp=SHIRT),
    "sock":       Preset(2, 0.25, 0.25, 2e4, 0.3, 1.5e-3, False, "both", 2e-3, verts=200),
    "pillowcase": Preset(2, 0.70, 0.12, 3e4, 0.0, 1.0e-3, True, "bend", 1e-3),
    "garment":    Preset(2, 0.70, 0.15, 4e4, 0.3, 1.2e-3, True, "bend", 1e-3, grasp=SHIRT),
}
