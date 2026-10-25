<div align="center">

<img src="docs/media/logo.svg" alt="img2softmesh" width="460">

Turn a photo of cloth into a textured, simulation-ready soft mesh.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![GPU optional](https://img.shields.io/badge/GPU-optional-lightgrey)

</div>

```mermaid
flowchart LR
    A[photo] --> B[segment]
    B --> C[silhouette<br><sub>fast, CPU</sub>]
    B --> D[TRELLIS.2<br><sub>3D shape, local GPU</sub>]
    C --> E[textured cloth mesh]
    D --> E
    E -.-> F[grasp demo]
```

| Photo | Demo |
|:---:|:---:|
| <img src="docs/media/tshirt.jpg" height="160"> | <img src="docs/media/tshirt.gif" width="380"> |
| <img src="docs/media/sock.jpg" height="160"> | <img src="docs/media/sock.gif" width="380"> |
| <img src="docs/media/towel.jpg" height="160"> | <img src="docs/media/towel.gif" width="380"> |
| <img src="docs/media/blanket.jpg" height="160"> | <img src="docs/media/blanket.gif" width="380"> |

## Quick start

```bash
uv sync --extra rembg --extra trellis
uv run img2softmesh demo examples/tshirt.jpg --kind tshirt
```

```bash
img2softmesh make photo.jpg --kind towel     # mesh + texture in out/photo/
img2softmesh make photo.jpg -g trellis       # use TRELLIS.2 for 3D shape
img2softmesh sim out/photo                   # run the demo
```

Kinds: `towel` `blanket` `sheet` `napkin` `flat` `tshirt` `sock` `pillowcase` (or `auto`).
Lay the item flat and shoot from above.
