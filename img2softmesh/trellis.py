"""Local image -> textured 3D via TRELLIS.2 (Q4 GGUF) on trellis.cpp.

The 13 MB Vulkan build runs on any recent NVIDIA/AMD/Intel GPU with no CUDA/Python stack.
Binary + ~4 GB of weights are downloaded once into $IMG2SOFTMESH_HOME (default ~/.cache/img2softmesh).
"""

import hashlib
import os
import platform
import subprocess
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/pwilkin/trellis.cpp/releases/download/v0.8.1"
WEIGHTS = "https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
STAGES = {512: ["dinov3", "ss_flow", "ss_dec", "shape_flow_512", "shape_dec", "tex_flow_512", "tex_dec"]}
STAGES[1024] = STAGES[512] + ["shape_flow_1024", "tex_flow_1024"]


def home():
    return Path(os.environ.get("IMG2SOFTMESH_HOME", Path.home() / ".cache" / "img2softmesh"))


def _download(url, dest):
    dest = Path(dest)
    with urllib.request.urlopen(url) as r:
        size = int(r.headers.get("Content-Length", 0))
        if dest.exists() and dest.stat().st_size == size:
            return dest
        print(f"downloading {dest.name} ({size / 2**20:.0f} MB)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    if size and tmp.stat().st_size != size:
        raise OSError(f"truncated download: {dest.name}")
    tmp.replace(dest)
    return dest


def ensure(res=512, quant="q4", backend="vulkan"):
    """Return (trellis-cli path, model dir), downloading whatever is missing."""
    system = {"Windows": "windows", "Linux": "linux"}.get(platform.system())
    if system is None:
        raise RuntimeError("prebuilt trellis.cpp binaries exist for Windows and Linux; build it from source on macOS")
    bin_dir = home() / f"trellis-{backend}"
    exe = bin_dir / ("trellis-cli.exe" if system == "windows" else "trellis-cli")
    if not exe.exists():
        ext = "zip" if system == "windows" else "tar.gz"
        archive = _download(f"{RELEASE}/trellis-{backend}-{system}-x64.{ext}", home() / f"trellis-{backend}.{ext}")
        bin_dir.mkdir(parents=True, exist_ok=True)
        if ext == "zip":
            zipfile.ZipFile(archive).extractall(bin_dir)
        else:
            import tarfile
            tarfile.open(archive).extractall(bin_dir, filter="data")
        exe = next(bin_dir.rglob(exe.name))
        exe.chmod(0o755)
    models = home() / f"trellis2-{quant}"
    models.mkdir(parents=True, exist_ok=True)
    for stage in STAGES[res]:
        _download(f"{WEIGHTS}/{quant}/{stage}.gguf", models / f"{stage}.gguf")
    return exe, models


def generate(image_rgba, out_glb, res=512, seed=42, quant="q4"):
    """Run TRELLIS.2 on a pre-matted RGBA image; writes a GLB with UVs + PNG base-color texture.

    Skips the run when out_glb was already generated from the same input and settings.
    """
    exe, models = ensure(res, quant)
    cmd = [str(exe), "-i", str(image_rgba), "-o", str(out_glb), "-m", str(models),
           "--res", str(res), "--seed", str(seed), "--atlas", "2048", "--webp", "off", "--require-gpu"]
    out_glb, stamp = Path(out_glb), Path(str(out_glb) + ".key")
    key = hashlib.sha256(Path(image_rgba).read_bytes() + " ".join(cmd[5:]).encode()).hexdigest()
    if out_glb.exists() and stamp.exists() and stamp.read_text() == key:
        print(f"reusing {out_glb.name} (same input)")
        return out_glb
    subprocess.run(cmd, check=True)
    stamp.write_text(key)
    return out_glb
