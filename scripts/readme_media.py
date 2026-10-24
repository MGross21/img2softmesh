"""Build the README gallery: input thumbnails + small autoplay GIFs from out/<name>/demo.mp4.

Run after the example demos (see examples/README.md):  uv run python scripts/readme_media.py
"""

from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
# output folder -> example input
EXAMPLES = {
    "tshirt_trellis": "tshirt",
    "tshirt": "tshirt",
    "redtee": "redtee",
    "sock": "sock",
    "towel": "towel",
    "blanket": "blanket",
}


def to_gif(video, dst, width=380, fps=10, colors=256):
    cap = cv2.VideoCapture(str(video))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / fps))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(f[:, :, ::-1]))
        i += 1
    # one shared palette from frames across the whole clip: no palette flicker, all phases' colors covered
    picks = frames[:: max(1, len(frames) // 12)]
    strip = Image.new("RGB", (width, sum(fr.height for fr in picks)))
    for k, fr in enumerate(picks):
        strip.paste(fr, (0, k * fr.height))
    palette = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True)


def thumbnail(image, dst, height=240):
    im = cv2.imread(str(image))
    im = cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    for out, example in EXAMPLES.items():
        thumbnail(ROOT / "examples" / f"{example}.jpg", MEDIA / f"{example}.jpg")
        to_gif(ROOT / "out" / out / "demo.mp4", MEDIA / f"{out}.gif")
        print(f"{out}: {(MEDIA / f'{out}.gif').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
