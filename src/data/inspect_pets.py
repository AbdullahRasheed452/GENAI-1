import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "oxford_pets" / "images"
SPLIT = ROOT / "configs" / "pet_split.json"
PREVIEW = ROOT / "outputs" / "pet_preview.png"
SIZE = 128


def main():
    split = json.loads(SPLIT.read_text())
    names = split["train"] + split["val"] + split["test"]

    modes, formats, widths, heights, broken = Counter(), Counter(), [], [], []
    for name in names:
        try:
            with Image.open(IMAGES / f"{name}.jpg") as img:
                modes[img.mode] += 1
                formats[img.format] += 1
                widths.append(img.width)
                heights.append(img.height)
                img.load()
        except Exception as error:
            broken.append((name, str(error)))

    print(f"checked {len(names)} images")
    print("color modes:", dict(modes))
    print("file formats:", dict(formats))
    print(f"width range: {min(widths)} to {max(widths)}")
    print(f"height range: {min(heights)} to {max(heights)}")
    print(f"unreadable: {len(broken)}")
    for name, error in broken[:5]:
        print("  ", name, error)

    sample = split["train"][:: len(split["train"]) // 12][:12]
    canvas = Image.new("RGB", (SIZE * 6, SIZE * 2))
    for i, name in enumerate(sample):
        with Image.open(IMAGES / f"{name}.jpg") as img:
            tile = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        canvas.paste(tile, ((i % 6) * SIZE, (i // 6) * SIZE))
    PREVIEW.parent.mkdir(exist_ok=True)
    canvas.save(PREVIEW)
    print(f"preview grid saved to {PREVIEW}")


if __name__ == "__main__":
    main()
