import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
VAL_FRACTION = 0.15
SIZE = 128
NUM_STYLES = 3


def first_existing(stem):
    for ext in (".jpg", ".png", ".jpeg"):
        path = Path(str(stem) + ext)
        if path.exists():
            return path
    return None


def load(path):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()


def collect(root, annotations):
    items, dropped = [], []
    for a in annotations:
        folder, name = a["image_name"].split("/")
        k, number = folder[-1], name.replace("image", "")
        photo = first_existing(root / "photo" / folder / name)
        sketch = first_existing(root / "sketch" / f"sketch{k}" / f"sketch{number}")
        if photo is None or sketch is None:
            dropped.append(a["image_name"])
            continue
        items.append({"name": a["image_name"], "style": int(a["style"]), "photo": photo, "sketch": sketch})
    return items, dropped


def stratified_val_indices(styles, seed, fraction):
    rng = np.random.default_rng(seed)
    val = []
    for s in range(NUM_STYLES):
        idx = rng.permutation(np.flatnonzero(styles == s))
        val.extend(idx[: int(round(len(idx) * fraction))].tolist())
    return set(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cache_dir", default=str(ROOT / "data" / "fs2k_cache"))
    args = parser.parse_args()

    root = Path(args.root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    train_items, train_dropped = collect(root, json.loads((root / "anno_train.json").read_text()))
    test_items, test_dropped = collect(root, json.loads((root / "anno_test.json").read_text()))
    print("dropped from train:", train_dropped)
    print("dropped from test:", test_dropped)

    styles = np.array([it["style"] for it in train_items])
    val_idx = stratified_val_indices(styles, SEED, VAL_FRACTION)
    splits = {
        "train": [it for i, it in enumerate(train_items) if i not in val_idx],
        "val": [it for i, it in enumerate(train_items) if i in val_idx],
        "test": test_items,
    }

    record = {"seed": SEED, "val_fraction": VAL_FRACTION, "dropped_train": train_dropped, "dropped_test": test_dropped}
    for split, items in splits.items():
        photos = np.stack([load(it["photo"]) for it in items])
        sketches = np.stack([load(it["sketch"]) for it in items])
        style = np.array([it["style"] for it in items], dtype=np.int64)
        np.save(cache / f"{split}_photo.npy", photos)
        np.save(cache / f"{split}_sketch.npy", sketches)
        np.save(cache / f"{split}_style.npy", style)
        record[split] = [it["name"] for it in items]
        print(f"{split}: {len(items)} pairs, per style {np.bincount(style, minlength=NUM_STYLES).tolist()}, array {photos.shape}")
    (cache / "fs2k_split.json").write_text(json.dumps(record))

    columns = []
    for s in range(NUM_STYLES):
        columns += [it for it in splits["train"] if it["style"] == s][:4]
    canvas = Image.new("RGB", (SIZE * len(columns), SIZE * 2))
    for c, it in enumerate(columns):
        canvas.paste(Image.fromarray(load(it["photo"])), (c * SIZE, 0))
        canvas.paste(Image.fromarray(load(it["sketch"])), (c * SIZE, SIZE))
    canvas.save(cache / "fs2k_pairs_preview.png")
    print("preview saved: top row photos, bottom row sketches, 4 pairs each for style 0, 1, 2")


if __name__ == "__main__":
    main()
