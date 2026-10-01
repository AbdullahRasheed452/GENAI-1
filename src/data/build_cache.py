import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.image_io import load_rgb

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "oxford_pets" / "images"
SPLIT = ROOT / "configs" / "pet_split.json"
CACHE = ROOT / "data" / "cache"


def main():
    split = json.loads(SPLIT.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        names = split[name]
        array = np.stack([load_rgb(IMAGES / f"{n}.jpg") for n in tqdm(names, desc=name)])
        np.save(CACHE / f"{name}.npy", array)
        print(f"{name}: shape {array.shape}, dtype {array.dtype}, {array.nbytes / 1e6:.0f} MB")

        saved = np.load(CACHE / f"{name}.npy", mmap_mode="r")
        assert saved.shape == (len(names), 128, 128, 3)
        assert np.array_equal(saved[0], load_rgb(IMAGES / f"{names[0]}.jpg"))
        assert np.array_equal(saved[-1], load_rgb(IMAGES / f"{names[-1]}.jpg"))


if __name__ == "__main__":
    main()
