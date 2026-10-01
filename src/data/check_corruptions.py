import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.corruptions import (
    CLASSES,
    IMAGE_SIZE,
    TEST_LEVELS,
    apply,
    gaussian_blur,
    salt_and_pepper,
    sample_random,
    sample_rects,
)
from src.data.image_io import load_rgb

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "oxford_pets" / "images"
SPLIT = ROOT / "configs" / "pet_split.json"
PREVIEW = ROOT / "outputs" / "corruption_preview.png"


def check_random_sampling(rng, n=4000):
    kinds, kernels, counts = Counter(), Counter(), Counter()
    probs, sigmas, covs = [], [], []
    for _ in range(n):
        p = sample_random(rng)
        kinds[p["type"]] += 1
        if p["type"] == "salt_pepper":
            probs.append(p["prob"])
        elif p["type"] == "blur":
            kernels[p["kernel_size"]] += 1
            sigmas.append(p["sigma"])
        elif p["type"] == "occlusion":
            covs.append(p["coverage"])
            counts[len(p["rects"])] += 1
    print(f"class frequency over {n} draws:", {k: f"{kinds[k] / n:.3f}" for k in CLASSES})
    print(f"salt-and-pepper prob range: {min(probs):.3f} to {max(probs):.3f}")
    print("blur kernel sizes:", dict(sorted(kernels.items())))
    print(f"blur sigma range: {min(sigmas):.2f} to {max(sigmas):.2f}")
    print(f"occlusion coverage range: {min(covs):.3f} to {max(covs):.3f}")
    print("occlusion rectangle counts:", dict(sorted(counts.items())))


def check_salt_and_pepper():
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    out = salt_and_pepper(gray, 0.08, seed=0)
    changed = out[..., 0] != 128
    same_channels = bool(np.all(out[changed] == out[changed][:, :1]))
    print(f"salt-and-pepper at p=0.08: {changed.mean():.3f} of pixels hit, "
          f"{(out[..., 0][changed] == 255).mean():.3f} of hits white, "
          f"channels identical: {same_channels}")


def check_blur():
    flat = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 100, dtype=np.uint8)
    out = gaussian_blur(flat, 7, 2.5)
    print(f"blur on a flat image leaves it unchanged: {bool(np.all(out == 100))}")


def check_test_levels(rng):
    for level in TEST_LEVELS["occlusion"]:
        covs = [sample_rects(rng, level["target_coverage"], level["count"], tolerance=0.005) for _ in range(200)]
        from src.data.corruptions import coverage
        values = [coverage(r) for r in covs]
        print(f"test occlusion {level}: coverage {min(values):.3f} to {max(values):.3f}")


def fixed_params(kind, level, rng):
    if kind == "salt_pepper":
        return {"type": kind, "prob": level["prob"], "seed": int(rng.integers(2 ** 31))}
    if kind == "blur":
        return {"type": kind, **level}
    return {"type": kind, "rects": sample_rects(rng, level["target_coverage"], level["count"])}


def save_preview(rng):
    split = json.loads(SPLIT.read_text())
    names = split["val"][:3]
    columns = [{"type": "clean"}] + [
        (kind, level) for kind in ("salt_pepper", "blur", "occlusion") for level in TEST_LEVELS[kind]
    ]
    canvas = Image.new("RGB", (IMAGE_SIZE * len(columns), IMAGE_SIZE * len(names)))
    for row, name in enumerate(names):
        image = load_rgb(IMAGES / f"{name}.jpg")
        for col, spec in enumerate(columns):
            params = spec if isinstance(spec, dict) else fixed_params(spec[0], spec[1], rng)
            tile = Image.fromarray(apply(image, params))
            canvas.paste(tile, (col * IMAGE_SIZE, row * IMAGE_SIZE))
    PREVIEW.parent.mkdir(exist_ok=True)
    canvas.save(PREVIEW)
    print(f"preview saved to {PREVIEW}")


def main():
    rng = np.random.default_rng(0)
    check_random_sampling(rng)
    check_salt_and_pepper()
    check_blur()
    check_test_levels(rng)
    save_preview(rng)


if __name__ == "__main__":
    main()

