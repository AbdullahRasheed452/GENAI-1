import numpy as np

CLASSES = ("clean", "salt_pepper", "blur", "occlusion")
IMAGE_SIZE = 128

SP_PROB_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_COVERAGE_RANGE = (0.10, 0.35)
OCC_COUNT_RANGE = (1, 3)
OCC_TOLERANCE = 0.02
OCC_ASPECT_RANGE = (0.5, 2.0)
OCC_SHARE_CONCENTRATION = 4.0

TEST_LEVELS = {
    "salt_pepper": [{"prob": 0.03}, {"prob": 0.08}, {"prob": 0.15}],
    "blur": [
        {"kernel_size": 3, "sigma": 0.7},
        {"kernel_size": 5, "sigma": 1.5},
        {"kernel_size": 7, "sigma": 2.5},
    ],
    "occlusion": [
        {"target_coverage": 0.10, "count": 1},
        {"target_coverage": 0.20, "count": 2},
        {"target_coverage": 0.35, "count": 3},
    ],
}


def salt_and_pepper(image, prob, seed):
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    out = image.copy()
    hit = rng.random((h, w)) < prob
    white = rng.random((h, w)) < 0.5
    out[hit & white] = 255
    out[hit & ~white] = 0
    return out


def gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(image, kernel_size, sigma):
    k = gaussian_kernel(kernel_size, sigma)
    r = kernel_size // 2
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    padded = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="reflect")
    rows = sum(k[i] * padded[i:i + h] for i in range(kernel_size))
    padded = np.pad(rows, ((0, 0), (r, r), (0, 0)), mode="reflect")
    out = sum(k[i] * padded[:, i:i + w] for i in range(kernel_size))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def occlusion(image, rects):
    out = image.copy()
    for x, y, w, h in rects:
        out[y:y + h, x:x + w] = 0
    return out


def coverage(rects, size=IMAGE_SIZE):
    mask = np.zeros((size, size), dtype=bool)
    for x, y, w, h in rects:
        mask[y:y + h, x:x + w] = True
    return float(mask.mean())


def sample_rects(rng, target, count, size=IMAGE_SIZE, max_tries=5000, tolerance=OCC_TOLERANCE):
    low, high = OCC_COVERAGE_RANGE
    for _ in range(max_tries):
        shares = rng.dirichlet(np.full(count, OCC_SHARE_CONCENTRATION))
        rects = []
        for share in shares:
            area = target * share * size * size
            aspect = rng.uniform(*OCC_ASPECT_RANGE)
            w = int(round(np.sqrt(area * aspect)))
            h = int(round(area / max(w, 1)))
            w = max(1, min(w, size))
            h = max(1, min(h, size))
            x = int(rng.integers(0, size - w + 1))
            y = int(rng.integers(0, size - h + 1))
            rects.append([x, y, w, h])
        cov = coverage(rects, size)
        if abs(cov - target) <= tolerance and low - 1e-9 <= cov <= high + 1e-9:
            return rects
    raise RuntimeError(f"could not place {count} rectangles covering {target:.2f}")


def sample_params(kind, rng):
    if kind == "clean":
        return {"type": "clean"}
    if kind == "salt_pepper":
        return {
            "type": kind,
            "prob": float(rng.uniform(*SP_PROB_RANGE)),
            "seed": int(rng.integers(2 ** 31)),
        }
    if kind == "blur":
        return {
            "type": kind,
            "kernel_size": int(rng.choice(BLUR_KERNELS)),
            "sigma": float(rng.uniform(*BLUR_SIGMA_RANGE)),
        }
    if kind == "occlusion":
        target = float(rng.uniform(*OCC_COVERAGE_RANGE))
        count = int(rng.integers(OCC_COUNT_RANGE[0], OCC_COUNT_RANGE[1] + 1))
        rects = sample_rects(rng, target, count)
        return {"type": kind, "target_coverage": target, "coverage": coverage(rects), "rects": rects}
    raise ValueError(f"unknown corruption: {kind}")


def sample_random(rng):
    kind = CLASSES[int(rng.integers(len(CLASSES)))]
    return sample_params(kind, rng)


def apply(image, params):
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image of shape {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {image.dtype} {image.shape}")
    kind = params["type"]
    if kind == "clean":
        return image.copy()
    if kind == "salt_pepper":
        return salt_and_pepper(image, params["prob"], params["seed"])
    if kind == "blur":
        return gaussian_blur(image, params["kernel_size"], params["sigma"])
    if kind == "occlusion":
        return occlusion(image, params["rects"])
    raise ValueError(f"unknown corruption: {kind}")

