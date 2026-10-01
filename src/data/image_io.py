import numpy as np
from PIL import Image

IMAGE_SIZE = 128


def load_rgb(path, size=IMAGE_SIZE):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()
