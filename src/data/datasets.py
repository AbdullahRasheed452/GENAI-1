import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.corruptions import CLASSES, apply, sample_params, sample_random

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
SPLIT = ROOT / "configs" / "pet_split.json"
MANIFESTS = {
    "val": ROOT / "configs" / "val_manifest.json",
    "test": ROOT / "configs" / "test_manifest.json",
}
SEVERITY_IDS = {"none": 0, "low": 1, "medium": 2, "high": 3}


def to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


class PetTrainDataset(Dataset):
    """Returns (corrupted, clean, label). Corruption is sampled at load time."""

    def __init__(self, condition="random", seed=42, cache_dir=CACHE):
        self.images = np.load(Path(cache_dir) / "train.npy", mmap_mode="r")
        self.condition = condition
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        n = len(self.images)
        rng = np.random.default_rng([self.seed, 99, epoch])
        self.balanced_kinds = rng.permutation(np.arange(n) % len(CLASSES))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        if self.condition == "random":
            params = sample_random(rng)
        elif self.condition == "balanced":
            params = sample_params(CLASSES[int(self.balanced_kinds[idx])], rng)
        else:
            params = sample_params(self.condition, rng)
        image = np.array(self.images[idx])
        corrupted = apply(image, params)
        return to_tensor(corrupted), to_tensor(image), CLASSES.index(params["type"])


class ManifestDataset(Dataset):
    """Returns (corrupted, clean, label, severity, entry_index) from a fixed manifest."""

    def __init__(self, split, cache_dir=CACHE):
        names = json.loads(SPLIT.read_text())[split]
        self.index = {name: i for i, name in enumerate(names)}
        self.images = np.load(Path(cache_dir) / f"{split}.npy", mmap_mode="r")
        self.entries = json.loads(MANIFESTS[split].read_text())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = np.array(self.images[self.index[entry["image"]]])
        corrupted = apply(image, entry)
        severity = SEVERITY_IDS.get(entry.get("severity"), -1)
        return to_tensor(corrupted), to_tensor(image), CLASSES.index(entry["type"]), severity, idx
