from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset, PetTrainDataset


def label_counts(dataset):
    return dict(sorted(Counter(dataset[i][2] for i in range(len(dataset))).items()))


def main():
    train = PetTrainDataset()
    corrupted, clean, label = train[0]
    print("item:", tuple(corrupted.shape), corrupted.dtype, f"range {corrupted.min():.2f} to {corrupted.max():.2f}", CLASSES[label])
    assert corrupted.shape == (3, 128, 128) and clean.shape == (3, 128, 128)

    a, b = train[5], train[5]
    assert torch.equal(a[0], b[0]) and a[2] == b[2]
    print("same epoch, same index: identical")

    epoch0 = [train[i][2] for i in range(len(train))]
    train.set_epoch(1)
    epoch1 = [train[i][2] for i in range(len(train))]
    print("random mode labels change between epochs:", epoch0 != epoch1)
    print("random mode counts:", dict(sorted(Counter(epoch0).items())))

    print("balanced mode counts:", label_counts(PetTrainDataset(condition="balanced")))

    for kind in CLASSES[1:]:
        ds = PetTrainDataset(condition=kind)
        assert all(ds[i][2] == CLASSES.index(kind) for i in range(40))
    print("single-corruption modes: correct")

    val, test = ManifestDataset("val"), ManifestDataset("test")
    print("val size:", len(val), " test size:", len(test))
    x1, _, label, severity, idx = test[7]
    x2 = test[7][0]
    assert torch.equal(x1, x2)
    print("test item 7:", test.entries[7]["type"], test.entries[7]["severity"], "-> label", label, "severity id", severity)
    assert CLASSES[label] == test.entries[7]["type"]

    loader = DataLoader(PetTrainDataset(condition="balanced"), batch_size=32, shuffle=True, num_workers=2)
    corrupted, clean, labels = next(iter(loader))
    print("batch:", tuple(corrupted.shape), tuple(clean.shape), tuple(labels.shape))


if __name__ == "__main__":
    main()
