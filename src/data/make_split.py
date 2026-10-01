import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "data" / "oxford_pets" / "annotations"
OUTPUT = ROOT / "configs" / "pet_split.json"
SEED = 42
VAL_FRACTION = 0.2


def read_names(filename):
    lines = (ANNOTATIONS / filename).read_text().splitlines()
    return sorted(line.split()[0] for line in lines if line.strip())


def main():
    trainval = read_names("trainval.txt")
    test = read_names("test.txt")

    order = np.random.default_rng(SEED).permutation(len(trainval))
    n_val = int(round(len(trainval) * VAL_FRACTION))
    val = sorted(trainval[i] for i in order[:n_val])
    train = sorted(trainval[i] for i in order[n_val:])

    assert not set(train) & set(val), "train and val overlap"
    assert not (set(train) | set(val)) & set(test), "test leaked into development data"
    assert len(train) + len(val) == len(trainval)

    split = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train": train,
        "val": val,
        "test": test,
    }
    OUTPUT.write_text(json.dumps(split, indent=1))
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f"saved to {OUTPUT}")


if __name__ == "__main__":
    main()
