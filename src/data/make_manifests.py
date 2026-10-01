import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.corruptions import CLASSES, TEST_LEVELS, coverage, sample_params, sample_rects

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "configs" / "pet_split.json"
VAL_MANIFEST = ROOT / "configs" / "val_manifest.json"
TEST_MANIFEST = ROOT / "configs" / "test_manifest.json"
BASE_SEED = 42
TEST_OCC_TOLERANCE = 0.005
SEVERITY_NAMES = ("low", "medium", "high")


def build_val(names):
    rng = np.random.default_rng([BASE_SEED, 1])
    kinds = rng.permutation(np.arange(len(names)) % len(CLASSES))
    entries = []
    for name, kind in zip(names, kinds):
        entry = {"image": name}
        entry.update(sample_params(CLASSES[int(kind)], rng))
        entries.append(entry)
    return entries


def build_test(names):
    entries = []
    for i, name in enumerate(names):
        entries.append({"image": name, "type": "clean", "severity": "none"})
        for kind in ("salt_pepper", "blur", "occlusion"):
            for level_index, level in enumerate(TEST_LEVELS[kind]):
                rng = np.random.default_rng([BASE_SEED, 2, i, CLASSES.index(kind), level_index])
                entry = {"image": name, "type": kind, "severity": SEVERITY_NAMES[level_index]}
                if kind == "salt_pepper":
                    entry["prob"] = level["prob"]
                    entry["seed"] = int(rng.integers(2 ** 31))
                elif kind == "blur":
                    entry["kernel_size"] = level["kernel_size"]
                    entry["sigma"] = level["sigma"]
                else:
                    rects = sample_rects(
                        rng, level["target_coverage"], level["count"], tolerance=TEST_OCC_TOLERANCE
                    )
                    entry["target_coverage"] = level["target_coverage"]
                    entry["coverage"] = coverage(rects)
                    entry["rects"] = rects
                entries.append(entry)
    return entries


def save(path, entries):
    path.write_text(json.dumps(entries, separators=(",", ":")))
    print(f"saved {len(entries)} entries to {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    split = json.loads(SPLIT.read_text())

    val = build_val(split["val"])
    assert val == build_val(split["val"]), "validation manifest is not reproducible"
    save(VAL_MANIFEST, val)
    print("validation types:", dict(Counter(e["type"] for e in val)))

    test = build_test(split["test"])
    assert build_test(split["test"][:200]) == test[:2000], "test manifest is not reproducible"
    save(TEST_MANIFEST, test)
    print("test types:", dict(Counter(e["type"] for e in test)))
    print("test severities:", dict(Counter(e["severity"] for e in test)))
    for severity in SEVERITY_NAMES:
        covs = [e["coverage"] for e in test if e["type"] == "occlusion" and e["severity"] == severity]
        print(f"test occlusion {severity}: coverage {min(covs):.3f} to {max(covs):.3f}")


if __name__ == "__main__":
    main()
