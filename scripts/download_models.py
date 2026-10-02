"""Download the seven ONNX models into models_onnx (skips files that are already present)."""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1O5NRmpWf2LKDXC6hxI7BO0HMAYznd3yv"
EXPECTED_MB = {
    "task1_universal_ae.onnx": 76.8,
    "task2_classifier.onnx": 1.2,
    "task2_specialist_salt_pepper.onnx": 45.4,
    "task2_specialist_blur.onnx": 45.4,
    "task2_specialist_occlusion.onnx": 45.4,
    "task3_soft_moe.onnx": 137.3,
    "task4_generator.onnx": 168.6,
}


def size_ok(path, expected_mb):
    if not path.exists():
        return False
    mb = path.stat().st_size / 1e6
    return 0.9 * expected_mb <= mb <= 1.1 * expected_mb


def missing(dest):
    return [name for name, mb in EXPECTED_MB.items() if not size_ok(dest / name, mb)]


def main():
    default_dest = Path(__file__).resolve().parents[1] / "models_onnx"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(default_dest))
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    todo = missing(dest)
    print(f"{len(EXPECTED_MB) - len(todo)} of {len(EXPECTED_MB)} models already present in {dest}")
    if not todo:
        print("Nothing to download.")
        return

    try:
        import gdown
    except ImportError:
        sys.exit("The gdown package is missing. Run:  python -m pip install gdown")

    print("Downloading (about 520 MB in total, this can take several minutes)...")
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(url=FOLDER_URL, output=tmp, quiet=False)
        for name in todo:
            found = list(Path(tmp).rglob(name))
            if found:
                shutil.move(str(found[0]), str(dest / name))

    still = missing(dest)
    if still:
        print("\nThese files are still missing or have the wrong size:")
        for name in still:
            print("  ", name)
        print("Download them by hand from:", FOLDER_URL)
        sys.exit(1)
    print(f"\nAll {len(EXPECTED_MB)} models are in {dest}")


if __name__ == "__main__":
    main()


