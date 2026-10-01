import tarfile
import urllib.request
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "oxford_pets"
BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
ARCHIVES = ("images.tar.gz", "annotations.tar.gz")


def download(name):
    target = DATA_ROOT / name
    if target.exists():
        print(f"{name} already downloaded")
        return target

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            print(f"\r{name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")

    urllib.request.urlretrieve(BASE_URL + name, target, progress)
    print()
    return target


def extract(archive, expected_folder):
    if (DATA_ROOT / expected_folder).exists():
        print(f"{expected_folder}/ already extracted")
        return
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA_ROOT, filter="data")


def read_split(name):
    lines = (DATA_ROOT / "annotations" / name).read_text().splitlines()
    return [line.split()[0] for line in lines if line.strip()]


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    extract(download("images.tar.gz"), "images")
    extract(download("annotations.tar.gz"), "annotations")

    for split in ("trainval", "test"):
        names = read_split(f"{split}.txt")
        missing = [n for n in names if not (DATA_ROOT / "images" / f"{n}.jpg").exists()]
        print(f"{split}: {len(names)} listed, {len(missing)} missing on disk")


if __name__ == "__main__":
    main()
