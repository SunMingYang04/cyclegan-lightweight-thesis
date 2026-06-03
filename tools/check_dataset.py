import argparse
from pathlib import Path

from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


def main():
    parser = argparse.ArgumentParser("Check CycleGAN dataset folders")
    parser.add_argument("--dataroot", required=True)
    args = parser.parse_args()
    root = Path(args.dataroot)
    missing = []
    for split in ["trainA", "trainB", "testA", "testB"]:
        directory = root / split
        paths = list_images(directory)
        if not directory.exists():
            missing.append(str(directory))
        print(f"{split}: {len(paths)} images")
        if paths:
            with Image.open(paths[0]) as image:
                print(f"  sample: {paths[0].name}, size={image.size}")
    if missing:
        print("Missing directories:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
