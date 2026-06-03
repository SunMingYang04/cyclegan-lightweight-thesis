import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_transform


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


class UnpairedImageDataset(Dataset):
    """Unpaired A/B domain dataset for CycleGAN."""

    def __init__(self, dataroot, phase="train", load_size=286, image_size=256):
        self.dataroot = Path(dataroot)
        self.phase = phase
        self.dir_A = self.dataroot / f"{phase}A"
        self.dir_B = self.dataroot / f"{phase}B"
        if not self.dir_A.exists() or not self.dir_B.exists():
            raise FileNotFoundError(f"Expected {self.dir_A} and {self.dir_B}")

        self.paths_A = list_images(self.dir_A)
        self.paths_B = list_images(self.dir_B)
        if not self.paths_A or not self.paths_B:
            raise RuntimeError(f"No images found under {self.dir_A} or {self.dir_B}")
        self.transform = get_transform(load_size, image_size, phase)

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, index):
        path_A = self.paths_A[index % len(self.paths_A)]
        if self.phase == "train":
            path_B = random.choice(self.paths_B)
        else:
            path_B = self.paths_B[index % len(self.paths_B)]

        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")
        return {
            "A": self.transform(image_A),
            "B": self.transform(image_B),
            "A_path": str(path_A),
            "B_path": str(path_B),
        }
