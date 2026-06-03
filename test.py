import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.transforms import get_transform
from models import LightweightGenerator, ResnetGenerator
from utils.visualizer import save_image_tensor


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class SingleDomainDataset(Dataset):
    def __init__(self, image_dir, load_size, image_size):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing directory: {self.image_dir}")
        self.paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        self.transform = get_transform(load_size, image_size, phase="test")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return {"image": self.transform(Image.open(path).convert("RGB")), "path": str(path)}


def parse_args():
    parser = argparse.ArgumentParser("Test one CycleGAN generator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--direction", choices=["AtoB", "BtoA"], required=True)
    parser.add_argument("--num_test", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_generator(cfg):
    model_type = cfg["model_type"]
    if model_type == "baseline":
        return ResnetGenerator(n_blocks=cfg.get("n_blocks", 9))
    if model_type == "lightweight":
        return LightweightGenerator(n_blocks=cfg.get("n_blocks", 6))
    raise ValueError("model_type must be baseline or lightweight")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device_name = args.device or cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    generator = build_generator(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state["G_A"] if isinstance(state, dict) and "G_A" in state else state)
    generator.eval()

    source = "testA" if args.direction == "AtoB" else "testB"
    out_name = "test_AtoB" if args.direction == "AtoB" else "test_BtoA"
    dataset = SingleDomainDataset(Path(cfg["dataroot"]) / source, cfg.get("load_size", 286), cfg.get("image_size", 256))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 4))
    output_dir = Path("results") / args.name / out_name
    input_dir = output_dir / "input"
    fake_dir = output_dir / "fake"

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"test {args.direction}")):
            if args.num_test is not None and idx >= args.num_test:
                break
            image = batch["image"].to(device)
            fake = generator(image)
            stem = Path(batch["path"][0]).stem
            save_image_tensor(image, input_dir / f"{stem}.png")
            save_image_tensor(fake, fake_dir / f"{stem}.png")
    print(f"Saved test results to {output_dir}")


if __name__ == "__main__":
    main()

