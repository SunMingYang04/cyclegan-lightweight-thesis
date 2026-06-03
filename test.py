import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.transforms import get_transform
from models.build import build_generator
from utils.config import dataloader_kwargs, load_and_resolve_config
from utils.seed import set_seed
from utils.visualizer import save_image_tensor


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SingleDomainDataset(Dataset):
    """Dataset for one test domain."""

    def __init__(self, image_dir, load_size, image_size):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing test directory: {self.image_dir}")
        self.paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        if not self.paths:
            raise RuntimeError(f"No test images found in {self.image_dir}")
        self.transform = get_transform(load_size, image_size, phase="test")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return {"image": self.transform(Image.open(path).convert("RGB")), "path": str(path)}


def parse_args():
    parser = argparse.ArgumentParser("Test CycleGAN generators")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True, help="Generator checkpoint, usually latest_G_A.pth.")
    parser.add_argument("--direction", choices=["AtoB", "BtoA", "both"], required=True)
    parser.add_argument("--num_test", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_generator(cfg, checkpoint, device, state_key):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    generator = build_generator(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and state_key in state:
        state = state[state_key]
    generator.load_state_dict(state)
    generator.eval()
    return generator


def paired_checkpoint(checkpoint, direction):
    checkpoint = Path(checkpoint)
    if direction == "AtoB":
        return checkpoint
    if direction == "BtoA":
        return Path(str(checkpoint).replace("latest_G_A.pth", "latest_G_B.pth"))
    raise ValueError(direction)


@torch.no_grad()
def run_direction(cfg, name, direction, checkpoint, device, num_test=None):
    source = "testA" if direction == "AtoB" else "testB"
    output_name = "test_AtoB" if direction == "AtoB" else "test_BtoA"
    state_key = "G_A" if direction == "AtoB" else "G_B"
    generator = load_generator(cfg, checkpoint, device, state_key)
    dataset = SingleDomainDataset(Path(cfg["dataroot"]) / source, cfg["load_size"], cfg["image_size"])
    loader_cfg = dict(cfg)
    loader_cfg["batch_size"] = 1
    loader = DataLoader(dataset, **dataloader_kwargs(loader_cfg, shuffle=False, drop_last=False))
    output_dir = Path("results") / name / output_name
    input_dir = output_dir / "input"
    fake_dir = output_dir / "fake"

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    count = 0
    for idx, batch in enumerate(tqdm(loader, desc=f"test {direction}")):
        if num_test is not None and idx >= num_test:
            break
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        fake = generator(image)
        stem = Path(batch["path"][0]).stem
        save_image_tensor(image, input_dir / f"{stem}.png")
        save_image_tensor(fake, fake_dir / f"{stem}.png")
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg = elapsed / max(1, count)
    print(f"{direction}: saved {count} images to {output_dir}")
    print(f"{direction}: total_time={elapsed:.3f}s avg_time={avg * 1000:.3f}ms/image")


def main():
    args = parse_args()
    cfg = load_and_resolve_config(args.config, args.name, {"device": args.device}, save=False)
    set_seed(cfg["seed"], cfg["cudnn_deterministic"], cfg["cudnn_benchmark"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")
    directions = ["AtoB", "BtoA"] if args.direction == "both" else [args.direction]
    for direction in directions:
        run_direction(cfg, args.name, direction, paired_checkpoint(args.checkpoint, direction), device, args.num_test)


if __name__ == "__main__":
    main()
