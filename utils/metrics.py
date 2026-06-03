import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(directory, max_images=None):
    paths = sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def compute_fid_kid(real_dir, fake_dir, max_images=None):
    """Compute FID and KID with clean-fid."""
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError("clean-fid is required for FID/KID. Install with: pip install clean-fid") from exc

    real_dir = str(Path(real_dir))
    fake_dir = str(Path(fake_dir))
    if max_images is not None:
        warnings.warn("clean-fid directory API uses all images; max_images is ignored for FID/KID.")
    return {"FID": round(float(fid.compute_fid(real_dir, fake_dir)), 6), "KID": round(float(fid.compute_kid(real_dir, fake_dir)), 6)}


def _load_pair_tensors(input_dir, fake_dir, image_size=256, max_images=None):
    import torch
    from torchvision import transforms

    input_paths = image_paths(input_dir, max_images=max_images)
    fake_paths = image_paths(fake_dir, max_images=max_images)
    if len(input_paths) != len(fake_paths):
        warnings.warn(f"input_dir has {len(input_paths)} images but fake_dir has {len(fake_paths)}; using the minimum count.")
    by_name = {p.name: p for p in fake_paths}
    pairs = [(p, by_name[p.name]) for p in input_paths if p.name in by_name]
    if not pairs:
        pairs = list(zip(input_paths, fake_paths))
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    for input_path, fake_path in pairs:
        real = transform(Image.open(input_path).convert("RGB")).unsqueeze(0)
        fake = transform(Image.open(fake_path).convert("RGB")).unsqueeze(0)
        yield real, fake


def compute_ssim_lpips(input_dir, fake_dir, max_images=None):
    """Compute sorted-pair SSIM and LPIPS. Returns null metrics if input_dir is missing."""
    if input_dir is None:
        return {"SSIM": None, "LPIPS": None}

    import torch
    from skimage.metrics import structural_similarity
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    ssim_values = []
    lpips_values = []
    with torch.no_grad():
        for real, fake in _load_pair_tensors(input_dir, fake_dir, max_images=max_images):
            real_np = np.transpose(real.squeeze(0).numpy(), (1, 2, 0))
            fake_np = np.transpose(fake.squeeze(0).numpy(), (1, 2, 0))
            ssim_values.append(structural_similarity(real_np, fake_np, channel_axis=2, data_range=1.0))
            real_l = real.to(device) * 2 - 1
            fake_l = fake.to(device) * 2 - 1
            lpips_values.append(float(lpips_model(real_l, fake_l).item()))

    if not ssim_values:
        return {"SSIM": None, "LPIPS": None}
    return {
        "SSIM": round(float(np.mean(ssim_values)), 6),
        "LPIPS": round(float(np.mean(lpips_values)), 6),
    }


def compute_cmmd(real_dir, fake_dir):
    """Reserved CMMD interface. Returns null unless a project-specific implementation is added."""
    _ = real_dir, fake_dir
    print("CMMD is not implemented in this project; returning null.")
    return {"CMMD": None}


def save_metrics(metrics, output_path):
    """Save metrics as JSON and adjacent CSV."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    return output_path, csv_path
