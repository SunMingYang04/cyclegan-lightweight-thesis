from pathlib import Path

import torch
from torchvision.utils import save_image


def denormalize(tensor):
    return (tensor.detach().cpu() + 1.0) / 2.0


def save_image_tensor(tensor, path):
    """Save one normalized tensor image to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(tensor), path)


def save_train_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, path):
    """Save a 2x3 grid: real_A/fake_B/rec_A and real_B/fake_A/rec_B."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([real_A[:1], fake_B[:1], rec_A[:1], real_B[:1], fake_A[:1], rec_B[:1]], dim=0)
    save_image(denormalize(grid), path, nrow=3)

