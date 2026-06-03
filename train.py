import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import UnpairedImageDataset
from engine import CycleGANTrainer
from engine.trainer import LOSS_NAMES
from utils.config import dataloader_kwargs, load_and_resolve_config
from utils.count_params import count_parameters
from utils.logger import log_environment, setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser("Train CycleGAN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", default=None)
    parser.add_argument("--dry_run_data", action="store_true")
    return parser.parse_args()


def append_losses(path, epoch, lr, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr"] + LOSS_NAMES)
        if write_header:
            writer.writeheader()
        row = {"epoch": epoch, "lr": lr}
        row.update({name: losses.get(name, 0.0) for name in LOSS_NAMES})
        writer.writerow(row)


def average_losses(sums, steps):
    return {name: value / max(1, steps) for name, value in sums.items()}


def main():
    args = parse_args()
    overrides = {
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_amp": args.use_amp,
    }
    cfg = load_and_resolve_config(args.config, args.name, overrides)
    set_seed(cfg["seed"], deterministic=cfg["cudnn_deterministic"], benchmark=cfg["cudnn_benchmark"])

    logger = setup_logger(Path(cfg["results_dir"]) / "log.txt")
    logger.info("Training started at %s", datetime.now().isoformat(timespec="seconds"))
    log_environment(logger, cfg)

    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")
    dataset = UnpairedImageDataset(cfg["dataroot"], phase="train", load_size=cfg["load_size"], image_size=cfg["image_size"])
    if args.dry_run_data:
        logger.info("Dry run dataset size: %d", len(dataset))
        sample = dataset[0]
        logger.info("Sample tensor shapes: A=%s B=%s", tuple(sample["A"].shape), tuple(sample["B"].shape))
        return

    loader = DataLoader(dataset, **dataloader_kwargs(cfg, shuffle=True, drop_last=True))
    trainer = CycleGANTrainer(cfg, device)
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        logger.info("Resumed from %s, start_epoch=%d", args.resume, start_epoch)

    logger.info("Experiment: %s", cfg["name"])
    logger.info("Model: %s, n_blocks=%d", cfg["model_type"], cfg["n_blocks"])
    logger.info("Dataset: %s, images_per_epoch=%d", cfg["dataroot"], len(dataset))
    logger.info("G params: %s", count_parameters(trainer.G_A))
    logger.info("D params: %s", count_parameters(trainer.D_A))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        trainer.train()
        sums = {name: 0.0 for name in LOSS_NAMES}
        steps = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for batch in pbar:
            trainer.set_input(batch)
            trainer.optimize_parameters()
            losses = trainer.get_current_losses()
            for name in LOSS_NAMES:
                sums[name] += losses.get(name, 0.0)
            steps += 1
            if steps % cfg["log_freq"] == 0:
                pbar.set_postfix({"G": f"{losses['G_total']:.3f}", "D_A": f"{losses['D_A']:.3f}", "D_B": f"{losses['D_B']:.3f}"})
            if steps % cfg["sample_freq"] == 0:
                trainer.save_samples(Path(cfg["results_dir"]) / "samples" / f"epoch_{epoch:03d}_step_{steps:06d}.png")

        avg = average_losses(sums, steps)
        current_lr = trainer.optimizer_G.param_groups[0]["lr"]
        append_losses(Path(cfg["results_dir"]) / "losses.csv", epoch, current_lr, avg)
        trainer.save_samples(Path(cfg["results_dir"]) / "samples" / f"epoch_{epoch:03d}.png")
        trainer.save_checkpoint(epoch)
        trainer.update_learning_rate()
        logger.info("epoch=%d lr=%.8f losses=%s", epoch, current_lr, avg)

    logger.info("Training ended at %s", datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()

