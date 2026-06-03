import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import UnpairedImageDataset
from models import GANLoss, ImagePool, LightweightGenerator, NLayerDiscriminator, ResnetGenerator
from utils.count_params import count_parameters
from utils.logger import setup_logger
from utils.visualizer import save_train_grid


LOSS_COLUMNS = ["G_A", "G_B", "cycle_A", "cycle_B", "idt_A", "idt_B", "D_A", "D_B", "G_total"]


def parse_args():
    parser = argparse.ArgumentParser("Train CycleGAN")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path, usually checkpoints/name/latest.pth")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(net):
    """CycleGAN normal initialization."""

    def init_func(module):
        name = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in name or "Linear" in name):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


def build_generator(model_type, n_blocks):
    if model_type == "baseline":
        return ResnetGenerator(n_blocks=9 if n_blocks is None else n_blocks)
    if model_type == "lightweight":
        return LightweightGenerator(n_blocks=6 if n_blocks is None else n_blocks)
    raise ValueError("model_type must be baseline or lightweight")


def set_requires_grad(nets, requires_grad):
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def lr_lambda(epoch, epochs, decay_start_epoch):
    if epoch < decay_start_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start_epoch) / float(epochs - decay_start_epoch))


def save_checkpoint(path, epoch, cfg, nets, opts):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "config": cfg,
            "G_A": nets["G_A"].state_dict(),
            "G_B": nets["G_B"].state_dict(),
            "D_A": nets["D_A"].state_dict(),
            "D_B": nets["D_B"].state_dict(),
            "opt_G": opts["G"].state_dict(),
            "opt_D": opts["D"].state_dict(),
        },
        path,
    )


def save_generator_weights(checkpoint_dir, nets):
    torch.save(nets["G_A"].state_dict(), checkpoint_dir / "latest_G_A.pth")
    torch.save(nets["G_B"].state_dict(), checkpoint_dir / "latest_G_B.pth")


def append_losses(path, epoch, lr, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr"] + LOSS_COLUMNS)
        if write_header:
            writer.writeheader()
        row = {"epoch": epoch, "lr": lr}
        row.update(losses)
        writer.writerow(row)


def load_resume(path, nets, opts, device):
    state = torch.load(path, map_location=device)
    for key in ["G_A", "G_B", "D_A", "D_B"]:
        nets[key].load_state_dict(state[key])
    opts["G"].load_state_dict(state["opt_G"])
    opts["D"].load_state_dict(state["opt_D"])
    return int(state.get("epoch", 0))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    results_dir = Path("results") / args.name
    checkpoint_dir = Path("checkpoints") / args.name
    sample_dir = results_dir / "samples"
    logger = setup_logger(results_dir / "log.txt")

    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_type = cfg["model_type"]
    n_blocks = cfg.get("n_blocks", 9 if model_type == "baseline" else 6)
    nets = {
        "G_A": build_generator(model_type, n_blocks).to(device),
        "G_B": build_generator(model_type, n_blocks).to(device),
        "D_A": NLayerDiscriminator().to(device),
        "D_B": NLayerDiscriminator().to(device),
    }
    for net in nets.values():
        init_weights(net)

    dataset = UnpairedImageDataset(
        cfg["dataroot"],
        phase="train",
        load_size=cfg.get("load_size", 286),
        image_size=cfg.get("image_size", 256),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    criterion_gan = GANLoss("lsgan").to(device)
    criterion_l1 = nn.L1Loss()
    opt_G = torch.optim.Adam(
        list(nets["G_A"].parameters()) + list(nets["G_B"].parameters()),
        lr=cfg.get("lr", 0.0002),
        betas=(cfg.get("beta1", 0.5), cfg.get("beta2", 0.999)),
    )
    opt_D = torch.optim.Adam(
        list(nets["D_A"].parameters()) + list(nets["D_B"].parameters()),
        lr=cfg.get("lr", 0.0002),
        betas=(cfg.get("beta1", 0.5), cfg.get("beta2", 0.999)),
    )
    sched_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=lambda e: lr_lambda(e, cfg.get("epochs", 200), cfg.get("decay_start_epoch", 100))
    )
    sched_D = torch.optim.lr_scheduler.LambdaLR(
        opt_D, lr_lambda=lambda e: lr_lambda(e, cfg.get("epochs", 200), cfg.get("decay_start_epoch", 100))
    )
    opts = {"G": opt_G, "D": opt_D}

    fake_A_pool = ImagePool(cfg.get("pool_size", 50))
    fake_B_pool = ImagePool(cfg.get("pool_size", 50))
    start_epoch = 1
    if args.resume:
        loaded_epoch = load_resume(args.resume, nets, opts, device)
        start_epoch = loaded_epoch + 1
        for _ in range(loaded_epoch):
            sched_G.step()
            sched_D.step()
        logger.info("Resumed from %s at epoch %d", args.resume, loaded_epoch)

    logger.info("Experiment: %s", args.name)
    logger.info("Model type: %s, n_blocks=%d", model_type, n_blocks)
    logger.info("Dataset: %s", cfg["dataroot"])
    logger.info("Device: %s, GPU: %s", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    logger.info("G params: %s", count_parameters(nets["G_A"]))
    logger.info("D params: %s", count_parameters(nets["D_A"]))
    logger.info("Hyperparameters: %s", cfg)

    lambda_cycle = cfg.get("lambda_cycle", 10)
    lambda_identity = cfg.get("lambda_identity", 5)
    use_cycle = cfg.get("use_cycle", True)
    use_identity = cfg.get("use_identity", True)

    for epoch in range(start_epoch, cfg.get("epochs", 200) + 1):
        sums = {key: 0.0 for key in LOSS_COLUMNS}
        steps = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.get('epochs', 200)}")
        for batch in pbar:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = nets["G_A"](real_A)
            rec_A = nets["G_B"](fake_B)
            fake_A = nets["G_B"](real_B)
            rec_B = nets["G_A"](fake_A)

            set_requires_grad([nets["D_A"], nets["D_B"]], False)
            opt_G.zero_grad(set_to_none=True)
            loss_G_A = criterion_gan(nets["D_A"](fake_B), True)
            loss_G_B = criterion_gan(nets["D_B"](fake_A), True)
            loss_cycle_A = criterion_l1(rec_A, real_A) * lambda_cycle if use_cycle else torch.tensor(0.0, device=device)
            loss_cycle_B = criterion_l1(rec_B, real_B) * lambda_cycle if use_cycle else torch.tensor(0.0, device=device)
            if use_identity:
                loss_idt_A = criterion_l1(nets["G_A"](real_B), real_B) * lambda_identity
                loss_idt_B = criterion_l1(nets["G_B"](real_A), real_A) * lambda_identity
            else:
                loss_idt_A = torch.tensor(0.0, device=device)
                loss_idt_B = torch.tensor(0.0, device=device)
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            opt_G.step()

            set_requires_grad([nets["D_A"], nets["D_B"]], True)
            opt_D.zero_grad(set_to_none=True)
            loss_D_A = 0.5 * (
                criterion_gan(nets["D_A"](real_B), True)
                + criterion_gan(nets["D_A"](fake_B_pool.query(fake_B).detach()), False)
            )
            loss_D_B = 0.5 * (
                criterion_gan(nets["D_B"](real_A), True)
                + criterion_gan(nets["D_B"](fake_A_pool.query(fake_A).detach()), False)
            )
            (loss_D_A + loss_D_B).backward()
            opt_D.step()

            values = {
                "G_A": loss_G_A.item(),
                "G_B": loss_G_B.item(),
                "cycle_A": loss_cycle_A.item(),
                "cycle_B": loss_cycle_B.item(),
                "idt_A": loss_idt_A.item(),
                "idt_B": loss_idt_B.item(),
                "D_A": loss_D_A.item(),
                "D_B": loss_D_B.item(),
                "G_total": loss_G.item(),
            }
            for key, value in values.items():
                sums[key] += value
            steps += 1
            if steps % cfg.get("sample_freq", 500) == 0:
                save_train_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, sample_dir / f"epoch_{epoch:03d}_step_{steps:06d}.png")
            pbar.set_postfix({"G": f"{loss_G.item():.3f}", "D_A": f"{loss_D_A.item():.3f}", "D_B": f"{loss_D_B.item():.3f}"})

        avg_losses = {key: value / max(1, steps) for key, value in sums.items()}
        save_train_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B, sample_dir / f"epoch_{epoch:03d}.png")
        current_lr = opt_G.param_groups[0]["lr"]
        append_losses(results_dir / "losses.csv", epoch, current_lr, avg_losses)
        save_checkpoint(checkpoint_dir / "latest.pth", epoch, cfg, nets, opts)
        save_generator_weights(checkpoint_dir, nets)
        if epoch % cfg.get("save_epoch_freq", 10) == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pth", epoch, cfg, nets, opts)
        logger.info("epoch=%d lr=%.8f losses=%s", epoch, current_lr, avg_losses)
        sched_G.step()
        sched_D.step()


if __name__ == "__main__":
    main()

