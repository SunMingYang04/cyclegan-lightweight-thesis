from pathlib import Path

import torch
import torch.nn as nn

from engine.checkpoint import load_checkpoint as load_training_checkpoint
from engine.checkpoint import save_checkpoint as save_training_checkpoint
from engine.checkpoint import save_generator_weights
from engine.schedulers import linear_decay_lambda
from models import GANLoss, ImagePool
from models.build import build_models
from utils.visualizer import save_train_grid


LOSS_NAMES = ["G_A", "G_B", "cycle_A", "cycle_B", "idt_A", "idt_B", "D_A", "D_B", "G_total"]


class CycleGANTrainer:
    """CycleGAN training engine for baseline, lightweight, and ablation runs."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        models = build_models(config, device)
        self.G_A = models["G_A"]
        self.G_B = models["G_B"]
        self.D_A = models["D_A"]
        self.D_B = models["D_B"]

        self.fake_A_pool = ImagePool(config["pool_size"])
        self.fake_B_pool = ImagePool(config["pool_size"])
        self.criterion_gan = GANLoss("lsgan").to(device)
        self.criterion_l1 = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(
            list(self.G_A.parameters()) + list(self.G_B.parameters()),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
        )
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=lambda e: linear_decay_lambda(e, config["epochs"], config["decay_start_epoch"]),
        )
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D,
            lr_lambda=lambda e: linear_decay_lambda(e, config["epochs"], config["decay_start_epoch"]),
        )
        self.use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.losses = {name: 0.0 for name in LOSS_NAMES}

    def set_input(self, batch):
        """Move batch tensors to device."""
        non_blocking = self.device.type == "cuda"
        self.real_A = batch["A"].to(self.device, non_blocking=non_blocking)
        self.real_B = batch["B"].to(self.device, non_blocking=non_blocking)

    def forward(self):
        """Run both translation directions and cycle reconstructions."""
        self.fake_B = self.G_A(self.real_A)
        self.rec_A = self.G_B(self.fake_B)
        self.fake_A = self.G_B(self.real_B)
        self.rec_B = self.G_A(self.fake_A)

    @staticmethod
    def set_requires_grad(nets, requires_grad):
        if not isinstance(nets, (list, tuple)):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def backward_G(self):
        """Compute generator losses and gradients."""
        lambda_cycle = self.config["lambda_cycle"]
        lambda_identity = self.config["lambda_identity"]
        loss_G_A = self.criterion_gan(self.D_A(self.fake_B), True)
        loss_G_B = self.criterion_gan(self.D_B(self.fake_A), True)
        loss_cycle_A = (
            self.criterion_l1(self.rec_A, self.real_A) * lambda_cycle
            if self.config["use_cycle"]
            else torch.tensor(0.0, device=self.device)
        )
        loss_cycle_B = (
            self.criterion_l1(self.rec_B, self.real_B) * lambda_cycle
            if self.config["use_cycle"]
            else torch.tensor(0.0, device=self.device)
        )
        if self.config["use_identity"]:
            loss_idt_A = self.criterion_l1(self.G_A(self.real_B), self.real_B) * lambda_identity
            loss_idt_B = self.criterion_l1(self.G_B(self.real_A), self.real_A) * lambda_identity
        else:
            loss_idt_A = torch.tensor(0.0, device=self.device)
            loss_idt_B = torch.tensor(0.0, device=self.device)
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        self.loss_G = loss_G
        self.losses.update(
            {
                "G_A": loss_G_A.item(),
                "G_B": loss_G_B.item(),
                "cycle_A": loss_cycle_A.item(),
                "cycle_B": loss_cycle_B.item(),
                "idt_A": loss_idt_A.item(),
                "idt_B": loss_idt_B.item(),
                "G_total": loss_G.item(),
            }
        )
        return loss_G

    def backward_D_basic(self, netD, real, fake):
        """Compute one discriminator loss."""
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        return 0.5 * (self.criterion_gan(pred_real, True) + self.criterion_gan(pred_fake, False))

    def backward_D(self):
        """Compute discriminator losses and gradients."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)
        loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)
        self.loss_D = loss_D_A + loss_D_B
        self.losses.update({"D_A": loss_D_A.item(), "D_B": loss_D_B.item()})
        return self.loss_D

    def optimize_parameters(self):
        """Run one optimization step."""
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad(set_to_none=True)
        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                self.forward()
                loss_G = self.backward_G()
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.optimizer_G)
        else:
            self.forward()
            loss_G = self.backward_G()
            loss_G.backward()
            self.optimizer_G.step()

        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad(set_to_none=True)
        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                loss_D = self.backward_D()
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.optimizer_D)
            self.scaler.update()
        else:
            loss_D = self.backward_D()
            loss_D.backward()
            self.optimizer_D.step()

    def save_samples(self, path):
        """Save current real/fake/reconstruction image grid."""
        save_train_grid(self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B, path)

    def save_checkpoint(self, epoch):
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        save_training_checkpoint(checkpoint_dir / "latest.pth", self, epoch, self.config)
        save_generator_weights(checkpoint_dir, self)
        if epoch % self.config["save_epoch_freq"] == 0:
            save_training_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pth", self, epoch, self.config)

    def load_checkpoint(self, path):
        return load_training_checkpoint(path, self, self.device)

    def update_learning_rate(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def get_current_losses(self):
        return dict(self.losses)

    def train(self):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

    def eval(self):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()
