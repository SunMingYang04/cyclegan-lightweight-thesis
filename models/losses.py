import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """GAN loss wrapper. This project uses LSGAN for CycleGAN training."""

    def __init__(self, gan_mode="lsgan", real_label=1.0, fake_label=0.0):
        super().__init__()
        if gan_mode != "lsgan":
            raise ValueError("Only lsgan is implemented for this reproduction.")
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        return self.loss(prediction, self.get_target_tensor(prediction, target_is_real))


def cycle_consistency_loss(recovered, real):
    return nn.functional.l1_loss(recovered, real)


def identity_loss(identity, real):
    return nn.functional.l1_loss(identity, real)

