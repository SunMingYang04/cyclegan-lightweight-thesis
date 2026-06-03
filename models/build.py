import torch.nn as nn

from .discriminators import NLayerDiscriminator
from .generators import LightweightGenerator, ResnetGenerator


def init_weights(net, init_gain=0.02):
    """Apply CycleGAN normal initialization consistently."""

    def init_func(module):
        name = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in name or "Linear" in name):
            nn.init.normal_(module.weight.data, 0.0, init_gain)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    net.apply(init_func)
    return net


def build_generator(config):
    """Build generator according to config.model_type without changing thesis semantics."""
    model_type = config["model_type"]
    n_blocks = config.get("n_blocks")
    if model_type == "baseline":
        return init_weights(ResnetGenerator(n_blocks=9))
    if model_type == "lightweight":
        return init_weights(LightweightGenerator(n_blocks=6))
    if model_type == "dwconv9":
        return init_weights(LightweightGenerator(n_blocks=9))
    if model_type == "custom":
        return init_weights(LightweightGenerator(n_blocks=n_blocks))
    raise ValueError("model_type must be one of: baseline, lightweight, dwconv9, custom")


def build_discriminator(config):
    """Build the unchanged 70x70 PatchGAN discriminator."""
    _ = config
    return init_weights(NLayerDiscriminator())


def build_models(config, device=None):
    """Build G_A, G_B, D_A, D_B."""
    models = {
        "G_A": build_generator(config),
        "G_B": build_generator(config),
        "D_A": build_discriminator(config),
        "D_B": build_discriminator(config),
    }
    if device is not None:
        models = {name: model.to(device) for name, model in models.items()}
    return models

