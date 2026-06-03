import functools

import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator returning a 2D patch score map."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True
        kw = 4
        padw = 1

        # First layer follows CycleGAN PatchGAN and has no InstanceNorm.
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

