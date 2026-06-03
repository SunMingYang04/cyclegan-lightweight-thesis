import functools

import torch.nn as nn


def get_norm_layer():
    return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


def use_bias_from_norm(norm_layer):
    return isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm2d


class ResnetBlock(nn.Module):
    """Standard residual block used in the original CycleGAN generator."""

    def __init__(self, dim):
        super().__init__()
        norm_layer = get_norm_layer()
        use_bias = use_bias_from_norm(norm_layer)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 2D convolution followed by 1x1 pointwise convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableDeconv(nn.Module):
    """Lightweight upsampling block: nearest upsample + depthwise separable conv."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, x):
        return self.block(x)


class LightweightResnetBlock(nn.Module):
    """Residual block with depthwise separable convolutions."""

    def __init__(self, dim):
        super().__init__()
        norm_layer = get_norm_layer()
        use_bias = use_bias_from_norm(norm_layer)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            DepthwiseSeparableConv(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            DepthwiseSeparableConv(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Standard CycleGAN ResNet generator for 256x256 RGB images."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        norm_layer = get_norm_layer()
        use_bias = use_bias_from_norm(norm_layer)

        # Encoder: c7s1-64, d128, d256.
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        ]

        # Residual transformer: 9 blocks for the baseline 256x256 setting.
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf * 4))

        # Decoder: u128, u64, output c7s1-3.
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LightweightGenerator(nn.Module):
    """Lightweight CycleGAN generator with depthwise separable convolutions."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super().__init__()
        norm_layer = get_norm_layer()
        use_bias = use_bias_from_norm(norm_layer)

        # Encoder in the thesis lightweight model.
        layers = [
            nn.ReflectionPad2d(3),
            DepthwiseSeparableConv(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            DepthwiseSeparableConv(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            DepthwiseSeparableConv(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        ]

        # Residual transformer: reduced from 9 to 6 blocks.
        for _ in range(n_blocks):
            layers.append(LightweightResnetBlock(ngf * 4))

        # Decoder: lightweight upsampling blocks preserve output range [-1, 1].
        layers += [
            DepthwiseSeparableDeconv(ngf * 4, ngf * 2, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            DepthwiseSeparableDeconv(ngf * 2, ngf, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            DepthwiseSeparableConv(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

