from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _domain_map(domain_ids: torch.Tensor, num_domains: int, spatial_shape) -> torch.Tensor:
    one_hot = F.one_hot(domain_ids, num_classes=num_domains).float()
    shape = (domain_ids.shape[0], num_domains, *([1] * len(spatial_shape)))
    maps = one_hot.view(shape)
    return maps.expand(domain_ids.shape[0], num_domains, *spatial_shape)


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, norm=True):
        super().__init__()
        padding = 1 if kernel_size == 4 and stride == 2 else kernel_size // 2
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)]
        if norm:
            layers.append(_group_norm(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2, inplace=True)
        x = self.norm2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2, inplace=True)


class AdaINResBlock3d(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(channels, affine=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=False)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.style1 = nn.Linear(style_dim, channels * 2)
        self.style2 = nn.Linear(style_dim, channels * 2)

    @staticmethod
    def _adain(x, style_params):
        gamma, beta = style_params.chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1], 1, 1, 1)
        return x * (1.0 + gamma.view(shape)) + beta.view(shape)

    def forward(self, x, style):
        residual = x
        h = self._adain(self.norm1(x), self.style1(style))
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv1(h)
        h = self._adain(self.norm2(h), self.style2(style))
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)
        return h + residual


class ContentEncoder3d(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, levels=2, res_blocks=4, max_channels=256):
        super().__init__()
        channels = base_channels
        layers = [ConvBlock3d(in_channels, channels, kernel_size=7, stride=1)]
        for _ in range(levels):
            next_channels = min(channels * 2, max_channels)
            layers.append(ConvBlock3d(channels, next_channels, kernel_size=4, stride=2))
            channels = next_channels
        layers.extend(ResBlock3d(channels) for _ in range(res_blocks))
        self.net = nn.Sequential(*layers)
        self.out_channels = channels
        self.levels = levels

    def forward(self, x):
        return self.net(x)


class StyleEncoder3d(nn.Module):
    def __init__(self, num_domains, in_channels=1, base_channels=16, style_dim=64, max_channels=256):
        super().__init__()
        channels = base_channels
        layers = [ConvBlock3d(in_channels, channels, kernel_size=4, stride=2, norm=False)]
        for _ in range(4):
            next_channels = min(channels * 2, max_channels)
            layers.append(ConvBlock3d(channels, next_channels, kernel_size=4, stride=2))
            channels = next_channels
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(channels, style_dim) for _ in range(num_domains)])

    def forward(self, x, domain_ids):
        h = self.trunk(x)
        h = h.mean(dim=(2, 3, 4))
        styles = torch.stack([head(h) for head in self.heads], dim=1)
        gather_index = domain_ids.view(-1, 1, 1).expand(-1, 1, styles.shape[-1])
        return styles.gather(1, gather_index).squeeze(1)


class StyleGenerator(nn.Module):
    def __init__(self, num_domains, latent_dim=16, style_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_domains, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, style_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, z, domain_ids):
        domain = self.embedding(domain_ids)
        return self.net(torch.cat([z, domain], dim=1))


class Generator3d(nn.Module):
    def __init__(
        self,
        content_channels,
        style_dim=64,
        out_channels=1,
        levels=2,
        res_blocks=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([AdaINResBlock3d(content_channels, style_dim) for _ in range(res_blocks)])
        channels = content_channels
        up_layers = []
        for _ in range(levels):
            next_channels = max(channels // 2, 16)
            up_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    ConvBlock3d(channels, next_channels, kernel_size=3, stride=1),
                ]
            )
            channels = next_channels
        up_layers.append(nn.Conv3d(channels, out_channels, 7, padding=3))
        self.up = nn.Sequential(*up_layers)

    def forward(self, content, style):
        x = content
        for block in self.blocks:
            x = block(x, style)
        return torch.tanh(self.up(x))


class Discriminator3d(nn.Module):
    def __init__(self, num_domains, in_channels=1, base_channels=16, max_channels=256):
        super().__init__()
        channels = base_channels
        layers = [nn.Conv3d(in_channels + num_domains, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for _ in range(4):
            next_channels = min(channels * 2, max_channels)
            layers.extend(
                [
                    nn.Conv3d(channels, next_channels, 4, stride=2, padding=1),
                    _group_norm(next_channels),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            channels = next_channels
        layers.append(nn.Conv3d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)
        self.num_domains = num_domains

    def forward(self, x, domain_ids):
        domains = _domain_map(domain_ids, self.num_domains, x.shape[2:]).to(dtype=x.dtype, device=x.device)
        return self.net(torch.cat([x, domains], dim=1))


class MURD3D(nn.Module):
    def __init__(
        self,
        num_domains,
        in_channels=1,
        base_channels=16,
        levels=2,
        res_blocks=4,
        style_dim=64,
        latent_dim=16,
        max_channels=256,
    ):
        super().__init__()
        self.content_encoder = ContentEncoder3d(
            in_channels=in_channels,
            base_channels=base_channels,
            levels=levels,
            res_blocks=res_blocks,
            max_channels=max_channels,
        )
        self.style_encoder = StyleEncoder3d(
            num_domains=num_domains,
            in_channels=in_channels,
            base_channels=base_channels,
            style_dim=style_dim,
            max_channels=max_channels,
        )
        self.style_generator = StyleGenerator(num_domains=num_domains, latent_dim=latent_dim, style_dim=style_dim)
        self.generator = Generator3d(
            content_channels=self.content_encoder.out_channels,
            style_dim=style_dim,
            out_channels=in_channels,
            levels=levels,
            res_blocks=res_blocks,
        )
        self.latent_dim = latent_dim

    def encode_content(self, x):
        return self.content_encoder(x)

    def encode_style(self, x, domain_ids):
        return self.style_encoder(x, domain_ids)

    def random_style(self, batch_size, domain_ids, device=None, z=None):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim, device=device or domain_ids.device)
        return self.style_generator(z, domain_ids)

    def decode(self, content, style):
        return self.generator(content, style)

    def translate(self, x, target_domain_ids, z=None):
        content = self.encode_content(x)
        style = self.random_style(x.shape[0], target_domain_ids, device=x.device, z=z)
        return self.decode(content, style), content, style

    def train_generator_outputs(self, x_src, src_domain_ids, tgt_domain_ids, z2=None):
        content_src = self.encode_content(x_src)
        style_src = self.encode_style(x_src, src_domain_ids)
        style_tgt = self.random_style(x_src.shape[0], tgt_domain_ids, device=x_src.device)
        fake_tgt = self.decode(content_src, style_tgt)

        content_fake = self.encode_content(fake_tgt)
        rec_src = self.decode(content_fake, style_src)
        id_src = self.decode(content_src, style_src)
        encoded_fake_style = self.encode_style(fake_tgt, tgt_domain_ids)

        if z2 is None:
            z2 = torch.randn(x_src.shape[0], self.latent_dim, device=x_src.device)
        style_tgt_2 = self.random_style(x_src.shape[0], tgt_domain_ids, device=x_src.device, z=z2)
        fake_tgt_2 = self.decode(content_src, style_tgt_2)

        return (
            fake_tgt,
            rec_src,
            id_src,
            content_src,
            content_fake,
            style_tgt,
            encoded_fake_style,
            fake_tgt_2,
        )

    def forward(self, mode, *args, **kwargs):
        if mode == "train_g":
            return self.train_generator_outputs(*args, **kwargs)
        if mode == "translate":
            return self.translate(*args, **kwargs)
        raise ValueError(f"Unknown MURD3D forward mode: {mode}")
