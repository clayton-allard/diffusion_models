import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, channels, layers=3, emb_dim=16, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.emb_dim = emb_dim

        # first two CNN layers
        self.input = Block(channels, emb_dim)

        # scale down
        self.down = nn.ModuleList([Block(emb_dim * 2 ** i, emb_dim * 2 ** (i + 1)) for i in range(layers)])

        # transition between downscale and upscale
        self.center = Block(emb_dim * 2 ** layers, emb_dim * 2 ** layers)

        # upscale
        self.trans_conv = nn.ModuleList([nn.ConvTranspose2d(emb_dim * 2 ** (i + 1),
                                                            emb_dim * 2 ** i,
                                                            kernel_size=2,
                                                            stride=2) for i in reversed(range(layers))])

        # Using channels from the transposed convolution and skip connections
        self.up = nn.ModuleList([Block(emb_dim * 2 ** (i + 1), emb_dim * 2 ** i) for i in reversed(range(layers))])

        # final prediction
        self.output = Block(emb_dim, channels)

    # copied from dome272
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # stack horizontally
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # store skip connections
        skip_connections = []
        # copied from dome272 to get inputs for position embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim)

        # run forward process through UNET

        x = self.input(x)

        # store skip connections prior to max pooling.
        for down in self.down:
            skip_connections += [x]
            x = nn.MaxPool2d(kernel_size=2)(x)
            x = down(x, t)

        skip_connections.reverse()

        x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.center(x)

        # upscaling includes transposed convolutions and combining skip connections for convolution blocks
        for up, skip, trans in zip(self.up, skip_connections, self.trans_conv):
            x = trans(x)
            x = torch.cat(x, skip)
            x = up(x, t)

        return self.output(x, None)


class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, padding=1, emb_dim=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # copied from dome272
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ))

        # convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=padding, device=self.device),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, device=self.device),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x, t):
        time = 0 if t is None else self.time_embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.conv(x) + time
