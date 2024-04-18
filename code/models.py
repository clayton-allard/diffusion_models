import numpy as np
import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, channels=3, layers=3, emb_dim=16, data = 'mnist',device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.emb_dim = emb_dim

        # first two CNN layers
        if data == 'mnist':
            self.input = Block(channels, emb_dim, emb_dim=emb_dim, padding=2, shape=32)
        else:
            self.input = Block(channels, emb_dim, emb_dim=emb_dim, shape=32)

        # scale down
        self.down = nn.ModuleList([Block(emb_dim * 2 ** i, emb_dim * 2 ** (i + 1), emb_dim=emb_dim, shape=2**(4 - i)) for i in range(layers)])
        # self.down = nn.ModuleList(
        #     [Block(emb_dim, emb_dim * 2, emb_dim=emb_dim, shape=16),
        #      Block(emb_dim * 2, emb_dim * 4, emb_dim=emb_dim, shape=8),
        #      Block(emb_dim * 4, emb_dim * 8, emb_dim=emb_dim, shape=4),])

        # transition between downscale and upscale
        self.center = Block(emb_dim * 2 ** layers, emb_dim * 2 ** layers, emb_dim=emb_dim)

        # upscale
        self.trans_conv = nn.ModuleList([nn.ConvTranspose2d(emb_dim * 2 ** (i + 1),
                                                            emb_dim * 2 ** i,
                                                            kernel_size=2,
                                                            stride=2) for i in reversed(range(layers))])

        # Using channels from the transposed convolution and skip connections
        self.up = nn.ModuleList([Block(emb_dim * 2 ** (i + 1), emb_dim * 2 ** i, emb_dim=emb_dim, shape=2**(5 - i)) for i in reversed(range(layers))])

        # final prediction
        if data == 'mnist':
            self.output = Block(emb_dim, channels, emb_dim=emb_dim, padding=0, shape=28)
        else:
            self.output = Block(emb_dim, channels, emb_dim=emb_dim, shape=28)

    # copied from dome272
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # print(inv_freq.shape)
        # print(t.shape)
        # print(channels // 2)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.view(1, -1))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.view(1, -1))
        # stack horizontally
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # store skip connections
        skip_connections = []
        # copied from dome272 to get inputs for position embedding
        # t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim)

        # run forward process through UNET

        x = self.input(x, t)

        # store skip connections prior to max pooling.
        for down in self.down:
            skip_connections += [x]
            x = nn.MaxPool2d(kernel_size=2)(x)
            x = down(x, t)

        skip_connections.reverse()

        # for s in skip_connections:
        #     print(f'skip con: {s.shape}')

        x = self.center(x, t)

        # upscaling includes transposed convolutions and combining skip connections for convolution blocks
        for up, skip, trans in zip(self.up, skip_connections, self.trans_conv):
            # print(x.shape)
            x = trans(x)
            # print(x.shape)
            # print(np.shape(skip))
            x = torch.cat([x, skip], dim=1)
            x = up(x, t)

        return self.output(x, None)


class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, padding=1, emb_dim=16, shape=None, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape if shape is None else (shape, shape)

        # copied from dome272
        self.time_embedding = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ))

        # convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=padding, device=device),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, device=device),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
            # nn.Dropout2d(0.5))

    def forward(self, x, t):
        self.shape = x.shape[-2:] if self.shape is None else self.shape
        time = 0 if t is None else self.time_embedding(t)[:, :, None, None].repeat(1, 1, self.shape[0], self.shape[1])
        return self.conv(x) + time
