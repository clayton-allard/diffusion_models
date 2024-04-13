import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, channels, device='cuda', layers=4, emb_dim=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.device = device
        self.emb_dim = emb_dim

        self.down = nn.ModuleList([Block(channels, emb_dim)] + \
                                  [Block(emb_dim * 2 ** i, emb_dim * 2 ** (i + 1)) for i in range(layers)])

        self.up = nn.ModuleList([layer for i in reversed(range(layers)) for layer in [
                                 nn.ConvTranspose2d(emb_dim * 2 ** i, emb_dim * 2 ** (i + 1), 3, padding=1),
                                 Block(emb_dim * 2 ** (i + 1), emb_dim * 2 ** (i + 2))
                          ]])

    def forward(self):
        raise NotImplementedError()


class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(nn.BatchNorm2d(inp_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)
