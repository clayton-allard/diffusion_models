import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, channels, device='cuda', layers=4, emb_dim=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.device = device
        self.emb_dim = emb_dim

        self.time_embedding = nn.Linear(1, emb_dim, device=device)

        self.down = nn.ModuleList([Block(channels, emb_dim)] + \
                                  [Block(emb_dim * 2 ** i, emb_dim * 2 ** (i + 1)) for i in range(layers)])

        self.up = nn.ModuleList([Block(emb_dim * 2 ** (i + 1), emb_dim * 2 ** i) for i in reversed(range(layers))] + \
                                [Block(emb_dim, channels)])

        self.trans_conv = nn.ModuleList([nn.ConvTranspose2d(emb_dim * 2 ** (i + 1),
                                                            emb_dim * 2 ** i,
                                                            kernel_size=2,
                                                            stride=2) for i in reversed(range(layers))] + \
                                [Block(emb_dim, channels)])

        self.center = Block(emb_dim * 2 ** layers, emb_dim * 2 ** layers)

    def forward(self, x, t):
        skip_connections = []
        t = t.unsqueeze(-1).type(torch.float)

        for down in self.down:
            x = down(x) + self.time_embedding(t)
            skip_connections += [x]
            x = nn.MaxPool2d(kernel_size=2)(x)

        skip_connections.reverse()

        x = self.center(x)

        for up, skip, trans in zip(self.up, skip_connections, self.trans_conv):
            x = trans(x)
            x = torch.cat(x, skip)
            x = up(x) + self.time_embedding(t)



class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(nn.Conv2d(inp_channels, out_channels, kernel_size, padding=padding, device=self.device),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, device=self.device),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)
