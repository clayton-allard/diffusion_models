import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import utils
from models import Unet


class Simple_DDPM():
    def __init__(self,
                 T=1000,
                 min_beta=1e-4,
                 max_beta=0.02,
                 lr=3e-4,
                 device="cuda"):
        self.device = device
        self.T = T
        self.lr = lr
        self.beta = torch.linspace(min_beta, max_beta).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.model = None
        self.shape = None
        self.optimizer = None
        self.loss = None

    def forward_process(self, X, t):
        ab = self.alpha_bar[t]
        # treat each pixel as independent
        epsilon = torch.randn_like(X)
        return {'epsilon': epsilon, 'state': torch.sqrt(ab) * X + torch.sqrt(1 - ab) * epsilon}

    def backward_process(self, Z, t, epsilon):
        cost = self.loss(epsilon, self.predict(Z, t))

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

    def fit(self, X, epochs=100, batch_size=30, path=None):
        # assume all shapes are the same
        self.shape = X[0].shape

        # configure model
        model = Unet(channels=self.shape[0]).to(self.device)
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        for i in tqdm(range(epochs)):

            dataloader = DataLoader(X, batch_size, shuffle=True)

            for batch in tqdm(dataloader, leave=False):
                rand_times = torch.randint(low=1, high=self.T, size=(len(batch),))
                for j in batch:
                    t = rand_times[j]
                    epsilon, Z = self.forward_process(X, t).values()
                    # update Unet weights
                    self.backward_process(Z, t, epsilon)

        # save model so we do not need to rerun program
        if path is not None:
            abspath = os.path.abspath(path)
            torch.save(model, abspath)
            print(f"The model has been saved to {abspath}")

    def predict(self, Z, t):
        # use the model
        return self.model(Z, t)

    def sample(self, return_seq=False):
        '''
        Generate a sample. Can generate the final output of the sample or the entire sequence of  the denoising.

        :param return_seq: set to True to return the whole sequence or False for just the final output (default: False)
        :return: The generated sample
        '''
        X = torch.zeros([self.T] + self.shape)
        X[-1] = torch.randn_like(self.shape)
        # traversing backwards in the process
        for i in range(self.T, 1):
            z = torch.randn_like(self.shape) if i > 1 else 0

            sigma = torch.sqrt(self.beta[i])
            X[i - 1] = (X[i] - self.beta[i] / torch.sqrt(1 - self.alpha_bar[i]) * self.predict(X[i], i)) + sigma * z
        return X if return_seq else X[0]
