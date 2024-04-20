import os
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pytorch_ssim

from models import Unet


class Simple_DDPM():
    def __init__(self,
                 T=1000,
                 # min_beta=1e-4,
                 # max_beta=0.02,
                 data = 'mnist',
                 device="cuda"):
        self.device = device
        self.T = T
        self.lr = None
        self.data = data
        # self.beta = torch.linspace(min_beta, max_beta, steps=self.T).to(device)
        # self.alpha = 1 - self.beta
        # self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        s = torch.tensor(0.008, device=device)
        t = torch.arange(0, self.T, 1).to(device)
        alpha_bar0 = torch.cos(s / (1 + s) * torch.pi / 2) ** 2
        self.alpha_bar = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2 / alpha_bar0
        self.alpha = self.alpha_bar[1:] / self.alpha_bar[:-1]
        number = torch.tensor([self.alpha_bar[0]], device=device)
        self.alpha = torch.cat([number, self.alpha], dim=0)
        self.beta = 1 - self.alpha
        self.shape = None
        self.model = None
        self.optimizer = None
        self.loss = None

    def forward_process(self, X, t):
        ab = self.alpha_bar[t][:, :, None, None]
        epsilon = torch.randn_like(X).to(self.device)
        return torch.sqrt(ab) * X + torch.sqrt(1 - ab) * epsilon, epsilon

    def backward_process(self, Z, t, epsilon):
        cost = self.loss(epsilon, self.predict(Z, t))

        # back propogation
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost

    def fit(self, X, epochs=100, batch_size=2500, lr=1e-3, emb_dim=16, path=None):
        # assume all shapes are the same
        self.shape = X[0].shape
        self.lr = lr

        # configure model
        model = Unet(channels=self.shape[0], layers=4, emb_dim=emb_dim, data=self.data).to(self.device)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.train(X, epochs=epochs, batch_size=batch_size, path=path)

    def train(self, X, epochs=100, batch_size=2500, lr=None, path=None):

        # if this is run before fitting
        if self.model is None:
            raise RuntimeError("Must fit before calling this function.")

        # use a different learning rate
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Initialize the tqdm progress bar
        progress_bar = tqdm(total=epochs, desc="Epoch")

        for i in range(epochs):
            # create batches
            dataloader = DataLoader(X, batch_size, shuffle=True)
            for batch in dataloader:
                # random times
                rand_times = torch.randint(low=1, high=self.T, size=(len(batch),))
                t = rand_times.unsqueeze(1).to(self.device)
                Z, epsilon = self.forward_process(batch.to(self.device), t)
                cost = self.backward_process(Z, t, epsilon)

                # Update the tqdm progress bar with the current loss
                progress_bar.set_postfix({'Loss': cost.item()})
            progress_bar.update()

            # update model every 100 epochs in case code breaks
            if (i + 1) % 100 == 0 and path is not None:
                self.save(path, False)

        # Close the tqdm progress bar
        progress_bar.close()

        # save model so we do not need to rerun program
        if path is not None:
            self.save(path)

    def save(self, path, verbose=True):
        abspath = os.path.abspath(path)
        abspath = abspath if abspath[-4:] == '.pkl' else abspath + '.pkl'
        with open(abspath, "wb") as f:
            pickle.dump(self, f)
        if verbose:
            print(f"The model has been saved to {abspath}")

    def predict(self, Z, t):
        # use the model
        return self.model(Z, t)

    def sample(self, num_samples=1, return_seq=False):
        """
        Generate a sample. Can generate the final output of the sample or the entire sequence of  the denoising.

        :param num_samples: number of samples (default: True)
        :param return_seq: set to True to return the whole sequence or False for just the final output (default: False)
        :return: The generated samples
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.zeros([self.T, num_samples] + list(self.shape)).to(self.device)
            X[-1] = torch.randn_like(X[-1])
            for i in range(self.T - 1, 0, -1):
                z = torch.randn_like(X[i]) if i > 1 else 0
                sigma = torch.sqrt(self.beta[i])
                # print(X[i].shape)
                X[i - 1] = (X[i] - self.beta[i] / torch.sqrt(1 - self.alpha_bar[i]) * self.predict(X[i],
                                                                                                   torch.tensor(i).to(
                                                                                                       self.device))) / torch.sqrt(
                    self.alpha[i]) + sigma * z
                im = X[i - 1].cpu().numpy()
        self.model.train()
        X = (X.clamp(-1, 1) + 1) / 2
        X = (X * 255).type(torch.uint8)
        if num_samples == 1:
            X = X.squeeze(dim=1)
            return X if return_seq else X[0]
        return X if return_seq else X[0, :, :, :, :]
