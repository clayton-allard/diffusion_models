import torch
from tqdm import tqdm

import numpy as np
import utils


class Simple_DDPM():
    def __init__(self,
                 T=1000,
                 min_beta=1e-4,
                 max_beta=0.02,
                 device="cuda"):
        self.device = device
        self.T = T
        self.beta = torch.linspace(min_beta, max_beta).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.weights = None

    def forward_process(self, X, t):
        ab = self.alpha_bar[:t]
        # treat each pixel as independent
        epsilon = np.random.normal(size=X.shape)
        return {'epsilon': epsilon, 'state': np.sqrt(ab) * X + np.sqrt(1 - ab) * epsilon}

    def backward_process(self, Z, t, epsilon):
        # some complicated unet shit
        raise NotImplementedError()

    def fit(self, X, steps):
        # assume all shapes are the same
        self.shape = X[0].shape
        for i in tqdm(range(steps)):
            t = np.random.choice(self.T)
            epsilon, Z = self.forward_process(X, t).values()
            # update Unet weights
            self.backward_process(Z, t, epsilon)

    def predict(self, Z, t):
        # use the model
        raise NotImplementedError()

    def sample(self, return_seq=False):
        '''
        Generate a sample. Can generate the final output of the sample or the entire sequence of  the denoising.

        :param return_seq: set to True to return the whole sequence or False for just the final output (default: False)
        :return: The generated sample
        '''
        X = np.zeros([self.T] + self.shape)
        X[-1] = np.random.normal(size=self.shape)
        for i in range(self.T, 1):
            z = np.random.normal(size = self.shape) if i > 1 else 0

            sigma = np.sqrt(self.beta[i])
            alpha_bar = utils.alpha(self.beta, i)
            X[i-1] = (X[i] - self.beta[i]/np.sqrt(1 - alpha_bar)*self.predict(X[i], i)) + sigma * z
        return X if return_seq else X[0]