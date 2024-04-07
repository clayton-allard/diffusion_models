import numpy as np
import utils


class simple_ddpm():
    def __init__(self,
                 T=1000,
                 min_beta=0.0001,
                 max_beta=0.02):
        self.T = T
        self.schedule = np.linspace(min_beta, max_beta)
        self.weights = None

    def forward_process(self, X, t):
        alpha_bar = utils.alpha(self.schedule, t)
        # treat each pixel as independent
        epsilon = np.random.normal(size=X.shape)
        return {'epsilon': epsilon, 'state': np.sqrt(alpha_bar) * X + np.sqrt(1 - alpha_bar) * epsilon}

    def backward_process(self, Z, t, epsilon):
        # some complicated unet shit
        raise NotImplementedError()

    def fit(self, X, steps):
        # assume all shapes are the same
        self.shape = X[0].shape
        for i in range(steps):
            t = np.random.choice(self.T)
            epsilon, Z = self.forward_process(X, t).values()
            # update Unet weights
            self.backward_process(Z, t, epsilon)

    def predict(self, Z, t):
        # use the model
        raise NotImplementedError()

    def sample(self, return_seq=False):
        X = np.zeros([self.T] + self.shape)
        X[-1] = np.random.normal(size=self.shape)
        for i in range(self.T, 1):
            z = np.random.normal(size = self.shape) if i > 1 else 0

            sigma = np.sqrt(self.schedule[i])
            alpha_bar = utils.alpha(self.schedule, i)
            X[i-1] = (X[i] - self.schedule[i]/np.sqrt(1 - alpha_bar)*self.predict(X[i], i)) + sigma * z
        return X if return_seq else X[0]