import os

import torch

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from keras.datasets import mnist
from pathlib import Path
# import tensorflow as tf
import matplotlib.pyplot as plt
import utils

import numpy as np

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    main,
    handle,
    run,
)

from ddpm import (
    Simple_DDPM
)


@handle('train-ddpm')
def simple_ddpm():
    mnist_data = utils.load_mnist()
    ddpm = Simple_DDPM()
    ddpm.fit(mnist_data, epochs=100, batch_size=1000, lr=1e-3, emb_dim=32, path='../models/cosine_schedule.pkl')
    # samp = ddpm.sample(True)
    # utils.display_mnist(samp[0])


@handle('sample-ddpm')
def sample_simple_ddpm(samples=1):
    # load model
    ddpm = utils.load('../models/3layers.pkl')
    # print(sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad))

    # sample model
    sample = ddpm.sample(return_seq=True)
    # utils.display_mnist_progression(sample)
    utils.create_mnist_gif(sample)


@handle('test')
def get_sample():
    device = "cuda"
    mnist_data = utils.load_mnist()
    ddpm = utils.load('../models/3layers.pkl')
    idx = torch.randint(0, len(mnist_data), (1,)).to(device)
    X = mnist_data[idx.cpu()].to(device)
    t = torch.randint(0, 300, (1,))[None, :].to(device)
    print(t)
    ab = ddpm.alpha_bar[t][:, :, None, None]
    Z, epsilon = ddpm.forward_process(X, t)
    ep_pred = ddpm.predict(Z, t).clone().detach()
    X_pred = (Z - torch.sqrt(1 - ab) * ep_pred)/torch.sqrt(ab)
    X_pred = X_pred
    utils.display_mnist(X)
    utils.display_mnist(Z)
    utils.display_mnist(X_pred)

    utils.display_mnist(ep_pred)
    utils.display_mnist(epsilon)



# def display_mnist(sample):
#     if sample.shape != (28, 28):
#         raise ValueError(f"Dimensions must be {(28, 28)}")
#     plt.imshow(sample, cmap=plt.get_cmap('gray'))
#     plt.axis('off')  # Turn off axis
#     plt.show()


if __name__ == "__main__":
    main()
