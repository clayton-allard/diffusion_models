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
    ddpm.fit(mnist_data, epochs=100, path='../models/simple_100epoch_1e-3lr.pkl')
    # samp = ddpm.sample(True)
    # utils.display_mnist(samp[0])


@handle('sample-ddpm')
def sample_simple_ddpm(samples=1):
    # load model
    ddpm = utils.load('../models/simple_100epoch_1e-3lr.pkl')
    # print(sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad))

    # sample model
    sample = ddpm.sample()
    utils.display_mnist(sample[0])


# def display_mnist(sample):
#     if sample.shape != (28, 28):
#         raise ValueError(f"Dimensions must be {(28, 28)}")
#     plt.imshow(sample, cmap=plt.get_cmap('gray'))
#     plt.axis('off')  # Turn off axis
#     plt.show()


if __name__ == "__main__":
    main()
