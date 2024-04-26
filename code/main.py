import os

import torch

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from keras.datasets import mnist
from pathlib import Path
# import tensorflow as tf
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

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
    ddpm = Simple_DDPM(T=100, data='mnist')
    ddpm.fit(mnist_data, epochs=500, batch_size=1000, lr=4e-3, emb_dim=32, path='../models/mnist1500.pkl')
    # samp = ddpm.sample(True)
    # utils.display_mnist(samp[0])


@handle('resume-training')
def resume():
    # load our saved model and continue training
    mnist = utils.load_mnist()
    ddpm = utils.load('../models/mnist1500.pkl')
    ddpm.train(mnist, epochs=500, batch_size=1000, lr=3e-3, path='../models/mnist2000.pkl')


@handle('image-samples')
def images_of_sample():
    N = 100
    ddpm = utils.load('../models/mnist2000.pkl')

    samples = ddpm.sample(num_samples=N, return_seq=False)
    utils.save_images(samples)

# @handle('main')
# def main():


@handle('sample-ddpm')
def sample_simple_ddpm():
    samples = 1
    # load model
    ddpm = utils.load('../models/mnist2000.pkl')

    # sample model
    sample = ddpm.sample(num_samples=samples, return_seq=True)
    utils.save_sample(sample)


@handle('gifs-ddpm')
def generate_samples():
    samples = 5
    # load model
    ddpm = utils.load('../models/mnist2000.pkl')
    # print(sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad))

    # sample model
    sample = ddpm.sample(num_samples=samples, return_seq=True)
    # utils.display_mnist(sample)
    for i, s in tqdm(enumerate(sample.unbind(1)), desc='Generate Samples', total=samples):
        utils.create_mnist_gif(s, filename=f'../samples/mnist/gifs/mnist_sample{i}.gif')
    # utils.create_mnist_gif(sample, filename=f'../samples/mnist_progression.gif')


@handle('params')
def parameter_count():
    ddpm = utils.load('../models/mnist2000.pkl')
    print(sum(p.numel() for p in ddpm.model.parameters() if p.requires_grad))


# def display_mnist(sample):
#     if sample.shape != (28, 28):
#         raise ValueError(f"Dimensions must be {(28, 28)}")
#     plt.imshow(sample, cmap=plt.get_cmap('gray'))
#     plt.axis('off')  # Turn off axis
#     plt.show()


if __name__ == "__main__":
    main()
