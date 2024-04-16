import argparse
import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from keras.datasets import mnist
import pickle
import gzip

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    abspath = os.path.abspath(path)
    abspath = abspath if abspath[-4:] == '.pkl' else abspath + '.pkl'
    with open(abspath, "rb") as f:
        model = pickle.load(f)
    return model

def load_mnist():
    # Download and load the MNIST training dataset
    with gzip.open('../data/MNIST/raw/train-images-idx3-ubyte.gz', 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)/255.
    return torch.tensor(data, dtype=torch.float32)

def display_mnist(sample):
    if sample.shape != (28, 28):
        raise ValueError(f"Dimensions must be {(28, 28)}")
    plt.imshow(sample.cpu(), cmap=plt.get_cmap('gray'))
    plt.axis('off')  # Turn off axis
    plt.show()

def alpha(X, t):
    return np.prod(1 - X[:t])

_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)
