import argparse
import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from keras.datasets import mnist
import pickle
import gzip
import time
from IPython.display import display, clear_output
from PIL import Image
import io

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
    data = data.reshape(-1, 1, 28, 28) / 255.
    data = data * 2 - 1
    return torch.tensor(data, dtype=torch.float32)


def display_mnist(sample):
    sample = torch.squeeze(sample).cpu()
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
    plt.axis('off')  # Turn off axis
    plt.show()


def create_mnist_gif(sample, filename='mnist_progression.gif', duration=10):
    sample = np.squeeze(sample)
    images = []
    for i, s in reversed(list(enumerate(sample))):
        plt.imshow(s.cpu(), cmap=plt.get_cmap('gray'))
        plt.title(f't = {i}')
        plt.axis('off')  # Turn off axis
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Convert buffer to PIL Image
        img = Image.open(buf)
        images.append(img)
        plt.close()
        # print(i)

    # frame_duration = duration * 1000 / len(images)
    # Save images as GIF
    images[0].save(filename, save_all=True, append_images=images[1:], duration=10, loop=1)


# def display_mnist_progression(sample):
#     sample = np.squeeze(sample)
#     for i, s in reversed(list(enumerate(sample))):
#         plt.imshow(s.cpu(), cmap=plt.get_cmap('gray'))
#         plt.title(f't = {i}')
#         plt.axis('off')  # Turn off axis
#         display(plt.gcf())
#         time.sleep(1)
#         clear_output(wait=True)

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
