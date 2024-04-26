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
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import numpy as np


# def augment(data, num_samples):
#     idx = torch.randint(0, len(data), (1,)).to("cuda")
#     X = data[idx.cpu()].to("cuda")
#     transform = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))

def load_cifar10():
    # Load images from .gz file
    with gzip.open('../data/cifar_images.gz', 'rb') as f:
        all_images_uint8 = np.load(f)
        all_images_uint8 = np.reshape(all_images_uint8, (-1, 3, 32, 32))

    # Convert pixel values to float32 and scale to range [0, 1]
    all_images_normalized = all_images_uint8.astype(np.float32) / 255.0
    return torch.tensor(all_images_normalized, dtype=torch.float32)
    # # Define the directory containing CIFAR-10 dataset files
    # data_dir = "../data/cifar-10-batches-py/"  # Example path to the directory containing dataset files
    #
    # # Initialize empty lists to store images and labels
    # all_images = []
    #
    # # Iterate over each dataset file in the directory
    # for filename in os.listdir(data_dir):
    #     if filename.startswith("data_batch_"):  # Check if the file is a data batch file
    #         file_path = os.path.join(data_dir, filename)
    #         with open(file_path, 'rb') as file:
    #             cifar_data = pickle.load(file, encoding='bytes')
    #         images = cifar_data[b'data']  # Extract raw image data
    #         images_reshaped = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape images to (3, 32, 32)
    #         all_images.append(images_reshaped)
    #
    # # Concatenate images from all batches
    # all_images_concatenated = np.concatenate(all_images, axis=0)
    #
    # # Convert images to uint8 and scale to [0, 255]
    # all_images_uint8 = (all_images_concatenated * 255).astype(np.uint8)
    #
    # # Save images to .gz file
    # with gzip.open('../data/cifar_images.gz', 'wb') as f:
    #     np.save(f, all_images_uint8)
    #
    # print("Images saved to cifar_images.gz")


def display_cifar(sample):
    sample = torch.squeeze(sample).cpu()
    sample = sample.reshape((32, 32, 3))
    plt.imshow(sample)
    plt.axis('off')  # Turn off axis
    plt.show()


def load(path):
    abspath = os.path.abspath(path)
    abspath = abspath if abspath[-4:] == '.pkl' else abspath + '.pkl'
    with open(abspath, "rb") as f:
        model = pickle.load(f)
    return model


def load_mnist():
    # Download and load the MNIST training dataset
    with gzip.open('../data/mnist.gz', 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28) / 255.
    data = data * 2 - 1
    return torch.tensor(data, dtype=torch.float32)


def display_mnist(sample):
    sample = torch.squeeze(sample).cpu()
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
    plt.axis('off')  # Turn off axis
    plt.show()


def save_images(sample):
    for i, s in enumerate(sample):
        # plt.figure(figsize=(4.80, 4.80))
        # plt.imshow(s.squeeze().cpu(), cmap=plt.get_cmap('gray'))
        # plt.axis('off')  # Turn off axis
        # plt.savefig(f'../samples/mnist/outputs/t{i}', transparent = True)
        # plt.close()

        # Convert the pixel values to an image
        image = Image.fromarray(s.squeeze().detach().cpu().numpy())

        # Save the image to a file
        image.save(f'../samples/mnist/outputs/t{i}.png')

def save_sample(sample):
    for i, s in enumerate(sample):
        # Convert the pixel values to an image
        image = Image.fromarray(s.squeeze().detach().cpu().numpy())

        # Save the image to a file
        image.save(f'../samples/mnist/single_sample/t{i}.png')

        # plt.figure()
        # plt.imshow(s.squeeze().cpu(), cmap=plt.get_cmap('gray'))
        # plt.axis('off')  # Turn off axis
        # plt.savefig(f'../samples/mnist/single_sample/t{i}')
        # plt.close()

    # , filename = f'../samples/mnist/single_sample/mnist_sample.png'


def create_cifar_gif(sample, filename='cifar_progression.gif'):
    sample = np.squeeze(sample).cpu()
    sample = np.transpose(sample, (0, 2, 3, 1))
    images = []
    for i, s in reversed(list(enumerate(sample))):
        plt.imshow(s)
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
