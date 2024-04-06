import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pathlib import Path
# import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

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


@handle('load-data')
def load_mnist():
    return mnist.load_data()


def display_mnist(sample):
    if sample.shape != (28, 28):
        raise ValueError(f"Dimensions must be {(28, 28)}")
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == "__main__":
    main()
