import os
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
def load_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    rand_index = np.random.choice(train_X.shape[0])
    plt.imshow(train_X[rand_index], cmap=plt.get_cmap('gray'))
    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == "__main__":
    main()
