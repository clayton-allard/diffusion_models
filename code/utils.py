import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

def load_mnist():
    return mnist.load_data()

def display_mnist(sample):
    if sample.shape != (28, 28):
        raise ValueError(f"Dimensions must be {(28, 28)}")
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
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
