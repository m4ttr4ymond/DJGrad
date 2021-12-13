import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from easydict import EasyDict


def add_pattern_bd(x: np.ndarray, distance: int = 2, pixel_value: int = 1) -> np.ndarray:
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix or N X W X H X C matrix, pixels will ne added to all channels
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width, height = x.shape[1:3]
        x[:, width - distance, height - distance, :] = pixel_value
        x[:, width - distance - 1, height - distance - 1, :] = pixel_value
        x[:, width - distance, height - distance - 2, :] = pixel_value
        x[:, width - distance - 2, height - distance, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        x[width - distance, height - distance, :] = pixel_value
        x[width - distance - 1, height - distance - 1, :] = pixel_value
        x[width - distance, height - distance - 2, :] = pixel_value
        x[width - distance - 2, height - distance, :] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    else:
        raise ValueError("Invalid array shape: " + str(shape))
    return x


def ld_mnist():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    dataset, info = tfds.load(
        "mnist", data_dir="gs://tfds-data/datasets", with_info=True, as_supervised=True
    )
    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(convert_types).batch(128)
    return EasyDict(train=mnist_train, test=mnist_test)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


data = ld_mnist()
for (x, y) in data.train:
    new_x = add_pattern_bd(x[0])
    print(y)
    break

new_x = (new_x * 255).astype(np.uint8)
cv2.imwrite('backdoortest.png', new_x)