import struct
import numpy as np
import matplotlib.pyplot as plt


def export_data(mode="train"):
    if mode == "train":
        data_path = "data/train-images-idx3-ubyte"
        label_path = "data/train-labels-idx1-ubyte"
    elif mode == "test":
        data_path = "data/t10k-images-idx3-ubyte"
        label_path = "data/t10k-labels-idx1-ubyte"

    with open(label_path, 'rb') as f_label:
        magic, num = struct.unpack(">II", f_label.read(8))
        labels = np.fromfile(f_label, dtype=np.int8)

    with open(data_path, 'rb') as f_image:
        magic, num, rows, cols = struct.unpack(">IIII", f_image.read(16))
        images = np.fromfile(f_image, dtype=np.uint8).reshape(len(labels), rows, cols)

    # Reshape and normalize
    images = np.reshape(images, [images.shape[0], images.shape[1] * images.shape[2]]) * 1.0 / 255

    return images, labels


if __name__ == "__main__":
    export_data("test")
    pass
