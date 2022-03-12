import cv2
import numpy as np


def load_img(img_path, target_size=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        img = cv2.resize(img, target_size)

    return img


def img_to_array(img):
    tensors = np.array(img) / 255.
    tensors = np.expand_dims(tensors, axis=0)

    return tensors
