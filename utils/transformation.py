from __future__ import division
import torch
import random

from PIL import Image


class RandomRotate(object):
    """Randomly flip or rotates the given PIL.Image
    """

    def __call__(self, img):
        l = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE, None]
        v = random.choice(l)
        if not v:
            return img
        return img.transpose(v)


class Scale(object):
    """Randomly flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


