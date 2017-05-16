from __future__ import division
import torch
import random

from PIL import Image

class RandomRotate(object):
    """Randomly rotate image
    """
    def __call__(self, img):
        angle = random.uniform(0,360)
        return img.rotate(angle, Image.BILINEAR, expand = True)
