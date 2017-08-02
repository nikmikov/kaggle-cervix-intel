#from __future__ import division, absolute_import, print_function

import os
import json
import argparse
import hashlib
import numpy as np
import random

import torch
import torch.utils.data

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.annotations import AnnotatedCervixImage, read_annotations

import torchvision.transforms as transforms
from utils.transformation import RandomRotate

def pil_loader(path):
    return Image.open(path).convert('RGB')

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform1 = transforms.Compose ( [
#    transforms.Scale(224),
    RandomRotate(),
#    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])

transform2 = transforms.Compose ( [
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])

transform3 = transforms.Compose ( [
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    normalize
])

transform4 = transforms.Compose ( [
    transforms.RandomSizedCrop(64),
    transforms.ToTensor()
])

transform5 = transforms.Compose ( [
    transforms.Scale(64),
    transforms.ToTensor(),
    normalize
])


def transform(image):
    """
    transform image to tensors
    """

    rotated = RandomRotate()(image)

    return transform1(image), transform2(rotated), transform3(rotated), transform4(rotated), transform5(rotated)


class SequentialSubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class CervixClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for cervix classification training/evaluation
    """
    def __init__(self, annotations_path, is_train, type_map):
        self.images = read_annotations(annotations_path)
        self.is_train = is_train
        self.type_map = type_map

    def __getitem__(self, index):
        im = self.images[index]
        image = pil_loader(im.filepath)

        image = transform(image)

        if self.is_train or im.cervix_type is not None:
            return image, self.type_map[im.cervix_type]
        else:
            return image, []


    def __len__(self):
        return len(self.images)


def create_weighted_random_sampler(images):
    """
    Random sampler to uniformly draw 1,2,3 type samples from images with equal probabilities

    """
    n = [0,0,0,0]
    for im in images:
        n[im.cervix_type] += 1
    nA = len(images)
    assert ( n[1] + n[2] + n[3] == nA )
    wmap = {  1: nA/n[1], 2: nA/n[2], 3: nA/n[3] }
    weights = [ wmap[ im.cervix_type ] for im in images ]

    return torch.utils.data.sampler.WeightedRandomSampler( weights, 3*min(n[1:]) )


def create_data_loader( annotations_path, batch_size, num_workers, validation_split, is_train, type_map, rseed = 0):
    """
    Return tuple of dataloaders according split

    """
    rngstate = random.getstate()
    random.seed(rseed)

    image_dataset = CervixClassificationDataset(
        annotations_path = annotations_path,
        is_train = is_train,
        type_map = type_map
    )

    def create_loader(sampler):
        return torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=num_workers
        )

    if not is_train:
        sampler1 = torch.utils.data.sampler.SequentialSampler(image_dataset)
        loader2 = None
    elif validation_split is None:
        random.shuffle(image_dataset.images)
        sampler1 =  create_weighted_random_sampler(image_dataset.images)
        loader2 = None
    else:
        random.shuffle(image_dataset.images)
        split_pt = int(len(image_dataset) * validation_split)
        sampler1 = create_weighted_random_sampler(image_dataset.images[:split_pt])
#        assert(len(sampler1) == split_pt)
        indices = range(split_pt, len(image_dataset.images))
        sampler2 = SequentialSubsetSampler(indices)
        print( image_dataset.images[ indices[0] ] )
        print( image_dataset.images[ indices[1] ] )
        loader2 = create_loader(sampler2)


    loader1 = create_loader(sampler1)

    random.setstate(rngstate) # restore rng state

    return (loader1, loader2)
