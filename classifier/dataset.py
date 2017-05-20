#from __future__ import division, absolute_import, print_function

import os
import json
import argparse
import hashlib
import numpy as np

import torch
import torch.utils.data

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.annotations import AnnotatedCervixImage, read_annotations

def pil_loader(path):
    return Image.open(path).convert('RGB')

class SequentialSubsetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.samples = data_source

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

class CervixClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for cervix classification training/evaluation
    """
    def __init__(self, annotations_path, transform, is_train):
        self.images = read_annotations(annotations_path)
        self.is_train = is_train
        self.transform = transform

    def __getitem__(self, index):
        im = self.images[index]
        image = pil_loader(im.filepath)

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, im.cervix_type - 1
        else:
            return image, []


    def __len__(self):
        return len(self.images)


def create_data_loader( annotations_path, transform, batch_size, num_workers, validation_split, is_train ):
    """
    Return tuple of dataloaders according split

    """

    image_dataset = CervixClassificationDataset(
        annotations_path = annotations_path,
        transform = transform,
        is_train = is_train
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
        sampler1 = torch.utils.data.sampler.RandomSampler(image_dataset)
        loader2 = None
    else:
        indices = np.random.permutation(len(image_dataset))
        split_pt = int(len(image_dataset) * validation_split)
        sampler1 = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_pt])
        sampler2 = SequentialSubsetSampler(indices[split_pt:])
        loader2 = create_loader(sampler2)


    loader1 = create_loader(sampler1)

    return (loader1, loader2)
