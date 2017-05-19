#from __future__ import division, absolute_import, print_function

import os
import json
import argparse
import hashlib
import numpy as np

from PIL import Image

import torch
import torch.utils.data


from utils.annotations import AnnotatedCervixImage, read_annotations

def pil_loader(path):
    return Image.open(path).convert('RGB')


class CervixLocalisationDataset(torch.utils.data.Dataset):
    """
    Dataset for cervix bounding boxes localisation training/evaluation
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
            return image, torch.FloatTensor( ( im.xmin, im.ymin, im.xmax, im.ymax ))
        else:
            return image,   torch.zeros(4,1) #if not self.is_train else None

    def __len__(self):
        return len(self.images)


def create_data_loader( annotations_path, transform, batch_size, num_workers, validation_split, is_train ):
    """
    Return tuple of dataloaders according

    """

    image_dataset = CervixLocalisationDataset(
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
        sampler2 = torch.utils.data.sampler.SubsetRandomSampler(indices[split_pt:])
        loader2 = create_loader(sampler2)


    loader1 = create_loader(sampler1)

    return (loader1, loader2)
