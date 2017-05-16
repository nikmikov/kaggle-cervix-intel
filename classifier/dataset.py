#from __future__ import division, absolute_import, print_function

import os
import json
import argparse
import hashlib
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data

from collections import namedtuple
AnnotatedImage =  namedtuple('AnnotatedImage', 'filepath cachedpath x y width height')

def pil_loader(path):
    return Image.open(path).convert('RGB')

def name_md5(path):
    m = hashlib.md5()
    m.update(path.encode('utf-8'))
    return m.hexdigest() + ".jpg"

def read_annotations(path):
    alist = []
    with open(path) as json_data:
        d = json.load(json_data)
        for elem in d:
            filepath = os.path.abspath(elem['filename'])
            hash = name_md5(filepath)
            annotations = elem['annotations']
            if len(annotations) == 1:
                a = annotations[0]
                ra = AnnotatedImage(filepath, [None], a['x'], a['y'],  a['width'],  a['height'])
                alist.append(ra)
            else:
                print("Image %s not annotated. Skipping" % filepath)

    return alist


class CervixClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for cervix classification training/evaluation
    """
    def __init__(self, annotations_path, cache_dir, target_size, transform, type_map, is_train):
        self.images = read_annotations(annotations_path)
        self.is_train = is_train
        for ai in self.images:
            cpath = os.path.join(cache_dir, name_md5(ai.filepath))
            if os.path.isfile(cpath):
                ai.cachedpath[0] = cpath

        self.cache_dir = cache_dir
        self.target_size = (target_size, target_size)
        self.transform = transform
        self.type_map = type_map

    def __getitem__(self, index):
        im = self.images[index]
        path = im.filepath

        if not im.cachedpath[0]:
            # load and transform
            image = pil_loader(path)
            # crop at bounding box
            sz = max(im.width, im.height)
            image = image.crop( (im.x, im.y, im.x + sz, im.y + sz) )
            # resize
            image = image.resize( self.target_size, Image.BILINEAR  )
        else:
            # load cached image
            image = pil_loader(im.cachedpath[0])

        if not im.cachedpath[0] and self.cache_dir:
            p = os.path.join(self.cache_dir, name_md5(path))
            im.cachedpath[0] = p
            # cache it
            assert( not os.path.isfile(im.cachedpath[0]))
            image.save(im.cachedpath[0])

        if self.transform:
            image = self.transform(image)

        t = path.split('/')[-2].lower()
        if t in self.type_map:
            v = self.type_map[t]
            return image, v
        else:
            return image, []

    def __len__(self):
        return len(self.images)


def create_data_loader( annotations_path, cache_dir, target_size, transform, batch_size, num_workers, validation_split, type_map, is_train ):
    """
    Return tuple of dataloaders according split

    """

    image_dataset = CervixClassificationDataset(
        annotations_path = annotations_path,
        cache_dir = cache_dir,
        target_size = target_size,
        transform = transform,
        type_map = type_map,
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
