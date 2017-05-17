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
AnnotatedImage =  namedtuple('AnnotatedImage', 'filepath cachedpath xmin ymin xmax ymax width height')

def pil_loader(path):
    return Image.open(path).convert('RGB')

def name_md5(path):
    m = hashlib.md5()
    m.update(path.encode('utf-8'))
    return m.hexdigest() + ".jpg"

def read_annotations(path, is_train):
    alist = []
    with open(path) as json_data:
        d = json.load(json_data)
        for elem in d:
            filepath = os.path.abspath(elem['filename'])
            hash = name_md5(filepath)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except OSError:
                print(" E Error opening file: %s. Ignoring" % filepath )
                continue

            annotations = elem['annotations']
            if len(annotations) == 1:
                a = annotations[0]
                x0, y0 = a['x'], a['y']
                x1, y1 = x0 + a['width'], y0 + a['height']
                # normalized bounding box
                ra = AnnotatedImage(filepath, [None], x0/width, y0/height, x1/width, y1/height, width, height)
                alist.append(ra)
            elif not is_train:
                alist.append( AnnotatedImage(filepath, [None], None, None, None, None,width,height) )

    return alist


class CervixLocalisationDataset(torch.utils.data.Dataset):
    """
    Dataset for cervix bounding boxes localisation training/evaluation
    """
    def __init__(self, annotations_path, cache_dir, target_size, transform, is_train):
        self.images = read_annotations(annotations_path, is_train)
        self.is_train = is_train
        for ai in self.images:
            cpath = os.path.join(cache_dir, name_md5(ai.filepath))
            if os.path.isfile(cpath):
                ai.cachedpath[0] = cpath

        self.cache_dir = cache_dir
        self.target_size = (target_size, target_size)
        self.transform = transform

    def __getitem__(self, index):
        im = self.images[index]
        path = im.filepath

        if not im.cachedpath[0]:
            # load and transform
            image = pil_loader(path)
            image = image.resize( self.target_size, Image.BILINEAR  )
        else:
            # load cached image
            image = pil_loader(im.cachedpath[0])

        if not im.cachedpath[0] and self.cache_dir:
            im.cachedpath[0] = os.path.join(self.cache_dir, name_md5(path))
            # cache it
            image.save(im.cachedpath[0])

        if self.transform:
            image = self.transform(image)


        if im.xmin is not None:
            return image, torch.FloatTensor( ( im.xmin, im.ymin, im.xmax, im.ymax ))
        else:
            return image, torch.zeros(4,1) #if not self.is_train else None

    def __len__(self):
        return len(self.images)


def create_data_loader( annotations_path, cache_dir, target_size, transform, batch_size, num_workers, validation_split, is_train ):
    """
    Return tuple of dataloaders according split

    """

    image_dataset = CervixLocalisationDataset(
        annotations_path = annotations_path,
        cache_dir = cache_dir,
        target_size = target_size,
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
