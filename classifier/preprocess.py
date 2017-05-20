#!/usr/bin/env python3

import os
import time
import argparse
import multiprocessing
import json
import random

from joblib import Parallel, delayed

from utils.misc import create_dir_if_not_exists, str_to_md5
from utils.annotations import read_annotations, save_annotations, AnnotatedCervixImage

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def process_image(output_dir, im, target_size):


    if im.xmin is None:
        return []

    image = Image.open(im.filepath).convert('RGB')
    w, h = image.size


    basename = str_to_md5(im.filepath)

    image = image.crop( ( im.xmin*w, im.ymin*h, im.xmax*w, im.ymax*h ) )
    image = image.resize( (target_size, target_size), Image.BILINEAR  )

    p = os.path.join(output_dir, basename + ("_%d.jpg" % target_size) )
    p = os.path.abspath(p)
    image.save(p)

    return [im.with_filename(p)]

def run(options):
    work_dir = os.path.join(options.work_dir, "classifier", str(options.target_image_size) )
    create_dir_if_not_exists(work_dir)
    print(" + Random seed: %d" % options.random_seed)

    #load annotaions file
    alist = read_annotations(options.input)
    print(" + Input: %s. Images loaded: %d" % (options.input, len(alist)) )

    target_size = options.target_image_size
    print(" + Target image size: ", str(target_size))
    # for each image
    res_ = Parallel(n_jobs=options.workers, backend="threading")(
        delayed(process_image)(work_dir, im, target_size) for im in alist)

    res = []
    for r in res_:
        res += r

    save_path = options.output
    print(" + Output: %s. Images exported: %d" % ( options.output, len(res)) )
    save_annotations(res, options.output)

    print(" + DONE")

def main():

    default_workers = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser()

    parser.add_argument("--workdir", dest="work_dir",
                        help="Work directory to store intermediate state and caches", required=True)

    parser.add_argument("--target-image-size", dest="target_image_size", type=int,
                        help="Target image size of smallest side in pixels.  ", required=True)

    parser.add_argument("--input", dest="input",
                        help="Path to sloth annotations file with input train data", required=True)

    parser.add_argument("--output", dest="output",
                        help="Path to output sloth annotations file with augumented train data", required=True)

    parser.add_argument('-j', '--workers', default=default_workers, type=int,
                        help='number of data loading workers (default: %d)' % default_workers)

    parser.add_argument("-r", "--rseed", dest="random_seed",type=int,
                        help="Random seed, will use current time if none specified",
                        required=False, default = int(time.time()))

    options = parser.parse_args()

    run(options)


if __name__ == '__main__':
    main()
