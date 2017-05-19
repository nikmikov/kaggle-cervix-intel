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


angles = {
    90  : Image.ROTATE_90,
    180 : Image.ROTATE_180,
    270 : Image.ROTATE_270
}

def get_crop_boxes(im):
    return [
        ( im.xmin, 0, 1, 1, "left" ),
        ( 0, 0, im.xmax, 1, "right" ),
        ( 0, im.ymin, 1, 1, "top" ),
        ( 0, 0, 1, im.ymax, "bottom")
    ]


def rotate(output_dir, basename, target_size, image, im):

    def rotate(angle):
        i = image
        m = im
        if angle != 0:
            i = image.transpose( angles[angle]  )
            m = m.with_rotate_coords(-angle)
        p = os.path.join(output_dir, basename + ("_%d" % angle) )
        return i,m,p

    res = []

    for a in [0,90,180,270]:
        i, m, p = rotate(a)
        p = os.path.abspath( p + ".jpg" )
        i = i.resize(target_size, Image.BILINEAR).save(p)
        res.append( m.with_filename(p).with_size(target_size))

    return res

def crop(output_dir, basename, target_size, image, im):
    res = []
    w, h = im.image_width, im.image_height
    r0 = min(w,h) / max(w,h)

    for l,t,r,b,n in get_crop_boxes(im):

        dx0, dy0 = l, t # new coord 0
        dxV, dyV = 1.0 / (r-l), 1.0/ (b-t) # scale
        i = image.crop( (l*w, t*h, r*w, b*h) )
        p = os.path.join(output_dir, basename + ("_%s" % n) )
        p = os.path.abspath( p + ".jpg" )
        m = im.with_coords( (im.xmin - dx0)*dxV, (im.ymin - dy0)*dyV, (im.xmax - dx0)*dxV, (im.ymax - dy0)*dyV  )
        w1,h1 = i.size
        r1 = min(w1,h1) / max(w1,h1)
        deformation = abs(r0-r1)
        if deformation < 0.1:
            i = i.resize(target_size, Image.BILINEAR).save(p)
            res.append( m.with_filename(p).with_size(target_size))

    return res

def flip_vertical(output_dir, basename, target_size, image, im):
    # vertical flip
    i = image.transpose( Image.FLIP_TOP_BOTTOM  )
    md = im.with_coords( im.xmin, 1 - im.ymax, im.xmax, 1 - im.ymin )
    p = os.path.join(output_dir, basename + "_fvert" )
    p = os.path.abspath( p + ".jpg" )
    i = i.resize(target_size, Image.BILINEAR).save(p)

    return md.with_filename(p).with_size(target_size)

def flip_horizontal(output_dir, basename, target_size, image, im):
    # horizontal flip
    i = image.transpose( Image.FLIP_LEFT_RIGHT  )
    md = im.with_coords( 1 - im.xmax, im.ymin, 1 - im.xmin, im.ymax )
    p = os.path.join(output_dir, basename + "_fhoriz" )
    p = os.path.abspath( p + ".jpg" )
    i = i.resize(target_size, Image.BILINEAR).save(p)

    return md.with_filename(p).with_size(target_size)


def process_image(output_dir, im, target_size):
    """
    (4 crops + 1 orig) Crop image by one of edges of the bounding box
        (3) Rotate image 90,180,270
        (1) Horizontal Flip
        (1) Vertical flip
        (1) Noop

    Summary: 5 * 6 = 30 images
    """
    if im.xmin is None:
        return []

    image = Image.open(im.filepath).convert('RGB')
    basename = str_to_md5(im.filepath)

    res = []
    res += rotate(output_dir, basename, target_size, image, im)
    res += crop(output_dir, basename, target_size, image, im)
    res += [ flip_vertical(output_dir, basename, target_size, image, im) ]
    res += [ flip_horizontal(output_dir, basename, target_size, image, im) ]
    return res

def run(options):
    work_dir = os.path.join(options.work_dir, "localizer", str(options.target_image_size) )
    create_dir_if_not_exists(work_dir)
    print(" + Random seed: %d" % options.random_seed)

    #load annotaions file
    alist = read_annotations(options.input)
    print(" + Input: %s. Images loaded: %d" % (options.input, len(alist)) )

    target_size = (options.target_image_size, options.target_image_size)
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
                        help="Target image size in pixels. Output images will be resized to SxS ", required=True)

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
