#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function

import os
import os.path
import sys
import argparse
import json
import glob

def annotations_find(annotations, directory):
    """
    Given annotations file and data directory will create
    sloth annotations file with images missing in annotations
    """

    alist = []
    with open(annotations) as json_data:
        d = json.load(json_data)
        for elem in d:
            filepath = elem['filename']
            assert(os.path.isfile(filepath))
            annotations = elem['annotations']
            if len(annotations) == 1:
                alist.append(filepath)
    alist= set(alist)

    images = set([ f for f in glob.glob(directory + '/**/*.jpg', recursive=True)])


    missing = set(images -  alist )
    for m in missing:
        if m in alist:
            print(m)

    print("Should be empty:",list(set(alist) - set(images)) )

    print("Images: %d, annotated: %d, missing:%d" % (len(images), len(alist), len(missing)) )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations", dest="annotations_path",
                        help="Data directory", required=True)
    parser.add_argument("-d", "--data-dir", dest="data_dir",
                        help="Data directory", required=True)

    options = parser.parse_args()
    annotations_find(options.annotations_path, options.data_dir)

if __name__ == '__main__':
    main()
