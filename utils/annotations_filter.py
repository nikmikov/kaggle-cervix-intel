#!/usr/bin/env python3
from __future__ import division, absolute_import, print_function

import os
import os.path
import sys
import argparse
import json
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations", dest="annotations_path",
                        help="Data directory", required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help="Data directory", required=True)


    options = parser.parse_args()

    filter_val = 2085
    l = []
    for fname in options.annotations_path.split(' '):
        with open(fname) as json_data:
            d = json.load(json_data)
            for elem in d:
                filepath = elem['filename']
                annotations = elem['annotations']
                if len(annotations) != 1:
                    continue
                bname,_ = os.path.basename(filepath).split('.')
                if int(bname) <= filter_val:
                    l.append(elem)

    s = json.dumps(l)
    with open(options.output, "w") as text_file:
        text_file.write(s)

if __name__ == '__main__':
    main()
