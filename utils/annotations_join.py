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

    l = []
    for fname in options.annotations_path.split(' '):
        with open(fname) as json_data:
            d = json.load(json_data)
            print("File: %s : %d" % (fname, len(d)))
            l += d

    s = json.dumps(l)
    with open(options.output, "w") as text_file:
        text_file.write(s)

if __name__ == '__main__':
    main()
