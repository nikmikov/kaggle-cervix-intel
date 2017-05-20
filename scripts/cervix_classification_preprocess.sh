#!/usr/bin/env bash

ROOTDIR=$(pwd)

PYTHONPATH=$PYTHONPATH:$ROOTDIR python classifier/preprocess.py \
          --input=augdata/test_annotated.json \
          --output=output/test_classifier_aug.json \
          --target-image-size=256 \
          --workdir=tmp/run01 \
          $*
