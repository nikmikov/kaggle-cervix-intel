#!/usr/bin/env bash

ROOTDIR=$(pwd)

PYTHONPATH=$PYTHONPATH:$ROOTDIR python localizer/preprocess.py \
          --input=augdata/train_annotated.json \
          --output=output/train_localizer_aug.json \
          --target-image-size=256 \
          --workdir=tmp/run01 \
          $*
