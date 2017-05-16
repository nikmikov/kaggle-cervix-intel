#!/usr/bin/env bash

ROOTDIR=$(pwd)

BATCH_SIZE=128

PYTHONPATH=$PYTHONPATH:$ROOTDIR python localisation/main.py \
          --eval-input=augdata/test_annotated.json \
          --eval-output=output/test_localisation.json \
          --workdir=tmp/run01 \
          --batch-size=$BATCH_SIZE \
          --model-path=models/cervix_localisation.pth \
          $*
