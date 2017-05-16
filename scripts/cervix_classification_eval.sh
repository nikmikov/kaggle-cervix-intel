#!/usr/bin/env bash

ROOTDIR=$(pwd)

BATCH_SIZE=64

PYTHONPATH=$PYTHONPATH:$ROOTDIR python classifier/main.py \
          --eval-input=augdata/test_annotated.json \
          --eval-output=output/result.csv \
          --workdir=tmp/run01 \
          --batch-size=$BATCH_SIZE \
          --model-path=models/cervix_classifier.pth \
          $*
