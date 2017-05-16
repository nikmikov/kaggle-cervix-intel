#!/usr/bin/env bash

ROOTDIR=$(pwd)

TRAIN_EPOCHS=300
BATCH_SIZE=128
VALIDATION_SPLIT=0.9

PYTHONPATH=$PYTHONPATH:$ROOTDIR python classifier/main.py \
          --train-input=augdata/train_annotated.json \
          --workdir=tmp/run01 \
          --train-epochs=$TRAIN_EPOCHS \
          --batch-size=$BATCH_SIZE \
          --validation-split=$VALIDATION_SPLIT \
          --model-path=models/cervix_classifier.pth \
          $*
