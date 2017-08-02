#!/usr/bin/env bash

ROOTDIR=$(pwd)

TRAIN_EPOCHS=50
BATCH_SIZE=64
NUM_CLUSTERS=64

PYTHONPATH=$PYTHONPATH:$ROOTDIR python cluster/main.py \
          --train-input=augdata/train_annotated.json \
          --workdir=tmp/run01 \
          --train-epochs=$TRAIN_EPOCHS \
          --batch-size=$BATCH_SIZE \
          --clusters-per-type=$NUM_CLUSTERS \
          --model-path=models/cervix_cluster.pth \
          $*
