#!/bin/bash

python train.py \
  --gpu \
  --save \
  --wandb

exit $?
