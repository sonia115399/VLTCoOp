#!/bin/bash

# usage:
# bash scripts/zeroshot/zeroshot_selective.sh /path/to/data corn

CFG=vit_b16
DATA=$1
DATASET=$2

METHOD=Zeroshot
TRAINER=ZeroshotCLIPSelective

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${METHOD}/${CFG}.yaml \
--output-dir output/${DATASET}/${TRAINER}/${CFG} \
--eval-only
