#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <data-root> <dataset> <model>"
    exit 1
fi

DATA=$1
DATASET=$2
MODEL=$3
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
SHOTS=16
NCTX=4
CSC=False
CTP=end
SUB_BASE=base
SUB_NOVEL=new

for SEED in 1 2 3; do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    TRAIN_DIR=output/base2new/train_${SUB_BASE}/${COMMON_DIR}
    if [[ ! -d "${TRAIN_DIR}" ]]; then
        python train.py \
            --root "${DATA}" \
            --seed "${SEED}" \
            --trainer "${TRAINER}" \
            --dataset-config-file "configs/datasets/${DATASET}.yaml" \
            --config-file "configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml" \
            --output-dir "${TRAIN_DIR}" \
            TRAINER.BIOMEDCOOP.N_CTX "${NCTX}" \
            TRAINER.BIOMEDCOOP.CSC "${CSC}" \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION "${CTP}" \
            DATASET.NUM_SHOTS "${SHOTS}" \
            DATASET.SUBSAMPLE_CLASSES "${SUB_BASE}"
    else
        echo "Results already exist at ${TRAIN_DIR}; skipping base training for seed ${SEED}."
    fi

    MODEL_DIR=${TRAIN_DIR}
    TEST_DIR=output/base2new/test_${SUB_NOVEL}/${COMMON_DIR}
    if [[ ! -d "${TEST_DIR}" ]]; then
        python train.py \
            --root "${DATA}" \
            --seed "${SEED}" \
            --trainer "${TRAINER}" \
            --dataset-config-file "configs/datasets/${DATASET}.yaml" \
            --config-file "configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml" \
            --output-dir "${TEST_DIR}" \
            --model-dir "${MODEL_DIR}" \
            --eval-only \
            TRAINER.BIOMEDCOOP.N_CTX "${NCTX}" \
            TRAINER.BIOMEDCOOP.CSC "${CSC}" \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION "${CTP}" \
            DATASET.NUM_SHOTS "${SHOTS}" \
            DATASET.SUBSAMPLE_CLASSES "${SUB_NOVEL}"
    else
        echo "Results already exist at ${TEST_DIR}; skipping novel evaluation for seed ${SEED}."
    fi
done
