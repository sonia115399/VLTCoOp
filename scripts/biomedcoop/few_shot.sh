#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <data-root> <dataset> <shots> <model>"
    exit 1
fi

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4
METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}
NCTX=4
CSC=False
CTP=end

for SEED in 1 2 3; do
    DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [[ -d "${DIR}" ]]; then
        echo "Results already exist at ${DIR}; skipping seed ${SEED}."
        continue
    fi

    python train.py \
        --root "${DATA}" \
        --seed "${SEED}" \
        --trainer "${TRAINER}" \
        --dataset-config-file "configs/datasets/${DATASET}.yaml" \
        --config-file "configs/trainers/${METHOD}/few_shot/${DATASET}.yaml" \
        --output-dir "${DIR}" \
        TRAINER.BIOMEDCOOP.N_CTX "${NCTX}" \
        TRAINER.BIOMEDCOOP.CSC "${CSC}" \
        TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION "${CTP}" \
        DATASET.NUM_SHOTS "${SHOTS}"
done
