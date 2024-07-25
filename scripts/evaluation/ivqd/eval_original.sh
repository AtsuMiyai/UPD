#!/bin/bash

RESULT_PATH=$1

DATA_NAME=mmivqd_base

python automatic_eval/calculate_scores.py \
    --upd_type ivqd \
    --eval_file ${RESULT_PATH} \
    --meta_file ${DATA_NAME} \
    --question_type original \
    --openai_api_key ${OPENAI_API_KEY}
