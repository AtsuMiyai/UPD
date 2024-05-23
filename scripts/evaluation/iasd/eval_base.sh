#!/bin/bash

RESULT_PATH=$1

DATA_NAME=mmiasd_base

python automatic_eval/calculate_scores.py \
    --upd_type iasd \
    --eval_file ${RESULT_PATH} \
    --meta_file ${DATA_NAME} \
    --openai_api_key ${OPENAI_API_KEY}
