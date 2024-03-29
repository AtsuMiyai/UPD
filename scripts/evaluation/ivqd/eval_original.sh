#!/bin/bash

STANDARD_RESULT_PATH=$1

STANDARD_DATA_NAME=mmivqd_standard_base

python automatic_eval/calculate_scores.py \
    --upd_type ivqd \
    --eval_file_standard ${STANDARD_RESULT_PATH} \
    --meta_file_standard ${STANDARD_DATA_NAME} \
    --question_type original \
    --openai_api_key ${OPENAI_API_KEY}
