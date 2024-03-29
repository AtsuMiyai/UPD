#!/bin/bash

UPD_RESULT_PATH=$1
STANDARD_RESULT_PATH=$2

UPD_DATA_NAME=mmaad_aad_base
STANDARD_DATA_NAME=mmaad_standard_base

python automatic_eval/calculate_scores.py \
    --upd_type aad \
    --eval_file_upd ${UPD_RESULT_PATH} \
    --eval_file_standard ${STANDARD_RESULT_PATH} \
    --meta_file_upd ${UPD_DATA_NAME} \
    --meta_file_standard ${STANDARD_DATA_NAME} \
    --openai_api_key ${OPENAI_API_KEY} \
