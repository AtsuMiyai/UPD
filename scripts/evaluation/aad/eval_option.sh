#!/bin/bash

RESULT_PATH=$1

UPD_DATA_NAME=mmaad_option

python automatic_eval/calculate_scores.py \
    --upd_type aad \
    --eval_file ${RESULT_PATH} \
    --meta_file ${DATA_NAME} \
    --openai_api_key ${OPENAI_API_KEY}
    
