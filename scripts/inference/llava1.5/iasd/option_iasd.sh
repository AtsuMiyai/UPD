#!/bin/bash

DATA_NAME="mmiasd_iasd_option"
TEMP=0.0
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="llava-v1.5-13b_${CURRENT_TIME}"

export PYTHONPATH="$PYTHONPATH:$PWD/vlms"

python -m vlms.llava.eval.llava_vqa_updbench \
    --model-path liuhaotian/llava-v1.5-13b \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/llava1.5/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --temperature ${TEMP} \
    --prompt_id 1 \
    --conv-mode vicuna_v1

mkdir -p output/iasd/answers_upload/llava1.5/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/llava1.5/option/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/llava1.5/option/${DATA_NAME} \
    --experiment ${FILE_NAME}