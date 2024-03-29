#!/bin/bash

DATA_NAME="mmiasd_standard_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="qwenvl_chat_${CURRENT_TIME}"

python -m vlms.qwen_vl.qwen_vl_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/qwenvl_chat/original/${DATA_NAME:0:-5}_original/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 1

mkdir -p output/iasd/answers_upload/qwenvl_chat/original/${DATA_NAME:0:-5}_original

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/qwenvl_chat/original/${DATA_NAME:0:-5}_original \
    --upload-dir ./output/iasd/answers_upload/qwenvl_chat/original/${DATA_NAME:0:-5}_original \
    --experiment ${FILE_NAME}