#!/bin/bash

DATA_NAME="mmiasd_iasd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="qwenvl_chat_${CURRENT_TIME}"

python -m vlms.qwen_vl.qwen_vl_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/qwenvl_chat/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 0

mkdir -p output/iasd/answers_upload/qwenvl_chat/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/qwenvl_chat/base/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/qwenvl_chat/base/${DATA_NAME} \
    --experiment ${FILE_NAME}