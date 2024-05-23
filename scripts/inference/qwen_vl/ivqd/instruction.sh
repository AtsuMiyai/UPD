#!/bin/bash

DATA_NAME="mmivqd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="qwenvl_chat_${CURRENT_TIME}"

python -m vlms.qwen_vl.qwen_vl_vqa_updbench \
    --data-name ${DATA_NAME}\
    --answers-file ./output/ivqd/answers/qwenvl_chat/instruction/${DATA_NAME:0:-5}_instruction/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 3

mkdir -p output/ivqd/answers_upload/qwenvl_chat/instruction/${DATA_NAME:0:-5}_instruction

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME}\
    --result-dir ./output/ivqd/answers/qwenvl_chat/instruction/${DATA_NAME:0:-5}_instruction \
    --upload-dir ./output/ivqd/answers_upload/qwenvl_chat/instruction/${DATA_NAME:0:-5}_instruction \
    --experiment ${FILE_NAME}