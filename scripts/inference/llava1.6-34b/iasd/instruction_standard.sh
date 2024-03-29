#!/bin/bash

DATA_NAME="mmiasd_standard_base"
TEMP=0.0
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="llava-v1.6-34b_${CURRENT_TIME}"

export PYTHONPATH="$PYTHONPATH:$PWD/vlms"

python -m vlms.llava.eval.llava_vqa_updbench \
    --model-path liuhaotian/llava-v1.6-34b \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/llava1.6_34b/instruction/${DATA_NAME:0:-5}_instruction/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --temperature ${TEMP} \
    --prompt_id 2 \
    --conv-mode mistral_direct

mkdir -p output/iasd/answers_upload/llava1.6_34b/instruction/${DATA_NAME:0:-5}_instruction

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/llava1.6_34b/instruction/${DATA_NAME:0:-5}_instruction \
    --upload-dir ./output/iasd/answers_upload/llava1.6_34b/instruction/${DATA_NAME:0:-5}_instruction \
    --experiment ${FILE_NAME}