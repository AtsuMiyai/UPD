#!/bin/bash

DATA_NAME="mmivqd_base"
TEMP=0.0
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="llava-v1.6-13b_${CURRENT_TIME}"

export PYTHONPATH="$PYTHONPATH:$PWD/vlms"

python -m vlms.llava.eval.llava_vqa_updbench \
    --model-path liuhaotian/llava-v1.6-vicuna-13b \
    --data-name ${DATA_NAME}\
    --answers-file ./output/ivqd/answers/llava1.6_13b/original/${DATA_NAME:0:-5}_original/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --temperature ${TEMP} \
    --prompt_id 1 \
    --conv-mode vicuna_v1

mkdir -p output/ivqd/answers_upload/llava1.6_13b/original/${DATA_NAME:0:-5}_original

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME}\
    --result-dir ./output/ivqd/answers/llava1.6_13b/original/${DATA_NAME:0:-5}_original \
    --upload-dir ./output/ivqd/answers_upload/llava1.6_13b/original/${DATA_NAME:0:-5}_original \
    --experiment ${FILE_NAME}