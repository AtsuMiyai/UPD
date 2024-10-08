#!/bin/bash

DATA_NAME="mmivqd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="cogvlm2-19b_${CURRENT_TIME}"

python -m vlms.cogvlm2.cogvlm2_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/ivqd/answers/cogvlm2/original/${DATA_NAME:0:-5}_original/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 1

mkdir -p output/ivqd/answers_upload/cogvlm2/original/${DATA_NAME:0:-5}_original

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/ivqd/answers/cogvlm2/original/${DATA_NAME:0:-5}_original \
    --upload-dir ./output/ivqd/answers_upload/cogvlm2/original/${DATA_NAME:0:-5}_original \
    --experiment ${FILE_NAME}