#!/bin/bash

DATA_NAME="mmiasd_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="cogvlm2-19b_${CURRENT_TIME}"

python -m vlms.cogvlm2.cogvlm2_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/cogvlm2/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 1

mkdir -p output/iasd/answers_upload/cogvlm2/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/cogvlm2/option/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/cogvlm2/option/${DATA_NAME} \
    --experiment ${FILE_NAME}