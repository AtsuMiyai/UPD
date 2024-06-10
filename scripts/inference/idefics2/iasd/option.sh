#!/bin/bash

DATA_NAME="mmiasd_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="idefics2_${CURRENT_TIME}"

python -m vlms.idefics2.idefics2_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/idefics2/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 1

mkdir -p output/iasd/answers_upload/idefics2/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/idefics2/option/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/idefics2/option/${DATA_NAME} \
    --experiment ${FILE_NAME}