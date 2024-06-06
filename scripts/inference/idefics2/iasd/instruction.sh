#!/bin/bash

DATA_NAME="mmiasd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="idefics2_${CURRENT_TIME}"

python -m vlms.idefics2.idefics2_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/idefics2/instruction/${DATA_NAME:0:-5}_instruction/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 2

mkdir -p output/iasd/answers_upload/idefics2/instruction/${DATA_NAME:0:-5}_instruction

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/idefics2/instruction/${DATA_NAME:0:-5}_instruction \
    --upload-dir ./output/iasd/answers_upload/idefics2/instruction/${DATA_NAME:0:-5}_instruction \
    --experiment ${FILE_NAME}