#!/bin/bash

DATA_NAME="mmivqd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="idefics2_${CURRENT_TIME}"

python -m vlms.idefics2.idefics2_vqa_updbench \
    --data-name ${DATA_NAME}\
    --answers-file ./output/ivqd/answers/idefics2/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 0

mkdir -p output/ivqd/answers_upload/idefics2/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME}\
    --result-dir ./output/ivqd/answers/idefics2/base/${DATA_NAME} \
    --upload-dir ./output/ivqd/answers_upload/idefics2/base/${DATA_NAME} \
    --experiment ${FILE_NAME}