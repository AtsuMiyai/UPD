#!/bin/bash

DATA_NAME="mmivqd_ivqd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="cogvlm-17b_${CURRENT_TIME}"

python -m vlms.cogvlms.cogvlm_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/ivqd/answers/cogvlm/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 0

mkdir -p output/ivqd/answers_upload/cogvlm/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/ivqd/answers/cogvlm/base/${DATA_NAME} \
    --upload-dir ./output/ivqd/answers_upload/cogvlm/base/${DATA_NAME} \
    --experiment ${FILE_NAME}