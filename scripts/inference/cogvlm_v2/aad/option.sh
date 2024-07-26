#!/bin/bash

DATA_NAME="mmaad_standard_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="cogvlm_v2-17b_${CURRENT_TIME}"

python -m vlms.cogvlm_v2.cogvlm_v2_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/aad/answers/cogvlm_v2/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --prompt_id 1

mkdir -p ./output/aad/answers_upload/cogvlm_v2/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/aad/answers/cogvlm_v2/option/${DATA_NAME} \
    --upload-dir ./output/aad/answers_upload/cogvlm_v2/option/${DATA_NAME} \
    --experiment ${FILE_NAME}