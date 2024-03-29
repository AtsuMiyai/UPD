#!/bin/bash

DATA_NAME="mmivqd_ivqd_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="gpt4v_${CURRENT_TIME}"

python -m vlms.gpt4v.gpt4v_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/ivqd/answers/gpt4v/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --openai-api-key ${OPENAI_API_KEY} \
    --prompt_id 0

mkdir -p output/ivqd/answers_upload/gpt4v/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/ivqd/answers/gpt4v/base/${DATA_NAME} \
    --upload-dir ./output/ivqd/answers_upload/gpt4v/base/${DATA_NAME} \
    --experiment ${FILE_NAME}