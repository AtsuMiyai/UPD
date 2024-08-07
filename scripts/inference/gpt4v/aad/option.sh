#!/bin/bash

DATA_NAME="mmaad_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="gpt4v_${CURRENT_TIME}"

python -m vlms.gpt4v.gpt4v_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/aad/answers/gpt4v/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --openai-api-key ${OPENAI_API_KEY} \
    --prompt_id 1

mkdir -p ./output/aad/answers_upload/gpt4v/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/aad/answers/gpt4v/option/${DATA_NAME} \
    --upload-dir ./output/aad/answers_upload/gpt4v/option/${DATA_NAME} \
    --experiment ${FILE_NAME}
