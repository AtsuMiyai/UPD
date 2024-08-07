#!/bin/bash

DATA_NAME="mmiasd_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="gpt4o-mini_${CURRENT_TIME}"

python -m vlms.gpt4v.gpt4v_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/gpt4o-mini/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --model_name gpt-4o-mini \
    --openai-api-key ${OPENAI_API_KEY} \
    --prompt_id 1

mkdir -p output/iasd/answers_upload/gpt4o-mini/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/gpt4o-mini/option/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/gpt4o-mini/option/${DATA_NAME} \
    --experiment ${FILE_NAME}