#!/bin/bash

DATA_NAME="mmaad_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="gpt4o_${CURRENT_TIME}"

python -m vlms.gpt4v.gpt4v_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/aad/answers/gpt4o/instruction/${DATA_NAME:0:-5}_instruction/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --model_name gpt-4o \
    --openai-api-key ${OPENAI_API_KEY} \
    --prompt_id 2

mkdir -p ./output/aad/answers_upload/gpt4o/instruction/${DATA_NAME:0:-5}_instruction

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/aad/answers/gpt4o/instruction/${DATA_NAME:0:-5}_instruction \
    --upload-dir ./output/aad/answers_upload/gpt4o/instruction/${DATA_NAME:0:-5}_instruction \
    --experiment ${FILE_NAME}
