#!/bin/bash

DATA_NAME="mmaad_standard_base"
TEMP=0.0
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="llava-v1.6-34b_${CURRENT_TIME}"

export PYTHONPATH="$PYTHONPATH:$PWD/vlms"

python -m vlms.llava.eval.llava_vqa_updbench \
    --model-path liuhaotian/llava-v1.6-34b \
    --data-name ${DATA_NAME} \
    --answers-file ./output/aad/answers/llava1.6_34b/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --temperature ${TEMP} \
    --prompt_id 0 \
    --conv-mode mistral_direct

mkdir -p output/aad/answers_upload/llava1.6_34b/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/aad/answers/llava1.6_34b/base/${DATA_NAME} \
    --upload-dir ./output/aad/answers_upload/llava1.6_34b/base/${DATA_NAME} \
    --experiment ${FILE_NAME}
