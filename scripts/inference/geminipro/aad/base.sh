#!/bin/bash

DATA_NAME="mmaad_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="geminipro_${CURRENT_TIME}"

python -m vlms.geminipro.geminipro_vqa_updbench \
    --data-name ${DATA_NAME} \
    --answers-file ./output/aad/answers/geminipro/base/${DATA_NAME}/${FILE_NAME}.jsonl \
    --single-pred-prompt \
    --gemini-api-key ${GEMINI_API_KEY} \
    --prompt_id 0

mkdir -p output/aad/answers_upload/geminipro/base/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/aad/answers/geminipro/base/${DATA_NAME} \
    --upload-dir ./output/aad/answers_upload/geminipro/base/${DATA_NAME} \
    --experiment ${FILE_NAME}
