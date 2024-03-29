#!/bin/bash

DATA_NAME="mmiasd_standard_base"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="geminipro_${CURRENT_TIME}"

python -m vlms.geminipro.geminipro_vqa_updbench_w_before_file \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/geminipro/instruction/${DATA_NAME:0:-5}_instruction/${FILE_NAME}.jsonl \
    --answers-file-before output/aad/answers/geminipro/aad_bench_standard_20240303_w_instruction/geminipro_20240310152030.jsonl \
    --single-pred-prompt \
    --gemini-api-key ${GEMINI_API_KEY} \
    --prompt_id 2

mkdir -p output/iasd/answers_upload/geminipro/instruction/${DATA_NAME:0:-5}_instruction

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/geminipro/instruction/${DATA_NAME:0:-5}_instruction \
    --upload-dir ./output/iasd/answers_upload/geminipro/instruction/${DATA_NAME:0:-5}_instruction \
    --experiment ${FILE_NAME}