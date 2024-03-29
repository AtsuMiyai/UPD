#!/bin/bash

DATA_NAME="mmiasd_standard_option"
CURRENT_TIME=$(date +"%Y%m%d%H%M%S")
FILE_NAME="geminipro_${CURRENT_TIME}"

python -m vlms.geminipro.geminipro_vqa_updbench_w_before_file \
    --data-name ${DATA_NAME} \
    --answers-file ./output/iasd/answers/geminipro/option/${DATA_NAME}/${FILE_NAME}.jsonl \
    --answers-file-before output/aad/answers/geminipro/aad_bench_standard_20240303_w_option/geminipro_20240311032513.jsonl \
    --single-pred-prompt \
    --gemini-api-key ${GEMINI_API_KEY} \
    --prompt_id 1

mkdir -p output/iasd/answers_upload/geminipro/option/${DATA_NAME}

python scripts/convert_mmbench_for_submission.py \
    --data-name ${DATA_NAME} \
    --result-dir ./output/iasd/answers/geminipro/option/${DATA_NAME} \
    --upload-dir ./output/iasd/answers_upload/geminipro/option/${DATA_NAME} \
    --experiment ${FILE_NAME}