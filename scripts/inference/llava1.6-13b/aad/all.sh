#!/bin/bash

bash scripts/inference/llava1.6-13b/aad/original_standard.sh

bash scripts/inference/llava1.6-13b/aad/base_standard.sh
bash scripts/inference/llava1.6-13b/aad/base_aad.sh

bash scripts/inference/llava1.6-13b/aad/option_standard.sh
bash scripts/inference/llava1.6-13b/aad/option_aad.sh

bash scripts/inference/llava1.6-13b/aad/instruction_standard.sh
bash scripts/inference/llava1.6-13b/aad/instruction_aad.sh

bash scripts/inference/llava1.6-13b/aad/tuning_aad.sh
bash scripts/inference/llava1.6-13b/aad/tuning_standard.sh