#!/bin/bash

bash scripts/inference/llava1.6-34b/aad/original_standard.sh

bash scripts/inference/llava1.6-34b/aad/base_standard.sh
bash scripts/inference/llava1.6-34b/aad/base_aad.sh

bash scripts/inference/llava1.6-34b/aad/option_standard.sh
bash scripts/inference/llava1.6-34b/aad/option_aad.sh

bash scripts/inference/llava1.6-34b/aad/instruction_standard.sh
bash scripts/inference/llava1.6-34b/aad/instruction_aad.sh

bash scripts/inference/llava1.6-34b/aad/tuning_standard.sh
bash scripts/inference/llava1.6-34b/aad/tuning_aad.sh


