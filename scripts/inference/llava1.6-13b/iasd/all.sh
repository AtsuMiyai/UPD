#!/bin/bash

bash scripts/inference/llava1.6-13b/iasd/original_standard.sh

bash scripts/inference/llava1.6-13b/iasd/base_standard.sh
bash scripts/inference/llava1.6-13b/iasd/base_iasd.sh

bash scripts/inference/llava1.6-13b/iasd/option_standard.sh
bash scripts/inference/llava1.6-13b/iasd/option_iasd.sh

bash scripts/inference/llava1.6-13b/iasd/instruction_standard.sh
bash scripts/inference/llava1.6-13b/iasd/instruction_iasd.sh

bash scripts/inference/llava1.6-13b/iasd/tuning_iasd.sh
bash scripts/inference/llava1.6-13b/iasd/tuning_standard.sh