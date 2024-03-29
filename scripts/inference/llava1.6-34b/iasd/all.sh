#!/bin/bash

bash scripts/inference/llava1.6-34b/iasd/original_standard.sh

bash scripts/inference/llava1.6-34b/iasd/base_standard.sh
bash scripts/inference/llava1.6-34b/iasd/base_iasd.sh

bash scripts/inference/llava1.6-34b/iasd/option_standard.sh
bash scripts/inference/llava1.6-34b/iasd/option_iasd.sh

bash scripts/inference/llava1.6-34b/iasd/instruction_standard.sh
bash scripts/inference/llava1.6-34b/iasd/instruction_iasd.sh

bash scripts/inference/llava1.6-34b/iasd/tuning_iasd.sh
bash scripts/inference/llava1.6-34b/iasd/tuning_standard.sh