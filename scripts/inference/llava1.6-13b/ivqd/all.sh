#!/bin/bash

bash scripts/inference/llava1.6-13b/ivqd/original_standard.sh

bash scripts/inference/llava1.6-13b/ivqd/base_standard.sh
bash scripts/inference/llava1.6-13b/ivqd/base_ivqd.sh

bash scripts/inference/llava1.6-13b/ivqd/option_standard.sh
bash scripts/inference/llava1.6-13b/ivqd/option_ivqd.sh

bash scripts/inference/llava1.6-13b/ivqd/instruction_standard.sh
bash scripts/inference/llava1.6-13b/ivqd/instruction_ivqd.sh

bash scripts/inference/llava1.6-13b/ivqd/tuning_ivqd.sh
bash scripts/inference/llava1.6-13b/ivqd/tuning_standard.sh