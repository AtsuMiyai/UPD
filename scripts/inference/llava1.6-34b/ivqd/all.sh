#!/bin/bash

bash scripts/inference/llava1.6-34b/ivqd/original_standard.sh

bash scripts/inference/llava1.6-34b/ivqd/base_standard.sh
bash scripts/inference/llava1.6-34b/ivqd/base_ivqd.sh

bash scripts/inference/llava1.6-34b/ivqd/option_standard.sh
bash scripts/inference/llava1.6-34b/ivqd/option_ivqd.sh

bash scripts/inference/llava1.6-34b/ivqd/instruction_standard.sh
bash scripts/inference/llava1.6-34b/ivqd/instruction_ivqd.sh

bash scripts/inference/llava1.6-34b/ivqd/tuning_standard.sh
bash scripts/inference/llava1.6-34b/ivqd/tuning_ivqd.sh