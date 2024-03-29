#!/bin/bash

bash scripts/inference/llava1.5/ivqd/original_standard.sh

bash scripts/inference/llava1.5/ivqd/base_standard.sh
bash scripts/inference/llava1.5/ivqd/base_ivqd.sh

bash scripts/inference/llava1.5/ivqd/option_standard.sh
bash scripts/inference/llava1.5/ivqd/option_ivqd.sh

bash scripts/inference/llava1.5/ivqd/instruction_standard.sh
bash scripts/inference/llava1.5/ivqd/instruction_ivqd.sh
