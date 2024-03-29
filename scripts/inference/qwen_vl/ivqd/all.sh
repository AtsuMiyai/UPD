#!/bin/bash

bash scripts/inference/qwen_vl/ivqd/original_standard.sh

bash scripts/inference/qwen_vl/ivqd/base_standard.sh
bash scripts/inference/qwen_vl/ivqd/base_ivqd.sh

bash scripts/inference/qwen_vl/ivqd/option_standard.sh
bash scripts/inference/qwen_vl/ivqd/option_ivqd.sh

bash scripts/inference/qwen_vl/ivqd/instruction_standard.sh
bash scripts/inference/qwen_vl/ivqd/instruction_ivqd.sh
