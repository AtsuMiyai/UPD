#!/bin/bash

bash scripts/inference/qwen_vl/iasd/original_standard.sh

bash scripts/inference/qwen_vl/iasd/base_standard.sh
bash scripts/inference/qwen_vl/iasd/base_iasd.sh

bash scripts/inference/qwen_vl/iasd/option_standard.sh
bash scripts/inference/qwen_vl/iasd/option_iasd.sh

bash scripts/inference/qwen_vl/iasd/instruction_standard.sh
bash scripts/inference/qwen_vl/iasd/instruction_iasd.sh
