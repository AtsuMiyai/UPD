#!/bin/bash

bash scripts/inference/qwen_vl/aad/original_standard.sh

bash scripts/inference/qwen_vl/aad/base_standard.sh
bash scripts/inference/qwen_vl/aad/base_aad.sh

bash scripts/inference/qwen_vl/aad/option_standard.sh
bash scripts/inference/qwen_vl/aad/option_aad.sh

bash scripts/inference/qwen_vl/aad/instruction_standard.sh
bash scripts/inference/qwen_vl/aad/instruction_aad.sh