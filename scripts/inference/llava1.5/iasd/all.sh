#!/bin/bash

bash scripts/inference/llava1.5/iasd/original_standard.sh

bash scripts/inference/llava1.5/iasd/base_standard.sh
bash scripts/inference/llava1.5/iasd/base_iasd.sh

bash scripts/inference/llava1.5/iasd/option_standard.sh
bash scripts/inference/llava1.5/iasd/option_iasd.sh

bash scripts/inference/llava1.5/iasd/instruction_standard.sh
bash scripts/inference/llava1.5/iasd/instruction_iasd.sh
