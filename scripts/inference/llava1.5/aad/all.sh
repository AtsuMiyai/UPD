#!/bin/bash

bash scripts/inference/llava1.5/aad/original_standard.sh

bash scripts/inference/llava1.5/aad/base_standard.sh
bash scripts/inference/llava1.5/aad/base_aad.sh

bash scripts/inference/llava1.5/aad/option_standard.sh
bash scripts/inference/llava1.5/aad/option_aad.sh

bash scripts/inference/llava1.5/aad/instruction_standard.sh
bash scripts/inference/llava1.5/aad/instruction_aad.sh

