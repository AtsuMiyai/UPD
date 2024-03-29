#!/bin/bash

bash scripts/inference/gpt4v/aad/original_standard.sh

bash scripts/inference/gpt4v/aad/base_standard.sh
bash scripts/inference/gpt4v/aad/base_aad.sh

bash scripts/inference/gpt4v/aad/option_standard.sh
bash scripts/inference/gpt4v/aad/option_aad.sh

bash scripts/inference/gpt4v/aad/instruction_standard.sh
bash scripts/inference/gpt4v/aad/instruction_aad.sh

