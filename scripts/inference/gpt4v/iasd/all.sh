#!/bin/bash

bash scripts/inference/gpt4v/iasd/original_standard.sh

bash scripts/inference/gpt4v/iasd/base_standard.sh
bash scripts/inference/gpt4v/iasd/base_iasd.sh

bash scripts/inference/gpt4v/iasd/option_standard.sh
bash scripts/inference/gpt4v/iasd/option_iasd.sh

bash scripts/inference/gpt4v/iasd/instruction_standard.sh
bash scripts/inference/gpt4v/iasd/instruction_iasd.sh
