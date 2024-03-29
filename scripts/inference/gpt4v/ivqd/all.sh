#!/bin/bash

bash scripts/inference/gpt4v/ivqd/original_standard.sh

bash scripts/inference/gpt4v/ivqd/base_standard.sh
bash scripts/inference/gpt4v/ivqd/base_ivqd.sh

bash scripts/inference/gpt4v/ivqd/option_standard.sh
bash scripts/inference/gpt4v/ivqd/option_ivqd.sh

bash scripts/inference/gpt4v/ivqd/instruction_standard.sh
bash scripts/inference/gpt4v/ivqd/instruction_ivqd.sh
