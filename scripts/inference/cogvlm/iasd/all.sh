#!/bin/bash

bash scripts/inference/cogvlm/iasd/original_standard.sh

bash scripts/inference/cogvlm/iasd/base_standard.sh
bash scripts/inference/cogvlm/iasd/base_iasd.sh

bash scripts/inference/cogvlm/iasd/option_standard.sh
bash scripts/inference/cogvlm/iasd/option_iasd.sh

bash scripts/inference/cogvlm/iasd/instruction_standard.sh
bash scripts/inference/cogvlm/iasd/instruction_iasd.sh
