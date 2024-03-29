#!/bin/bash

bash scripts/inference/cogvlm/aad/original_standard.sh

bash scripts/inference/cogvlm/aad/base_standard.sh
bash scripts/inference/cogvlm/aad/base_aad.sh

bash scripts/inference/cogvlm/aad/option_standard.sh
bash scripts/inference/cogvlm/aad/option_aad.sh

bash scripts/inference/cogvlm/aad/instruction_standard.sh
bash scripts/inference/cogvlm/aad/instruction_aad.sh

