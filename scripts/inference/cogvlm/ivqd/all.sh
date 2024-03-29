#!/bin/bash

bash scripts/inference/cogvlm/ivqd/original_standard.sh

bash scripts/inference/cogvlm/ivqd/base_standard.sh
bash scripts/inference/cogvlm/ivqd/base_ivqd.sh

bash scripts/inference/cogvlm/ivqd/option_standard.sh
bash scripts/inference/cogvlm/ivqd/option_ivqd.sh

bash scripts/inference/cogvlm/ivqd/instruction_standard.sh
bash scripts/inference/cogvlm/ivqd/instruction_ivqd.sh
