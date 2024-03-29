#!/bin/bash

bash scripts/inference/geminipro/aad/original_standard.sh

bash scripts/inference/geminipro/aad/base_standard.sh
bash scripts/inference/geminipro/aad/base_aad.sh

bash scripts/inference/geminipro/aad/option_standard.sh
bash scripts/inference/geminipro/aad/option_aad.sh

bash scripts/inference/geminipro/aad/instruction_standard.sh
bash scripts/inference/geminipro/aad/instruction_aad.sh

