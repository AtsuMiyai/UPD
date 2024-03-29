#!/bin/bash

bash scripts/inference/geminipro/iasd/original_standard.sh

bash scripts/inference/geminipro/iasd/base_standard.sh
bash scripts/inference/geminipro/iasd/base_iasd.sh

bash scripts/inference/geminipro/iasd/option_standard.sh
bash scripts/inference/geminipro/iasd/option_iasd.sh

bash scripts/inference/geminipro/iasd/instruction_standard.sh
bash scripts/inference/geminipro/iasd/instruction_iasd.sh
