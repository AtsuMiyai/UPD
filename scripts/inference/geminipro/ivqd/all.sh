#!/bin/bash

bash scripts/inference/geminipro/ivqd/original_standard.sh

bash scripts/inference/geminipro/ivqd/base_standard.sh
bash scripts/inference/geminipro/ivqd/base_ivqd.sh

bash scripts/inference/geminipro/ivqd/option_standard.sh
bash scripts/inference/geminipro/ivqd/option_ivqd.sh

bash scripts/inference/geminipro/ivqd/instruction_standard.sh
bash scripts/inference/geminipro/ivqd/instruction_ivqd.sh
