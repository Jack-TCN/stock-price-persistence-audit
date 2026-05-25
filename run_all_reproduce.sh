#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after installing the dependencies in
# requirements_submission.txt. These commands are a conservative reproduction
# sketch; exact runtime depends on hardware and whether Chronos is run on CPU
# or GPU.

python code/scripts/run_cross_market_data_audit.py
python code/scripts/run_cross_market_model_audit.py
python code/scripts/run_chronos_zero_shot_audit.py
python code/scripts/build_cross_market_evidence_tables.py
python code/scripts/build_cross_market_figures.py
python code/scripts/build_cross_market_supplement_tables.py
python code/scripts/build_split_appendix.py
python verify_results.py
