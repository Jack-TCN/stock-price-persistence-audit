#!/usr/bin/env bash
set -euo pipefail
# Required: --data-run SNAPSHOT --out-dir NEW_RUN --model-revision IMMUTABLE_SHA
python "$(dirname "$0")/run_all_reproduce.py" "$@"
