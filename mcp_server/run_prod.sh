#!/bin/bash

# Production environment variables
export PEROVSKITE_WEIGHT_DIR="${PEROVSKITE_WEIGHT_DIR:-/unimol_weight_perovskite/fixed_hyper_param}"
export TIEFBLUE_BASE_URL=https://tiefblue-nas-acs-bj.bohrium.com
export BOHRIUM_USE_SANDBOX=1

cd "$(cd "$(dirname "$0")" && pwd)"
export UNIMOL_WEIGHT_DIR=/opt/conda/lib/python3.8/site-packages/unimol_tools/weights

# Run server with production environment parameters
# You can customize host and port as needed for production
python server.py --host 0.0.0.0 --port 50002
