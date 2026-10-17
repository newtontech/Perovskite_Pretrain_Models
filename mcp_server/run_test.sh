#!/bin/bash

# Test environment variables
export BOHRIUM_BOHRIUM_URL="https://bohrium.test.dp.tech"
export BOHRIUM_TIEFBLUE_URL="https://tiefblue.test.dp.tech"
export BOHRIUM_OPENAPI_URL="https://openapi.test.dp.tech"
export BOHRIUM_BASE_URL="https://openapi.test.dp.tech"
export TIEFBLUE_BASE_URL="https://tiefblue-nas-acs-bj.test.bohrium.com"
export BOHRIUM_USE_SANDBOX=1

export PEROVSKITE_WEIGHT_DIR="${PEROVSKITE_WEIGHT_DIR:-/unimol_weight_perovskite/fixed_hyper_param}"

cd "$(cd "$(dirname "$0")" && pwd)"
export UNIMOL_WEIGHT_DIR=/opt/conda/lib/python3.8/site-packages/unimol_tools/weights

# Run server with test environment parameters
python server.py --host 0.0.0.0 --port 50001
