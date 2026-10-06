#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python}"
"$PYTHON_BIN" setup_cpp.py build_ext --inplace
"$PYTHON_BIN" -c 'from a3dizhu_cpp import VectorizedEngine; assert hasattr(VectorizedEngine, "advance_to_decision_with_actions"); print("Native A3 backend ready")'
