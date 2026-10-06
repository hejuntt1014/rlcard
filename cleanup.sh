#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ || "$1" -le 1 ]]; then
    echo "Usage: bash cleanup.sh TRAINER_PID" >&2
    exit 2
fi
kill -INT "$1"
