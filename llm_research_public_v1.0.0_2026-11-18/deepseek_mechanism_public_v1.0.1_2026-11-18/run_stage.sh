#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${MECHANISM_PYTHON:-python}" -B -u run.py "$@"
