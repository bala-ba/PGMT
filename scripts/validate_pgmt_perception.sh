#!/usr/bin/env bash
set -euo pipefail
PGMT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PGMT_SCRIPT_DIR}/validate_pgmt_v6.sh" --check perception "$@"
