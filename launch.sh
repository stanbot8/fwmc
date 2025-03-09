#!/usr/bin/env bash
# Launch FWMC Brain Viewer
# Builds if needed, then runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${FWMC_BUILD_DIR:-$SCRIPT_DIR/build}"
VIEWER_EXE="$BUILD_DIR/fwmc-viewer"

if [[ ! -x "$VIEWER_EXE" ]]; then
    echo "Viewer not found, building..."
    cmake --build "$BUILD_DIR" --config Release --target fwmc-viewer
fi

echo "Launching FWMC Brain Viewer..."
exec "$VIEWER_EXE"
