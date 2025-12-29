#!/usr/bin/env bash
# Download and prepare FlyWire connectome data for FWMC
#
# Usage:
#   bash scripts/download_flywire.sh                    # test circuit (no CAVE)
#   bash scripts/download_flywire.sh --region MB        # mushroom body from CAVE
#   bash scripts/download_flywire.sh --size 1000        # limit neuron count
#   bash scripts/download_flywire.sh --test --size 500  # test circuit, 500 neurons
#   bash scripts/download_flywire.sh --cave             # full brain from CAVE
#
# Environment variables:
#   FWMC_DATA_DIR  - output directory (default: data/)
#   FWMC_PYTHON    - python executable (default: python3)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGION=""
SIZE=""
TEST_MODE=1
CAVE_MODE=0
VALIDATE=1
SMOKE_TEST=0
DATA_DIR="${FWMC_DATA_DIR:-data}"
PYTHON="${FWMC_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPORT_SCRIPT="${SCRIPT_DIR}/import_connectome.py"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)
            REGION="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=1
            CAVE_MODE=0
            shift
            ;;
        --cave)
            CAVE_MODE=1
            TEST_MODE=0
            shift
            ;;
        --no-validate)
            VALIDATE=0
            shift
            ;;
        --smoke-test)
            SMOKE_TEST=1
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help|-h)
            head -12 "${BASH_SOURCE[0]}" | tail -8
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# If a region is specified but no explicit mode, assume CAVE
if [[ -n "$REGION" && $TEST_MODE -eq 1 && $CAVE_MODE -eq 0 ]]; then
    # If no --test flag was explicitly given, default to CAVE for region queries
    # But keep test mode if no CAVE deps available
    if "$PYTHON" -c "import caveclient" 2>/dev/null; then
        CAVE_MODE=1
        TEST_MODE=0
    else
        echo "Note: caveclient not installed, using test mode with region=$REGION"
    fi
fi

echo "============================================"
echo "  FWMC FlyWire Connectome Pipeline"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check Python dependencies
# ---------------------------------------------------------------------------
echo "[1/5] Checking Python dependencies..."

missing=0
for pkg in numpy; do
    if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
        echo "  MISSING: $pkg"
        missing=1
    else
        echo "  OK: $pkg"
    fi
done

if [[ $CAVE_MODE -eq 1 ]]; then
    for pkg in caveclient pandas; do
        if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
            echo "  MISSING: $pkg (required for CAVE import)"
            missing=1
        else
            echo "  OK: $pkg"
        fi
    done
fi

if [[ $missing -eq 1 ]]; then
    echo ""
    echo "Install missing dependencies:"
    echo "  pip install numpy pandas caveclient cloud-volume"
    if [[ $CAVE_MODE -eq 1 ]]; then
        echo "Cannot proceed with CAVE mode. Exiting."
        exit 1
    fi
    echo "Continuing in test mode (numpy only)..."
fi

if [[ ! -f "$IMPORT_SCRIPT" ]]; then
    echo "ERROR: import_connectome.py not found at $IMPORT_SCRIPT"
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Create data directory
# ---------------------------------------------------------------------------
echo "[2/5] Setting up data directory..."
mkdir -p "$DATA_DIR"
echo "  Output: $DATA_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run import
# ---------------------------------------------------------------------------
echo "[3/5] Importing connectome data..."

IMPORT_ARGS=("--output" "$DATA_DIR")

if [[ $TEST_MODE -eq 1 ]]; then
    IMPORT_ARGS+=("--test")
    if [[ -n "$SIZE" ]]; then
        IMPORT_ARGS+=("--test-size" "$SIZE")
    else
        IMPORT_ARGS+=("--test-size" "200")
    fi
    echo "  Mode: test circuit generation"
else
    if [[ -n "$REGION" ]]; then
        IMPORT_ARGS+=("--region" "$REGION")
        echo "  Region: $REGION"
    fi
    if [[ -n "$SIZE" ]]; then
        IMPORT_ARGS+=("--max-neurons" "$SIZE")
        echo "  Max neurons: $SIZE"
    fi
    IMPORT_ARGS+=("--cell-types" "--nt-predictions")
    echo "  Mode: CAVE import (cell types + NT predictions)"
fi

echo ""
"$PYTHON" "$IMPORT_SCRIPT" "${IMPORT_ARGS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Validation
# ---------------------------------------------------------------------------
if [[ $VALIDATE -eq 1 ]]; then
    echo "[4/5] Running validation..."
    "$PYTHON" "$IMPORT_SCRIPT" --validate --output "$DATA_DIR" --test --test-size 0 2>/dev/null || \
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$(dirname "$IMPORT_SCRIPT")')
from import_connectome import run_validation
run_validation('$DATA_DIR')
"
    echo ""
else
    echo "[4/5] Validation skipped (--no-validate)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: Summary statistics
# ---------------------------------------------------------------------------
echo "[5/5] Summary"
echo "--------------------------------------------"

if [[ -f "$DATA_DIR/meta.json" ]]; then
    "$PYTHON" -c "
import json, os
with open('$DATA_DIR/meta.json') as f:
    m = json.load(f)
print(f\"  Neurons:  {m['n_neurons']:>8,}\")
print(f\"  Synapses: {m['n_synapses']:>8,}\")
print(f\"  Region:   {m.get('region', 'unknown')}\")
print(f\"  Source:   {m.get('source', 'unknown')}\")
if 'regions' in m:
    print(f\"  Regions:\")
    for r, c in m['regions'].items():
        print(f\"    {r}: {c} neurons\")
nbin = os.path.getsize('$DATA_DIR/neurons.bin')
sbin = os.path.getsize('$DATA_DIR/synapses.bin')
print(f\"  Files:\")
print(f\"    neurons.bin:  {nbin:>10,} bytes\")
print(f\"    synapses.bin: {sbin:>10,} bytes\")
"
fi

if [[ -f "$DATA_DIR/validation.json" ]]; then
    "$PYTHON" -c "
import json
with open('$DATA_DIR/validation.json') as f:
    v = json.load(f)
status = 'PASS' if v.get('valid', False) else 'FAIL'
print(f\"  Validation: {status}\")
if v.get('errors'):
    for e in v['errors']:
        print(f\"    ERROR: {e}\")
if v.get('warnings'):
    for w in v['warnings']:
        print(f\"    WARN:  {w}\")
"
fi

echo "--------------------------------------------"
echo ""

# ---------------------------------------------------------------------------
# Optional smoke test with simulator
# ---------------------------------------------------------------------------
if [[ $SMOKE_TEST -eq 1 ]]; then
    echo "Running smoke test..."
    FWMC_BIN=""
    for candidate in ./build/fwmc ./build/Release/fwmc ./fwmc; do
        if [[ -x "$candidate" ]]; then
            FWMC_BIN="$candidate"
            break
        fi
    done
    if [[ -n "$FWMC_BIN" ]]; then
        echo "  Using simulator: $FWMC_BIN"
        "$FWMC_BIN" --data-dir "$DATA_DIR" --steps 100 --output-dir "$DATA_DIR/smoke_test" || {
            echo "  Smoke test failed (exit code $?)"
            exit 1
        }
        echo "  Smoke test passed"
    else
        echo "  Simulator binary not found. Build with: cmake --build build"
        echo "  Skipping smoke test."
    fi
    echo ""
fi

echo "Done. Connectome data is in: $DATA_DIR/"
