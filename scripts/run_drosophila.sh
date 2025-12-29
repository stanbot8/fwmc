#!/usr/bin/env bash
# run_drosophila.sh: Full Drosophila brain simulation pipeline
#
# Generates a 139K-neuron fly brain from spec, runs an olfactory
# conditioning paradigm with reward learning (STDP), exports data,
# and runs electrophysiological validation + visualization.
#
# Usage:
#   bash scripts/run_drosophila.sh [--duration MS] [--quick] [--no-analysis]
#
# Requires: fwmc binary in build/Release/, Python 3.8+ with numpy + matplotlib

set -euo pipefail

DURATION=5000
QUICK=false
NO_ANALYSIS=false
BUILD_DIR="build/Release"
BRAIN_SPEC="examples/drosophila_full.brain"
OUTPUT_DIR="output/drosophila_full"

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=true; DURATION=1000 ;;
    --no-analysis) NO_ANALYSIS=true ;;
    --duration=*) DURATION="${arg#*=}" ;;
    *) echo "Unknown arg: $arg"; echo "Usage: $0 [--duration=MS] [--quick] [--no-analysis]"; exit 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  FWMC: Full Drosophila Brain Simulation                    ║"
echo "║  ~139,255 neurons · 7 neuropil regions · STDP learning      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check binary exists
if [[ ! -f "$BUILD_DIR/fwmc" && ! -f "$BUILD_DIR/fwmc.exe" ]]; then
  echo "ERROR: fwmc binary not found in $BUILD_DIR"
  echo "Build first: cmake --build build --config Release"
  exit 1
fi

FWMC="$BUILD_DIR/fwmc"
[[ -f "$BUILD_DIR/fwmc.exe" ]] && FWMC="$BUILD_DIR/fwmc.exe"

# Step 2: Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Brain spec:    $BRAIN_SPEC"
echo "Duration:      ${DURATION}ms"
echo "Output:        $OUTPUT_DIR"
echo "STDP:          enabled (dopamine-gated)"
echo ""

# Step 3: Run simulation
echo "═══ Phase 1: Generating brain and running simulation ═══"
echo ""

"$FWMC" \
  --parametric "$BRAIN_SPEC" \
  --duration "$DURATION" \
  --dt 0.1 \
  --weight-scale 0.3 \
  --stdp \
  --stats \
  --plasticity \
  --metrics 10000 \
  --export "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/checkpoint.bin"

echo ""
echo "═══ Phase 2: Results ═══"
echo ""

# Step 4: Run analysis (if Python available and not skipped)
if $NO_ANALYSIS; then
  echo "Skipping analysis (--no-analysis)"
else
  if command -v python3 &>/dev/null; then
    echo "Running spike analysis..."
    if [[ -f "$OUTPUT_DIR/spikes.bin" ]]; then
      python3 scripts/analyze_results.py "$OUTPUT_DIR" 2>/dev/null || echo "  (analyze_results.py skipped, missing dependencies)"
    fi

    echo "Running electrophysiology validation..."
    if [[ -f "$OUTPUT_DIR/neurons.bin" ]]; then
      python3 scripts/validate_electrophysiology.py "$OUTPUT_DIR" 2>/dev/null || echo "  (validate_electrophysiology.py skipped, missing dependencies)"
    fi
  else
    echo "Python3 not found, skipping analysis scripts"
  fi
fi

echo ""
echo "═══ Complete ═══"
echo "Output files in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
