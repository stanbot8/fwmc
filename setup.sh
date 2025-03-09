#!/usr/bin/env bash
# FWMC Setup Script
# Generates test data, builds all targets, runs tests and benchmarks.
#
# Usage:
#   bash setup.sh              # full setup (generate data + build + test)
#   bash setup.sh --build-only # skip data generation
#   bash setup.sh --test-only  # skip build, just run tests
#
# Requirements:
#   - CMake 3.19+
#   - C++23 compiler (MSVC 19.40+, GCC 13+, Clang 18+)
#   - Python 3 with numpy (for test data generation)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${GREEN}==> $1${NC}"; }
warn() { echo -e "${YELLOW}    $1${NC}"; }
fail() { echo -e "${RED}    FAILED: $1${NC}"; exit 1; }

BUILD_ONLY=false
TEST_ONLY=false
TEST_SIZE=500

for arg in "$@"; do
  case $arg in
    --build-only) BUILD_ONLY=true ;;
    --test-only)  TEST_ONLY=true ;;
    --test-size=*) TEST_SIZE="${arg#*=}" ;;
  esac
done

# --- Step 1: Generate test data ---
if [ "$BUILD_ONLY" = false ] && [ "$TEST_ONLY" = false ]; then
  step "Generating test connectome ($TEST_SIZE neurons)"
  if command -v python3 &>/dev/null; then
    python3 scripts/import_connectome.py --test --test-size "$TEST_SIZE"
  elif command -v python &>/dev/null; then
    python scripts/import_connectome.py --test --test-size "$TEST_SIZE"
  else
    warn "Python not found, skipping test data generation"
    warn "You can generate data later: python3 scripts/import_connectome.py --test"
  fi

  if [ -f "data/neurons.bin" ] && [ -f "data/synapses.bin" ]; then
    echo "    data/neurons.bin and data/synapses.bin ready"
  else
    warn "Test data not generated (need Python 3 + numpy)"
  fi
fi

# --- Step 2: Build ---
if [ "$TEST_ONLY" = false ]; then
  step "Configuring CMake"
  mkdir -p build

  # Find cmake
  CMAKE=""
  if command -v cmake &>/dev/null; then
    CMAKE="cmake"
  elif [ -f "/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ]; then
    CMAKE="/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
  elif [ -f "/c/Program Files/CMake/bin/cmake.exe" ]; then
    CMAKE="/c/Program Files/CMake/bin/cmake.exe"
  else
    fail "CMake not found. Install CMake 3.19+ and add to PATH."
  fi

  cd build
  "$CMAKE" .. || fail "CMake configuration failed"

  step "Building (Release)"
  "$CMAKE" --build . --config Release --parallel || fail "Build failed"
  cd ..

  # Count warnings
  echo "    Build complete"
fi

# --- Step 3: Run tests ---
step "Running tests"
TEST_EXE=""
if [ -f "build/Release/fwmc-tests.exe" ]; then
  TEST_EXE="build/Release/fwmc-tests.exe"
elif [ -f "build/fwmc-tests" ]; then
  TEST_EXE="build/fwmc-tests"
else
  fail "Test binary not found. Run setup.sh without --test-only first."
fi

"$TEST_EXE" || fail "Tests failed"

# --- Step 4: Run benchmarks ---
step "Running benchmarks"
BENCH_EXE=""
if [ -f "build/Release/fwmc-bench.exe" ]; then
  BENCH_EXE="build/Release/fwmc-bench.exe"
elif [ -f "build/fwmc-bench" ]; then
  BENCH_EXE="build/fwmc-bench"
fi

if [ -n "$BENCH_EXE" ]; then
  "$BENCH_EXE" || warn "Benchmarks had errors (non-fatal)"
else
  warn "Benchmark binary not found, skipping"
fi

# --- Step 5: Quick smoke test ---
if [ -f "data/neurons.bin" ] && [ -f "data/synapses.bin" ]; then
  step "Smoke test (100ms simulation)"
  FWMC_EXE=""
  if [ -f "build/Release/fwmc.exe" ]; then
    FWMC_EXE="build/Release/fwmc.exe"
  elif [ -f "build/fwmc" ]; then
    FWMC_EXE="build/fwmc"
  fi

  if [ -n "$FWMC_EXE" ]; then
    "$FWMC_EXE" --data data --duration 100 --stats --stdp || warn "Smoke test had issues"
  fi
else
  warn "Skipping smoke test (no test data)"
fi

# --- Done ---
step "Setup complete"
echo ""
echo "  Quick start:"
echo "    build/Release/fwmc.exe --data data --duration 1000"
echo "    build/Release/fwmc.exe --data data --duration 5000 --stdp --stats"
echo "    build/Release/fwmc.exe --experiment examples/phase1_openloop.cfg"
echo ""
echo "  For real FlyWire data:"
echo "    pip install caveclient cloud-volume numpy"
echo "    python3 scripts/import_connectome.py --region MB --max-neurons 5000"
echo ""
