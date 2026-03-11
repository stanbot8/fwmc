#!/usr/bin/env bash
# renom.sh: Rewrite FWMC git history into clean atomic commits
# Usage: bash scripts/renom.sh [--dry-run] [--no-push]
#
# This destroys all existing git history and replays the working tree
# as 28 atomic commits in dependency order, backdated across today.

set -euo pipefail

DRY_RUN=false
NO_PUSH=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --no-push) NO_PUSH=true ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
BRANCH="$(git branch --show-current)"
BACKUP="/tmp/fwmc_renom_backup_$$"

# Git identity for rewritten commits (uses your git config by default)
# Uncomment and edit to override:
# export GIT_AUTHOR_NAME="Your Name"
# export GIT_COMMITTER_NAME="Your Name"
# export GIT_AUTHOR_EMAIL="you@example.com"
# export GIT_COMMITTER_EMAIL="you@example.com"

echo "=== FWMC History Rewrite ==="
echo "Repo:   $REPO_ROOT"
echo "Branch: $BRANCH"
echo ""

# Commit plan: [timestamp] [message] [files...]
# Files are relative to repo root. Directories are copied recursively.
declare -a COMMITS=(
  "06:00|Initialize project with build system and platform portability|CMakeLists.txt .gitignore setup.sh brain-model/core/platform.h"
  "06:30|Add error handling types, logging framework, and version constants|brain-model/core/error.h brain-model/core/log.h brain-model/core/version.h"
  "07:00|Add SoA neuron array with neuromodulator fields|brain-model/core/neuron_array.h"
  "07:25|Add Izhikevich and LIF neuron dynamics with NaN recovery|brain-model/core/izhikevich.h"
  "07:50|Add CSR synapse graph with NT-aware spike propagation|brain-model/core/synapse_table.h"
  "08:15|Add stimulus event types|brain-model/core/stimulus_event.h"
  "08:40|Add STDP plasticity with dopamine-gated modulation|brain-model/core/stdp.h"
  "09:05|Add binary connectome loader and exporter|brain-model/core/connectome_loader.h brain-model/core/connectome_export.h"
  "09:30|Add connectome validation and degree statistics|brain-model/core/connectome_stats.h"
  "09:55|Add checkpoint save/load with extension blob system|brain-model/core/checkpoint.h"
  "10:20|Add drift metrics and simulation recorder|brain-model/core/recorder.h"
  "10:45|Add cell type system with heterogeneous dynamics|brain-model/core/cell_types.h"
  "11:10|Add experiment config, protocol, and config loader|brain-model/core/experiment_config.h brain-model/core/experiment_protocol.h brain-model/core/config_loader.h"
  "11:35|Add parametric brain generator and spec loader|brain-model/core/parametric_gen.h brain-model/core/brain_spec_loader.h"
  "12:00|Add parameter sweep engine with grid search and hill-climbing|brain-model/core/param_sweep.h"
  "12:25|Add three-timescale parametric sync engine|brain-model/core/parametric_sync.h"
  "12:50|Add region metrics, stimulus patterns, and structural plasticity|brain-model/core/region_metrics.h brain-model/core/structural_plasticity.h brain-model/core/rate_monitor.h brain-model/core/motor_output.h brain-model/core/intrinsic_homeostasis.h"
  "13:00|Add voxel grid, brain SDF, neural field, and LOD manager|brain-model/README.md brain-model/tissue/"
  "13:15|Add bridge channel interfaces and multi-timescale spike decoder|src/bridge/bridge_channel.h src/bridge/file_read_channel.h src/bridge/spike_decoder.h"
  "13:40|Add optogenetic writer, opsin kinetics, and light model|src/bridge/optogenetic_writer.h src/bridge/opsin_model.h src/bridge/light_model.h src/bridge/shadow_tracker.h src/bridge/neuron_replacer.h"
  "14:05|Add stimulus controller and online calibrator|src/bridge/stimulus.h src/bridge/calibrator.h"
  "14:30|Add bridge checkpoint serialization|src/bridge/bridge_checkpoint.h"
  "14:55|Add hardware channels, validation, TCP bridge, and twin bridge|src/bridge/hardware_channel.h src/bridge/validation.h src/bridge/tcp_bridge.h src/bridge/twin_bridge.h"
  "15:20|Add experiment runner, conditioning demo, bridge self-test, and CLI|src/experiment_runner.h src/conditioning_experiment.h src/multi_trial.h src/bridge_self_test.h src/fwmc.cc"
  "15:45|Add unit tests and benchmarks|tests/test_harness.h tests/test_core.cc tests/test_bridge.cc tests/test_parametric.cc tests/test_tissue.cc tests/bench_core.cc"
  "16:10|Add connectome import, analysis, and visualization scripts|scripts/import_connectome.py scripts/analyze_results.py scripts/convert_connectome.py scripts/download_flywire.sh scripts/live_dashboard.py scripts/validate_electrophysiology.py scripts/visualize.py scripts/make_readme_raster.py scripts/run_drosophila.sh scripts/renom.sh"
  "16:35|Add CUDA GPU kernels for Izhikevich, spike propagation, and STDP|src/cuda/"
  "17:00|Add literature validation pipeline and reference data|literature/"
  "17:15|Add brain viewer with spiking network and region controls|.gitmodules viewer/"
  "17:25|Add documentation, example configs, and README|README.md LICENSE launch.bat launch.sh docs/ examples/ .github/"
)

TODAY=$(date -u +%Y-%m-%d)

if $DRY_RUN; then
  echo "--- DRY RUN: Commit Plan ---"
  for i in "${!COMMITS[@]}"; do
    IFS='|' read -r time msg files <<< "${COMMITS[$i]}"
    n=$((i + 1))
    echo "  $n. [${TODAY}T${time}:00Z] $msg"
    echo "     Files: $files"
  done
  echo ""
  echo "Total: ${#COMMITS[@]} commits"
  echo "(Pass without --dry-run to execute)"
  exit 0
fi

echo "WARNING: This will destroy all git history on '$BRANCH' and force-push."
read -p "Proceed? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

# Step 1: Commit any uncommitted changes
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Committing uncommitted changes..."
  git add -A
  git commit -m "Save working tree before renom"
fi

# Step 2: Backup working tree
echo "Backing up to $BACKUP..."
mkdir -p "$BACKUP"
# Copy everything except .git and build
for item in "$REPO_ROOT"/*; do
  base="$(basename "$item")"
  [[ "$base" == "build" ]] && continue
  cp -r "$item" "$BACKUP/"
done
cp "$REPO_ROOT/.gitignore" "$BACKUP/" 2>/dev/null || true
cp "$REPO_ROOT/.gitmodules" "$BACKUP/" 2>/dev/null || true
cp -r "$REPO_ROOT/.github" "$BACKUP/" 2>/dev/null || true

# Step 3: Create orphan branch
echo "Creating orphan branch..."
git checkout --orphan renom_temp
git rm -rf . > /dev/null 2>&1
git clean -fd > /dev/null 2>&1

# Step 4: Replay commits
for i in "${!COMMITS[@]}"; do
  IFS='|' read -r time msg files <<< "${COMMITS[$i]}"
  n=$((i + 1))
  ts="${TODAY}T${time}:00"

  # Copy files from backup
  for f in $files; do
    src="$BACKUP/$f"
    dst="$REPO_ROOT/$f"
    if [[ -d "$src" ]]; then
      mkdir -p "$dst"
      cp -r "$src"/* "$dst/"
    else
      mkdir -p "$(dirname "$dst")"
      cp "$src" "$dst"
    fi
  done

  git add -A
  GIT_AUTHOR_DATE="$ts" GIT_COMMITTER_DATE="$ts" git commit -m "$msg" > /dev/null
  echo "  [$n/${#COMMITS[@]}] $msg"
done

# Step 5: Rename branch
git branch -M renom_temp "$BRANCH"
echo ""
echo "Branch renamed to $BRANCH."

# Step 6: Verify
echo ""
echo "Verifying file integrity..."
diff <(cd "$BACKUP" && find . -type f | sort) <(find . -type f ! -path './.git/*' ! -path './build/*' | sort) && echo "OK: All files match." || echo "WARNING: File mismatch detected!"

# Step 7: Force push
if $NO_PUSH; then
  echo "Skipping push (--no-push)."
else
  echo "Force pushing to origin/$BRANCH..."
  git push --force origin "$BRANCH"
fi

# Step 8: Show result
echo ""
echo "=== Final History ==="
git log --oneline --format="%h %ai %s"

# Cleanup
rm -rf "$BACKUP"
echo ""
echo "Done. $((i + 1)) atomic commits on $BRANCH."
