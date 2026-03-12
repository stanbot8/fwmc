# FlyWire Mind Couple (FWMC)

**FWMC** is a C++23 spiking neural network simulator built around the [FlyWire](https://flywire.ai) *Drosophila melanogaster* whole-brain connectome (~140k neurons, ~50M synapses). It runs Izhikevich dynamics on biologically constrained circuits, supports parametric brain generation from high-level specs, and includes a bidirectional bridge system for interfacing digital and biological neural activity. No framework dependencies. Optional CUDA and OpenMP acceleration.

The spiking network drives a virtual fly body through [flygame](https://github.com/stanbot8/flygame), an embedded C++ MuJoCo port of [flygym](https://github.com/NeLy-EPFL/flygym) (NeuroMechFly). WASD keyboard input injects current into descending neurons; the brain's spike activity maps to locomotion in real time, with a live brain viewport showing ~223k neurons firing.

## Motivation

The fruit fly brain is the largest connectome mapped at synaptic resolution. FWMC puts that data to work as a computational neuroscience tool: load real connectivity, assign biologically tuned cell types, and simulate spiking dynamics with plasticity and neuromodulation.

The bridge system extends this into closed-loop neuroscience. A digital twin reads biological neural activity through calcium imaging, measures prediction drift in shadow mode, and (in closed-loop mode) writes stimulation commands back through optogenetics. Neurons that the model predicts accurately can be gradually handed off from biological to digital substrate. The long-term direction is identity-preserving brain-computer integration, where neural function transfers continuously rather than through abrupt replacement. See [docs/bidirectional_neural_twinning.md](docs/bidirectional_neural_twinning.md) for the research outline.

---

## Quick start

### Linux / macOS

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests (237 total across 4 executables)
./test_core && ./test_bridge && ./test_parametric && ./test_tissue
```

### Windows (MSVC)

```cmd
mkdir build && cd build
cmake ..
cmake --build . --config Release
Release\test_core.exe && Release\test_bridge.exe && Release\test_parametric.exe && Release\test_tissue.exe
```

---

## Running modes

FWMC supports four distinct modes of operation. Each is self-contained. Pick the one that matches your use case.

### 1. Direct mode: simulate from binary connectome data

Load a FlyWire-derived connectome and run Izhikevich dynamics with spike propagation.

```bash
# Generate a test circuit (no CAVE access needed)
python3 scripts/import_connectome.py --test --test-size 500

# Run open-loop simulation
./fwmc --data data --duration 1000 --stdp --stats

# Shadow mode: read bio, measure drift, no write-back
./fwmc --data data --duration 5000 --shadow

# Closed-loop: full bidirectional bridge with neuron replacement
./fwmc --data data --duration 10000 --closed-loop

# Import real FlyWire data via Codex (no auth needed)
python3 scripts/import_connectome.py --codex --region MB

# Or via CAVE API (requires caveclient auth)
pip install caveclient cloud-volume numpy
python3 scripts/import_connectome.py --region MB --max-neurons 5000
```

### 2. Experiment mode: structured protocols with recording

Config-driven experiments with stimulus timing, recording, and bridge thresholds. Automatically saves full configuration and output data for reproducibility.

```bash
./fwmc --experiment examples/phase1_openloop.cfg
```

```ini
# examples/phase1_openloop.cfg
name = phase1_odor_response
dt_ms = 0.1
duration_ms = 10000
enable_stdp = true
bridge_mode = 0                  # 0=open-loop, 1=shadow, 2=closed-loop
connectome_dir = data
output_dir = results/phase1
weight_scale = 1.0
metrics_interval = 10000
recording_interval = 10

# Bridge thresholds (shadow/closed-loop)
monitor_threshold = 0.6
bridge_threshold = 0.8
resync_threshold = 0.4
min_observation_ms = 10000

# Calibration
calibration_interval = 10000     # 0 = disabled
calibration_lr = 0.001

# Recording
record_spikes = true
record_voltages = false
record_shadow_metrics = true
record_per_neuron_error = true

# Stimulus protocol: label start_ms end_ms intensity target_neurons
stimulus: odor_A 1000 2000 0.8 0,1,2,3,4
stimulus: odor_B 5000 6000 0.6 5,6,7,8,9
```

Output files are written to `output_dir`:

| File | Format | Contents |
|------|--------|----------|
| `spikes.bin` | Binary | Per-step spike vectors |
| `voltages.bin` | Binary | Per-step voltage traces |
| `metrics.csv` | CSV | Correlation, RMSE, false positives/negatives, resyncs |
| `per_neuron_error.bin` | Binary | Per-neuron prediction error over time |
| `experiment.cfg` | Text | Auto-saved copy of configuration |

```bash
# Analyze results
python3 scripts/analyze_results.py results/phase1
python3 scripts/analyze_results.py results/phase1 --plot
```

### 3. Parametric mode: synthetic connectome generation

Generate connectomes from high-level brain specifications. No binary data files needed. Define regions with neuron counts, cell type distributions, and neurotransmitter ratios, then wire them with projection specs.

```bash
# Generate and simulate a mushroom body circuit
./fwmc --parametric examples/parametric_mushroom_body.brain --duration 5000 --stats

# Auto-tune neuron parameters with grid search + hill-climbing
./fwmc --parametric examples/parametric_mushroom_body.brain \
       --sweep --sweep-target 15 --duration 2000

# Central complex navigation circuit with timed stimuli
./fwmc --parametric examples/central_complex.brain --duration 5000 --stdp

# Antennal lobe odor discrimination with per-region metrics
./fwmc --parametric examples/antennal_lobe.brain --duration 2000

# Export parametric brain to binary format (for use with direct mode)
./fwmc --parametric examples/parametric_mushroom_body.brain \
       --export data/generated --duration 1000

# Enable structural plasticity (synapse pruning and sprouting)
./fwmc --parametric examples/parametric_mushroom_body.brain \
       --plasticity --duration 10000

# Full Drosophila brain (~139K neurons, 7 regions, olfactory conditioning)
./fwmc --parametric examples/drosophila_full.brain \
       --duration 5000 --weight-scale 0.3 --stdp --stats
```

Brain spec format (`.brain` files):

```ini
# examples/parametric_mushroom_body.brain
name = mushroom_body_model
seed = 42
weight_mean = 1.0
weight_std = 0.3

# Background synaptic bombardment (tonic drive + noise)
# Mean keeps neurons near threshold; std adds realistic variability
background_mean = 12.0
background_std = 4.0

# Antennal lobe: 500 neurons, lateral inhibition
region.0.name = antennal_lobe
region.0.n_neurons = 500
region.0.density = 0.12
region.0.types = PN:0.4 LN:0.6
region.0.nt_dist = ACh:0.4 GABA:0.6

# Mushroom body: 2000 neurons, sparse coding
region.1.name = mushroom_body
region.1.n_neurons = 2000
region.1.density = 0.02
region.1.types = KC:0.85 MBON_cholinergic:0.08 DAN_PAM:0.04 DAN_PPL1:0.03
region.1.nt_dist = ACh:0.7 GABA:0.15 DA:0.15

# Lateral horn: 300 neurons, innate odor responses
region.2.name = lateral_horn
region.2.n_neurons = 300
region.2.density = 0.08
region.2.types = PN:0.5 LN:0.3 FastSpiking:0.2
region.2.nt_dist = ACh:0.6 GABA:0.4

# Inter-region projections
projection.0.from = antennal_lobe
projection.0.to = mushroom_body
projection.0.density = 0.01
projection.0.nt = ACh
projection.0.weight_mean = 1.5
projection.0.weight_std = 0.4

projection.1.from = antennal_lobe
projection.1.to = lateral_horn
projection.1.density = 0.05
projection.1.nt = ACh

projection.2.from = mushroom_body
projection.2.to = lateral_horn
projection.2.density = 0.005
projection.2.nt = ACh

projection.3.from = lateral_horn
projection.3.to = mushroom_body
projection.3.density = 0.003
projection.3.nt = GABA

# Timed stimulus patterns
stimulus.0.label = odor_presentation
stimulus.0.region = antennal_lobe
stimulus.0.start = 500
stimulus.0.end = 1500
stimulus.0.intensity = 8.0
stimulus.0.fraction = 0.3       # target 30% of region neurons

stimulus.1.label = reward_signal
stimulus.1.region = mushroom_body
stimulus.1.start = 1000
stimulus.1.end = 1200
stimulus.1.intensity = 5.0
stimulus.1.fraction = 0.07      # DANs only (~7%)
```

Supported cell types: `KC`, `MBON_cholinergic`, `MBON_gabaergic`, `MBON_glutamatergic`, `DAN_PPL1`, `DAN_PAM`, `PN` / `PN_excitatory`, `PN_inhibitory`, `LN` / `LN_local`, `ORN`, `FastSpiking`, `Bursting`.

Supported neurotransmitters: `ACh`, `GABA`, `Glut`, `DA`, `5HT`, `OA`.

Each cell type gets biologically-tuned Izhikevich parameters (KC = regular spiking, LN = fast spiking, PN = bursting, etc.) and neurons step with per-neuron heterogeneous dynamics via OpenMP.

**Parameter sweep** auto-tunes (a, b, c, d) for each cell type using grid search (N^4) + stochastic hill-climbing. Built-in scoring functions: `TargetFiringRate`, `ActivityInRange`, `RealisticCV`.

### 4. Sync mode: adapt a parametric brain to match a reference

Tune a parametric brain to converge toward a reference brain (real or simulated). Three adaptation mechanisms run at different timescales:

| Timescale | Interval | Mechanism |
|-----------|----------|-----------|
| Fast | Every step | Corrective current injection: pushes model voltages toward reference |
| Medium | 100 steps | Synaptic weight updates via momentum SGD on per-synapse error attribution |
| Slow | 1000 steps | Izhikevich parameter nudges: adjusts (a, c, d) based on per-neuron firing rate mismatch |

```bash
# Sync a model brain to a reference brain
./fwmc --parametric model.brain --sync reference.brain --duration 10000

# Set convergence threshold (fraction of neurons that must match)
./fwmc --parametric model.brain --sync reference.brain \
       --duration 20000 --sync-target 0.98

# Sync then checkpoint the tuned model
./fwmc --parametric model.brain --sync reference.brain \
       --duration 10000 --checkpoint tuned_state.bin
```

The sync engine tracks per-neuron convergence state (EMA correlation, voltage error, firing rate mismatch) and terminates early when the target fraction of neurons have converged. Use this to:

- Calibrate a parametric model against recorded data before deploying as a digital twin
- Test how well different brain specs approximate the same dynamics
- Find minimal parameter differences between two circuit hypotheses

### 5. Built-in demos (no data needed)

Two self-contained demos validate the core scientific claims without requiring connectome data or hardware.

**Olfactory conditioning** (`--conditioning`): classical conditioning on a programmatically generated mushroom body circuit. Builds ORN -> PN -> KC -> MBON pathway with DAN dopamine neurons, runs pre-test/training/post-test with three-factor STDP, synaptic scaling, intrinsic homeostasis, motor output readout, and per-region firing rate validation against Drosophila literature values.

```bash
./fwmc --conditioning --seed 42
# Output:
# Pre-test:  CS+ MBON spikes=120  CS-=120
# Post-test: CS+ MBON spikes=141  CS-=141
# KC->MBON weight: 0.997 -> 3.356 (ratio=3.368)
# Approach drive: CS+ 0.000 -> 0.150  CS- 0.020
# Behavioral learning: 0.150
# Firing rates: 3/5 regions in biological range
# Learned: YES
```

**Multi-trial analysis** (`--multi-trial N`): runs N conditioning trials with different random seeds and computes aggregate statistics (mean, std, min, max of learning index, discrimination, behavioral learning, weight change ratio, success rate).

```bash
./fwmc --multi-trial 10 --seed 42
# Output:
# Learning index:   0.150 +/- 0.030  [0.090, 0.210]
# Discrimination:   0.120 +/- 0.025
# Behavioral:       0.080 +/- 0.015
# Success rate:     8/10 (80%)
```

**Bridge self-test** (`--bridge-test`): validates the entire twinning pipeline in software. Generates a 1000-neuron circuit, pre-records "biological" activity with noise, then runs shadow tracking, online calibration, closed-loop neuron replacement, and perturbation recovery.

```bash
./fwmc --bridge-test
# Output:
# Calibration: error 0.1303 -> 0.1296 (1% reduction)
# Replacement: 0.0% of neurons, 1000 promoted, 13 resyncs
# Overall: PASS
```

---

## CLI reference

```
./fwmc [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| **Simulation** | | |
| `--dt` | 0.1 | Timestep in ms |
| `--duration` | 1000 | Simulation duration in ms |
| `--weight-scale` | 1.0 | Global synaptic weight multiplier |
| `--metrics` | 1000 | Print metrics every N steps |
| `--stdp` | off | Enable spike-timing-dependent plasticity |
| **Data sources** | | |
| `--data` | `data` | Directory with `neurons.bin` and `synapses.bin` |
| `--experiment` | - | Run from experiment config file |
| `--parametric` | - | Generate connectome from `.brain` spec file |
| **Bridge** | | |
| `--shadow` | off | Shadow mode (read bio, measure drift, no write-back) |
| `--closed-loop` | off | Full bidirectional bridge with neuron replacement |
| **Demos** | | |
| `--conditioning` | off | Run olfactory conditioning experiment (no data needed) |
| `--multi-trial` | 0 | Run N conditioning trials with aggregate statistics |
| `--bridge-test` | off | Run bridge self-test (no hardware needed) |
| **Parametric extras** | | |
| `--sweep` | off | Run parameter sweep (with `--parametric`) |
| `--sweep-target` | 10 | Target firing rate in Hz for sweep |
| `--sync` | - | Reference brain spec for sync mode |
| `--sync-target` | 0.95 | Convergence threshold for sync (0-1) |
| `--export` | - | Export parametric brain to binary files in DIR |
| `--plasticity` | off | Enable structural plasticity (synapse pruning/sprouting) |
| **Reproducibility** | | |
| `--seed` | 42 | Random seed for deterministic simulation |
| **State management** | | |
| `--stats` | off | Print connectome statistics after loading |
| `--checkpoint` | - | Save simulation state to PATH at end of run |
| `--checkpoint-every` | 0 | Save checkpoint every N steps (0=disabled) |
| `--resume` | - | Resume simulation from checkpoint file |
| **Info** | | |
| `--help` | - | Show help message |
| `--version` | - | Show version |

---

## Project structure

```
brain-model/                  Spiking neural network engine (header-only C++23)
  core/                       Neuron dynamics, synapses, plasticity, proprioception, CPG
  tissue/                     Procedural 3D brain volume and Wilson-Cowan neural field
src/
  bridge/                     Bidirectional bridge: spike decoding, optogenetics, shadow tracking
  cuda/                       GPU kernels (Izhikevich, spike propagation, STDP)
  experiment_runner.h         Top-level experiment orchestrator
  fwmc.cc                     Entry point and CLI
viewer/                       OpenGL brain viewer with live spiking visualization
scripts/                      Import, analysis, visualization, validation tools
examples/                     Config files for experiments, parametric brains, protocols
tests/                        Unit tests (237 total) and benchmarks
docs/                         Methods, architecture, API reference, research outline
literature/                   Validation pipeline and reference data
```

See [brain-model/README.md](brain-model/README.md) for detailed module documentation.

---

## Bridge system

The bridge connects biological and digital neural circuits. Three operational modes map to the research phases.

### Open-loop (default)

Pure simulation. Load connectome, run Izhikevich dynamics, propagate spikes. No biological I/O.

### Shadow mode (Phase 2)

Digital twin runs in parallel with the biological brain via the read channel. Measures prediction accuracy and drift rate over time. No write-back to biology. Answers: *how fast does the model diverge from reality?*

### Closed-loop (Phase 3+)

Full bidirectional bridge. Reads biological activity, runs the digital twin, writes stimulation commands back through optogenetics. Neurons progress through a four-state machine:

```
BIOLOGICAL → MONITORED → BRIDGED → REPLACED
```

Each transition requires sustained correlation above threshold with hysteresis (must exceed `threshold + margin` to promote, drops below `threshold - margin` to demote). Neurons that exceed `max_rollbacks` are permanently held at MONITORED.

**Adaptive boundaries**: When a MONITORED or BRIDGED neuron drifts, its post-synaptic neighbors are auto-promoted from BIOLOGICAL to MONITORED. Tracks the spreading divergence front without manual neuron selection.

**Resync cooldown**: When drift exceeds threshold, the shadow tracker resyncs digital state from biological readings. Configurable cooldown (`resync_cooldown_ms`, default 100ms) prevents resync thrashing.

### Hierarchical tick structure

| Tick | Interval | Operations |
|------|----------|------------|
| Fast | Every step (0.1ms) | Izhikevich dynamics, spike propagation |
| Medium | 10 steps (1ms) | Read channel decode, shadow measurement, boundary expansion |
| Slow | 50 steps (5ms) | Galvo-SLM command generation, neuron state advancement |

Fast ticks reuse cached bio readings from the last medium tick. Simulation accuracy stays at 0.1ms while shadow tracking and actuation run at their natural hardware timescales.

**Latency monitoring**: Tracks `last_step_us`, `max_step_us`, `mean_step_us`, and `deadline_misses` (steps exceeding `dt_ms`).

---

## Neuron models

**Izhikevich** (default): 2 ODEs per neuron, captures 20+ firing patterns. Two half-step integration for numerical stability. NaN/Inf guard and voltage clamp (>100mV reset) for divergent neurons. Per-cell-type parameters for Kenyon cells, projection neurons, local interneurons, MBONs, DANs, and ORNs. OpenMP parallelized for >10k neurons. Spike propagation and STDP updates are also OpenMP parallelized (CSR row-parallel with atomic accumulation for spike propagation, conflict-free for STDP since each synapse is owned by one pre-neuron).

**AVX2 SIMD**: `IzhikevichStepFast` automatically dispatches to an AVX2-vectorized path when available, processing 8 neurons per iteration using 256-bit float vectors with FMA. Benchmarks at ~87M neurons/sec (10K neurons, MSVC Release). Falls back to scalar on non-AVX2 hardware.

**LIF**: Single ODE, fastest. Good for large-scale connectivity studies where individual neuron dynamics are less important.

---

## Plasticity and neuromodulation

**STDP**: Exponential timing windows. Pre-before-post potentiates, post-before-pre depresses. Weights bounded to `[w_min, w_max]`.

**Dopamine-gated STDP** (Izhikevich 2007): STDP weight changes modulated by local dopamine concentration: `dw_effective = dw * (1 + da_scale * dopamine[post])`. Implements reward-modulated learning in the mushroom body.

**Neuromodulator dynamics**: DAN neurons release dopamine to post-synaptic targets on spike. Exponential decay. Autoreceptor feedback at 50% release rate. Octopamine released by fast-spiking interneurons (arousal signal). All concentrations clamped to [0, 1].

**Supervised calibration**: Perturbation-based gradient-free weight optimization. Accumulates prediction error between digital and biological spike patterns, applies momentum SGD weight updates periodically.

**Short-term plasticity** (Tsodyks-Markram): Per-synapse facilitation and depression. Each synapse tracks utilization `u` (residual calcium) and available resources `x` (vesicle pool). Effective weight = `w * u * x`. Three presets: `STPFacilitating` (low release, slow facilitation decay), `STPDepressing` (high release, slow recovery), `STPCombined` (non-monotonic response). OpenMP parallelized.

**Structural plasticity**: Synapse pruning removes weak connections (weight < threshold, default 0.05). Synapse sprouting creates new connections between correlated neurons that co-fire. Both operations run at configurable intervals (default every 5000 steps). Enable with `--plasticity`.

**Gap junctions**: Electrical synapses passing current proportional to voltage difference: `I = g * (Vb - Va)`. Bidirectional (symmetric). `BuildFromRegion` connects neurons within a region at a given density. Important for Drosophila clock neurons, giant fiber escape circuit, and antennal lobe oscillatory coupling.

---

## Per-region metrics

When running in parametric mode, FWMC automatically tracks per-region activity:

- **Spike count**: Number of spikes per region per snapshot
- **Firing rate** (Hz): Spikes per neuron per second within the measurement window
- **Fraction active**: Proportion of neurons that have spiked at least once
- **Mean voltage**: Average membrane potential across the region

Region summaries are logged at the end of each run, showing total spikes, peak firing rate, and mean fraction active across all snapshots.

---

## Firing rate validation

The `RateMonitor` compares per-region firing rates against literature-derived reference ranges for *Drosophila* brain regions:

| Region | Spontaneous rate | Range | Source |
|--------|-----------------|-------|--------|
| KC | 1.5 Hz | 0.5-10 Hz | Murthy & Turner 2013 |
| PN | 8 Hz | 2-30 Hz | Wilson 2013 |
| MBON | 5 Hz | 1-25 Hz | Aso et al. 2014 |
| DAN | 3 Hz | 0.5-15 Hz | Cohn et al. 2015 |
| Optic lobe | 15 Hz | 5-60 Hz | Behnia et al. 2014 |

The rate monitor runs automatically in both experiment mode and the conditioning demo, reporting which regions fall within biological range and flagging deviations.

---

## Motor output

The `MotorOutput` system maps neural activity to fictive locomotion commands (Namiki et al. 2018, Bidaye et al. 2014):

- **Forward velocity**: mean descending neuron drive (L+R), mapped to mm/s (max 30 mm/s, matching *Drosophila* walking speed)
- **Turning**: L/R asymmetry in descending neuron activity
- **Approach/avoid**: balance between cholinergic MBONs (approach) and GABAergic MBONs (avoidance), following the Aso & Rubin 2016 valence model
- **Freeze**: triggered when total descending drive drops below threshold

This closes the sensory-motor loop: the conditioning experiment now reports behavioral changes (approach drive shift) alongside synaptic weight changes.

---

## Proprioceptive feedback

The `ProprioMap` closes the sensorimotor loop by mapping MuJoCo body state back into VNC sensory neurons. Channels:

| Channel | Count | Signal |
|---------|-------|--------|
| Joint angle sensors | 42 (6 legs x 7 joints) | Sigmoid of joint angle magnitude |
| Contact sensors | 6 (per leg) | Ground contact force [0,1] |
| Body velocity | 3 | Forward, lateral speed |
| Haltere (L/R) | 2 | Yaw rate, asymmetric L/R for corrective steering |

The first 30% of VNC neurons are designated as sensory afferents and distributed across channels. Gains are configurable via `ProprioConfig`. When running with a MuJoCo body sim, `ReadProprioFromMuJoCo()` extracts joint angles, velocities, contacts, and body velocity from `mjData`.

---

## Central pattern generator

The `CPGOscillator` produces spontaneous rhythmic locomotion without keyboard input, matching Drosophila VNC CPG circuits (Bidaye et al. 2018, Mendes et al. 2013).

- Two anti-phase neuron groups split by midline x-coordinate (tripod gait pattern)
- Oscillatory current at 8 Hz (configurable, Drosophila range: 5-15 Hz)
- Tonic baseline drive keeps neurons near threshold
- `drive_scale` modulated by descending commands: 0 = CPG silent, 1 = full amplitude
- Smooth 50ms time constant prevents abrupt on/off transitions
- Skips sensory neurons (first 30% of VNC, reserved for proprioception)

In the viewer, WASD forward input serves as the descending drive signal, so pressing W activates the CPG rhythm while releasing it smoothly decays.

---

## Binary connectome export

Export parametric brains to the same binary format used by direct mode, enabling a workflow where you design circuits with `.brain` specs and then run them with the full bridge system:

```bash
# Generate and export
./fwmc --parametric examples/antennal_lobe.brain --export data/al --duration 100

# Run with bridge
./fwmc --data data/al --duration 10000 --shadow
```

---

## NWB-compatible data export

Export simulation results in [Neurodata Without Borders](https://www.nwb.org/) compatible format (no HDF5 dependency):

```bash
# In code:
NWBExporter nwb;
nwb.SetVoltageSubset({0, 1, 2, 3, 4});  // optional voltage traces
nwb.BeginSession("output/", "Olfactory conditioning trial", neurons);
// ... simulation loop ...
nwb.RecordTimestep(time_ms, neurons);
nwb.EndSession();
```

Output files:

| File | Format | Contents |
|------|--------|----------|
| `spikes.nwb.csv` | CSV | `neuron_id, spike_time_ms, region, cell_type` |
| `voltages.nwb.csv` | CSV | `time_ms, neuron_0, neuron_1, ...` (configurable subset) |
| `session.nwb.json` | JSON | NWB 2.7 session metadata: subject, units, stimuli, acquisition |

---

## Optogenetic safety model

The optogenetic writer converts digital twin spike decisions into holographic two-photon stimulation commands with multiple safety layers:

- **Refractory period**: Per-neuron minimum time between stimulations (default 5ms)
- **Thermal energy tracking**: Cumulative energy with dissipation; stimulation blocked at energy limit
- **Nonlinear power curve**: Hill equation for CsChrimson activation
- **SLM target limit**: Commands exceeding `max_simultaneous_targets` prioritized (excitatory first)
- **Galvo-SLM hybrid**: Top priority neurons route to galvo mirrors (~0.1ms), remaining batch to SLM (~5ms)
- **Predictive pre-staging**: Pre-computes next 5 SLM hologram patterns using voltage-trend prediction

### Opsin kinetics

Three-state channel model (Closed, Open, Desensitized) for biophysically realistic photocurrent generation. Supported opsins:

| Opsin | Type | Peak wavelength | Conductance | Use case |
|-------|------|----------------|-------------|----------|
| ChR2 | Excitatory | 470nm (blue) | 0.4 nS | Standard activation |
| ChRmine | Excitatory | 590nm (red) | 2.5 nS | Deep tissue, large photocurrent |
| stGtACR2 | Inhibitory | 515nm (green) | 1.5 nS | Soma-targeted silencing |

Kinetics include light-dependent channel opening, thermal closing, use-dependent desensitization, and slow recovery. Photocurrent computed as `I = g_max * open_fraction * (V - E_rev)`.

### Tissue light model

Beer-Lambert attenuation with wavelength-dependent scattering/absorption coefficients calibrated to Drosophila brain tissue (Prakash et al. 2012). Models two-photon excitation optics including Gaussian beam lateral spread, axial defocus, and multi-spot holographic power splitting.

### Validation engine

Compares simulated spike trains against experimental recordings at multiple scales:

- **Single neuron**: spike timing precision/recall (F1), van Rossum distance, binned rate correlation
- **Population**: mean correlation, spatial firing pattern similarity, synchrony index
- **Temporal**: sliding window analysis with 50% overlap for stationarity assessment

### Hardware integration

Callback-based and shared-memory channel adapters for connecting to real experimental rigs. Configuration structs provided for Open Ephys (Neuropixels), ScanImage (two-photon), and Bonsai (reactive programming). Lock-free ring buffer included for streaming between acquisition and simulation threads.

---

## Spike decoder

Multi-timescale calcium-to-spike deconvolution approximating CASCADE (Rupprecht et al., 2021). Three exponential kernels:

| Kernel | Time constant | Role |
|--------|--------------|------|
| Fast | 20ms | Rise/onset detection |
| Medium | 100ms | Standard GCaMP8f decay |
| Slow | 500ms | Sustained/burst activity |

Adaptive thresholding based on running noise statistics. Nonlinear spike probability via soft saturation.

**Adaptive resolution** (`DecodeSelective`): Full multi-timescale deconvolution only for actively tracked neurons (MONITORED/BRIDGED). Remaining neurons use a cheap population-rate threshold approximation.

---

## Checkpointing

Save and restore full simulation state for long-running experiments:

```bash
# Save at end of run
./fwmc --data data --duration 10000 --checkpoint state.bin

# Periodic checkpoints every 100k steps
./fwmc --data data --duration 100000 --checkpoint state.bin --checkpoint-every 100000

# Resume from checkpoint
./fwmc --data data --duration 10000 --resume state.bin
```

Checkpoint files include: neuron voltages, recovery variables, neuromodulator concentrations, synapse weights, replacer state machine, and shadow tracker history.

---

## Connectome validation

On load, the connectome is validated and statistics logged:

- Degree distribution (in/out min/max/mean/median)
- NT type ratios (excitatory/inhibitory/modulatory)
- Weight statistics (min/max/mean)
- Integrity checks: self-loops, out-of-bounds indices, NaN weights, isolated neurons

Enable with `--stats`.

---

## Performance

Benchmarks on a single CPU core (MSVC 19.44, AVX2, Release):

| Benchmark | Size | Result |
|-----------|------|--------|
| IzhikevichStep (AVX2) | 10k neurons | ~87M neurons/sec |
| IzhikevichStep (AVX2) | 140k neurons | ~81M neurons/sec |
| Spike propagation | 140k neurons, 7M synapses | 1.85 ms/iter |
| STDP update | 140k neurons, 7M synapses | 24.6 ms/iter |
| Full step (izh+prop+stdp) | 140k neurons | 11.8 ms/step |
| Full bridge step | 10k neurons, 500k synapses | 0.78x real-time |

---

## flygame

FWMC uses [flygame](https://github.com/stanbot8/flygame), an embedded C++ MuJoCo port of [flygym](https://github.com/NeLy-EPFL/flygym) (NeuroMechFly), with an additional brain viewport embedded in the ImGui layout.

**Main window (MuJoCo + ImGui):**
- NeuroMechFly model (48 actuators: 42 joint position + 6 adhesion) with Kuramoto CPG
- Brain activity maps to locomotion: SEZ spike rates become forward/angular velocity, CPG converts to joint trajectories from recorded fly walking data
- Haltere feedback: body yaw rate feeds back as differential L/R SEZ inhibition for self-correcting turn bias
- Smooth EMA camera follow, physics at 0.0001s timestep (10 substeps per controller tick)
- Automatic instability detection and pose reset
- WASD keyboard controls inject current into SEZ descending neurons
- Activity monitor, scrolling spike raster, motor output readout
- STDP learning toggle, TCP bridge toggle

**Brain viewport (bottom-right):**
- ~223k neuron positions rendered as an OpenGL 3.3 point cloud with per-region coloring
- Izhikevich spiking network with real spike propagation (~441k synapses)
- Mouse orbit, pan, and zoom on the brain image
- Transparent collapsible regions overlay with per-region visibility/size controls, TOML config persistence

Built with GLFW, glad, GLM, Dear ImGui (vendored as git submodules), and MuJoCo.

```bash
# Build (requires OpenGL 3.3+, MuJoCo, and submodules)
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fwmc-viewer
./build/fwmc-viewer
```

---

## CUDA GPU acceleration

When built with CUDA, three hot-path kernels run on the GPU:

| Kernel | Strategy | Notes |
|--------|----------|-------|
| `IzhikevichStepGPU` | 1 thread per neuron | Same math as CPU, NaN recovery included |
| `PropagateSpikesGPU` | 1 thread per pre-neuron (CSR row scan) | `atomicAdd` for i_syn accumulation |
| `STDPUpdateGPU` | 1 thread per synapse | Dopamine-gated modulation supported |

All kernels support CUDA streams for async execution. The `GPUManager` handles device memory lifecycle (alloc/upload/download/free) with `CUDA_CHECK` error macro on every API call.

```bash
# Build with CUDA (auto-detected by CMake)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# If CUDA is found: "CUDA enabled: /usr/local/cuda/bin/nvcc"
# If not: "CUDA not found, CPU-only build" (all CPU code still works)
```

Guard: all GPU code is behind `#ifdef FWMC_CUDA` so CPU-only builds compile cleanly.

---

## Multi-phase experiment protocols

Define scripted multi-phase experiments with automatic transitions:

```bash
# Run a full twinning protocol (3 phases)
# Protocol files are INI-like, loaded by ExperimentProtocol::LoadFromFile()
cat examples/protocols/full_twinning.protocol
```

Built-in protocol factories:

| Factory | Phases | Use case |
|---------|--------|----------|
| `OpenLoopBaseline` | 1 | Establish baseline activity stats |
| `ShadowValidation` | 2 | Warmup + shadow with early exit on high correlation |
| `FullTwinning` | 3 | Open-loop → shadow → closed-loop replacement |
| `OdorLearning` | 2N+1 | N interleaved CS+/CS- trials with ITI, dopamine-gated STDP |
| `AblationStudy` | 4 | Baseline → silence neurons → observe → recovery |

Transition conditions: `kTimeOnly` (run full duration), `kCorrelationAbove` (end when correlation exceeds threshold), `kSpikeRateBelow` (end when activity drops).

The experiment-level `ExperimentSweep` engine extends `ParamSweep` with grid search, random search, and hill climbing over arbitrary config parameters (weight_scale, dt_ms, STDP params, etc.).

---

## Visualization and analysis

### Static plots

```bash
# Generate publication-quality figures from simulation output
python3 scripts/visualize.py results/ --output figures/
```

Generates 9 plot types: spike raster (color-coded by region), firing rate heatmap, population rate curves, connectivity matrix, weight distribution, Izhikevich phase plot, ISI distribution, correlation matrix, and 3D brain position scatter.

### Live dashboard

```bash
# Launch real-time browser dashboard during simulation
python3 scripts/live_dashboard.py --data data --duration 10000 --port 8050
# Open http://localhost:8050 in browser
```

Streams simulation data via Server-Sent Events to a Canvas2D dashboard showing live spike raster, population firing rate, and neuron state counts. No external JS dependencies.

### Electrophysiology validation

```bash
# Compare simulation against published Drosophila electrophysiology data
python3 scripts/validate_electrophysiology.py results/ --report validation.json
```

Validates 7 metrics against published reference values:

| Metric | Reference | Source |
|--------|-----------|--------|
| Firing rates by cell type | KC: 0.5-5 Hz, PN: 5-30 Hz, LN: 10-50 Hz | Turner et al. 2008 |
| Population sparseness | MB <10% KCs active, AL 30-60% PNs | Honegger et al. 2011 |
| LFP oscillations | AL: 10-30 Hz, MB: asynchronous | Tanaka et al. 2009 |
| Temporal adaptation | PN: decreasing rate over 500ms | Mazor & Laurent 2005 |
| Correlation structure | Within-region > between-region | Lin et al. 2014 |
| CV of ISI | ~0.5-1.0 for Poisson-like spiking | N/A |
| Fano factor | ~1.0 for renewal processes | N/A |

Outputs an overall "biological plausibility score" (0-1) and flags any non-biological metrics.

---

## Data pipeline

### Import from FlyWire

```bash
# Codex API (no auth needed, easiest path)
python3 scripts/import_connectome.py --codex --region MB

# Full automated pipeline (CAVE API, requires auth)
bash scripts/download_flywire.sh --region MB --size 5000

# Manual import with cell type annotations and validation
python3 scripts/import_connectome.py --region MB --cell-types --validate

# Multi-region import with merge
python3 scripts/import_connectome.py --region AL --output data/al
python3 scripts/import_connectome.py --region MB --output data/mb
python3 scripts/import_connectome.py --merge data/al data/mb --output data/al_mb

# Enhanced test circuit (multi-region with inter-region projections)
python3 scripts/import_connectome.py --test --test-size 1000 --output data
```

### Convert from other formats

```bash
# CSV adjacency list → FWMC binary
python3 scripts/convert_connectome.py --input edges.csv --format csv --output data/

# NeuPrint hemibrain JSON → FWMC binary
python3 scripts/convert_connectome.py --input hemibrain.json --format neuprint --output data/

# FWMC binary → GraphML for Gephi visualization
python3 scripts/convert_connectome.py --input data/ --format fwmc --output-format graphml --output graph.graphml
```

Supported formats: CSV, EdgeList, GraphML, NeuPrint JSON, FWMC binary.

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

| Platform | Compiler | Status |
|----------|----------|--------|
| Ubuntu | GCC 13 | Build + test + smoke |
| Ubuntu | Clang 18 | Build + test + smoke |
| Windows | MSVC 2022 | Build + test + smoke |

Pipeline: build, run 237 tests, run benchmarks, generate test data, smoke test full simulation. Results cached by CMake build directory.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/methods.md](docs/methods.md) | Paper-ready methods: neuron/synapse/plasticity models, equations, parameter tables, citations |
| [docs/architecture.md](docs/architecture.md) | Software architecture: module dependencies, data flow, file formats, performance |
| [docs/api_reference.md](docs/api_reference.md) | API reference for all public structs and functions |
| [docs/bidirectional_neural_twinning.md](docs/bidirectional_neural_twinning.md) | Research outline and motivation |

---

## Data sources

- Connectome: [FlyWire](https://flywire.ai) (Dorkenwald et al., 2024)
- Neurotransmitter predictions: Eckstein et al., 2024

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
