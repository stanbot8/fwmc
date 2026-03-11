# brain-model

Header-only C++23 spiking neural network engine for whole-brain scale simulation. Designed around the *Drosophila melanogaster* connectome (~140k neurons, ~50M synapses) but general enough for any spiking network.

No external dependencies. Optional OpenMP parallelism.

## Core modules

### Neuron dynamics

| File | Description |
|------|-------------|
| [neuron_array.h](core/neuron_array.h) | SoA (structure-of-arrays) neuron state: membrane voltage, recovery variable, synaptic current, spike flags, neuromodulator concentrations |
| [izhikevich.h](core/izhikevich.h) | Izhikevich model (2 coupled ODEs, 20+ firing patterns) and leaky integrate-and-fire. Half-step integration, NaN recovery, OpenMP parallel |
| [cell_types.h](core/cell_types.h) | Per-neuron heterogeneous dynamics. Biologically tuned parameters for KC, PN, LN, MBON, DAN, ORN, and fast-spiking cell types |

### Connectivity

| File | Description |
|------|-------------|
| [synapse_table.h](core/synapse_table.h) | CSR (compressed sparse row) synapse graph. NT-aware sign, stochastic vesicle release, Tsodyks-Markram short-term plasticity. OpenMP spike propagation with atomic accumulation |
| [connectome_loader.h](core/connectome_loader.h) | Binary connectome reader with `std::expected` error handling |
| [connectome_export.h](core/connectome_export.h) | Export parametric brains to binary format |
| [connectome_stats.h](core/connectome_stats.h) | Degree distributions, NT ratios, weight statistics, integrity checks |

### Plasticity and synaptic dynamics

| File | Description |
|------|-------------|
| [stdp.h](core/stdp.h) | Spike-timing-dependent plasticity with dopamine-gated modulation (Izhikevich 2007). Exponential timing windows, bounded weights |
| [short_term_plasticity.h](core/short_term_plasticity.h) | Tsodyks-Markram STP update, preset factories (facilitating, depressing, combined), diagnostics |
| [structural_plasticity.h](core/structural_plasticity.h) | Synapse pruning (weak weights) and sprouting (correlated co-firing) |
| [gap_junctions.h](core/gap_junctions.h) | Electrical gap junctions: bidirectional current, region-based construction, OpenMP parallel |

### Parametric generation

| File | Description |
|------|-------------|
| [parametric_gen.h](core/parametric_gen.h) | Generate connectomes from high-level brain specs: regions, cell type distributions, NT ratios, inter-region projections |
| [brain_spec_loader.h](core/brain_spec_loader.h) | Parser for `.brain` config files |
| [param_sweep.h](core/param_sweep.h) | Grid search + stochastic hill-climbing for neuron parameter tuning |
| [parametric_sync.h](core/parametric_sync.h) | Three-timescale sync engine: corrective current (fast), weight SGD (medium), parameter nudge (slow) |
| [region_metrics.h](core/region_metrics.h) | Per-region spike counts, firing rates, fraction active, mean voltage |

### Experiments and I/O

| File | Description |
|------|-------------|
| [experiment_config.h](core/experiment_config.h) | Experiment parameters, cell types, regions, stimulus events |
| [experiment_protocol.h](core/experiment_protocol.h) | Multi-phase protocols with transition conditions |
| [config_loader.h](core/config_loader.h) | Key-value config file parser |
| [checkpoint.h](core/checkpoint.h) | Binary save/load of full simulation state with extension blobs |
| [recorder.h](core/recorder.h) | Binary and CSV recording of spikes, voltages, drift metrics |
| [nwb_export.h](core/nwb_export.h) | NWB-compatible export: spike CSV, voltage CSV, session metadata JSON |
| [stimulus_event.h](core/stimulus_event.h) | Stimulus event types |

### Sensorimotor

| File | Description |
|------|-------------|
| [motor_output.h](core/motor_output.h) | Maps descending neuron spike rates to fictive locomotion commands (forward velocity, turning, approach/avoid) |
| [proprioception.h](core/proprioception.h) | Maps MuJoCo body state (joint angles, contacts, velocity, haltere rotation) to VNC sensory neuron currents |
| [cpg.h](core/cpg.h) | Central pattern generator: anti-phase oscillatory drive to VNC motor neurons for tripod gait locomotion |
| [intrinsic_homeostasis.h](core/intrinsic_homeostasis.h) | Target firing rate maintenance via excitability adaptation |

### Infrastructure

| File | Description |
|------|-------------|
| [platform.h](core/platform.h) | Compiler portability: `FWMC_RESTRICT`, `FWMC_PACK`, MSVC/GCC/Clang detection |
| [error.h](core/error.h) | `Result<T> = std::expected<T, Error>` with error codes |
| [log.h](core/log.h) | Minimal `std::format`-based logger with timestamp and level tags |
| [version.h](core/version.h) | Version constants |

## Tissue volume

The [tissue/](tissue/) subdirectory provides a procedural 3D brain volume with Wilson-Cowan neural field dynamics, voxel-based neuromodulator diffusion, and a level-of-detail manager. See [tissue/README.md](tissue/README.md).

## Design principles

- **Header-only**: every module is a single `.h` file. Include what you need, no link step.
- **Flat arrays**: SoA layout for cache-friendly iteration and easy SIMD/GPU porting. No virtual dispatch on hot paths.
- **No framework dependencies**: standard C++23 only. OpenMP is optional (`#ifdef _OPENMP` guards).
- **Biologically grounded**: neuron parameters, NT classifications, and circuit motifs drawn from FlyWire data (Dorkenwald et al. 2024) and published Drosophila electrophysiology.
