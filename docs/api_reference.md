# API Reference

> Core data structures, neuron functions, connectome I/O, parametric generation, and plasticity APIs are documented in [mechabrain/docs/api_reference.md](../../mechabrain/docs/api_reference.md). This file covers the FWMC bridge, experiment, and GPU APIs.

All public types reside in the `fwmc` and `mechabrain` namespaces. Include the relevant header to access each component.

---

## Core Data Structures

### NeuronArray
**Header**: `core/neuron_array.h`

Structure-of-arrays neuron storage. Each array index `i` refers to the same neuron. Fields include membrane potential `v` (mV), recovery variable `u`, synaptic input `i_syn`, external current `i_ext`, spike flag `spiked` (uint8), FlyWire root ID, 3D position `(x, y, z)`, cell type and region indices, neuromodulator concentrations `(dopamine, serotonin, octopamine)`, and `last_spike_time`.

- `void Resize(size_t count)`:Allocate all arrays to `count` neurons with default initial values (v=-65, u=-13, last_spike_time=-1e9).
- `void ClearSynapticInput()`:Zero all `i_syn` entries.
- `int CountSpikes() const`:Return the number of neurons with `spiked == 1`.

```cpp
NeuronArray neurons;
neurons.Resize(10000);
neurons.ClearSynapticInput();
int n = neurons.CountSpikes();
```

### SynapseTable
**Header**: `core/synapse_table.h`

CSR (Compressed Sparse Row) synapse storage. Sorted by pre-synaptic neuron for cache-friendly spike propagation. Fields: `row_ptr` (CSR index), `post` (post-synaptic indices), `weight`, `nt_type` (NTType enum).

- `size_t Size() const`:Return total synapse count.
- `static float Sign(uint8_t nt)`:Return -1.0 for GABA and Glutamate (GluCl in *Drosophila*), +1.0 for all others.
- `void BuildFromCOO(size_t num_neurons, pre, post, weight, nt)`:Sort COO data and build CSR index.
- `void PropagateSpikes(const uint8_t* spiked, float* i_syn, float weight_scale) const`:Deliver weighted synaptic current from all spiking pre-neurons to their post-synaptic targets.

```cpp
SynapseTable synapses;
synapses.BuildFromCOO(n, pre_vec, post_vec, weight_vec, nt_vec);
synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
```

### IzhikevichParams
**Header**: `core/izhikevich.h`

Parameters for the Izhikevich neuron model: `a` (recovery rate, default 0.02), `b` (recovery sensitivity, default 0.2), `c` (reset voltage, default -65 mV), `d` (after-spike reset increment, default 8), `v_thresh` (spike threshold, default 30 mV).

### LIFParams
**Header**: `core/izhikevich.h`

Parameters for the leaky integrate-and-fire model: `tau_ms` (membrane time constant, default 20), `v_rest` (resting potential, default -70 mV), `v_thresh` (spike threshold, default -55 mV), `v_reset` (reset voltage, default -70 mV), `r_membrane` (membrane resistance, default 10).

### STDPParams
**Header**: `core/stdp.h`

STDP rule parameters: `a_plus` (potentiation amplitude, default 0.01), `a_minus` (depression amplitude, default 0.012), `tau_plus` / `tau_minus` (time constants, default 20 ms each), `w_min` / `w_max` (weight bounds, default 0 / 10), `dopamine_gated` (enable DA modulation, default false), `da_scale` (DA modulation strength, default 5.0).

### ExperimentConfig
**Header**: `core/experiment_config.h`

Complete experiment specification: metadata (name, fly strain, date, notes), simulation parameters (dt_ms, duration_ms, weight_scale, metrics_interval, enable_stdp), bridge mode (0=open-loop, 1=shadow, 2=closed-loop), replacement thresholds (monitor, bridge, resync), calibration settings (interval, learning rate), stimulus protocol (ordered list of `StimulusEvent`), data paths (connectome_dir, recording_input, output_dir), and recording options (record_spikes, record_voltages, record_shadow_metrics, record_per_neuron_error, recording_interval).

### StimulusEvent
**Header**: `core/experiment_config.h`

A timed stimulus event: `start_ms`, `end_ms`, `intensity` (normalized [0,1]), `label` (string), and `target_neurons` (vector of neuron indices).

---

## Core Functions

### IzhikevichStep
**Header**: `core/izhikevich.h`

```cpp
void IzhikevichStep(NeuronArray& neurons, float dt_ms,
                    float sim_time_ms, const IzhikevichParams& p);
```

Advance all neurons by one timestep using the Izhikevich model with uniform parameters. Uses two half-step integration for stability. NaN/Inf guard resets divergent neurons. OpenMP parallelized for neuron counts > 10,000.

### IzhikevichStepHeterogeneous
**Header**: `core/cell_types.h`

```cpp
void IzhikevichStepHeterogeneous(NeuronArray& neurons, float dt_ms,
                                  float sim_time_ms,
                                  const CellTypeManager& types);
```

Same as `IzhikevichStep` but uses per-neuron parameters from a `CellTypeManager`. Required for parametric brains with multiple cell types.

### LIFStep
**Header**: `core/izhikevich.h`

```cpp
void LIFStep(NeuronArray& neurons, float dt_ms,
             float sim_time_ms, const LIFParams& p);
```

Advance all neurons using the leaky integrate-and-fire model. Faster than Izhikevich but captures fewer firing patterns. NaN/Inf guard and OpenMP parallelization included.

### STDPUpdate
**Header**: `core/stdp.h`

```cpp
void STDPUpdate(SynapseTable& synapses, const NeuronArray& neurons,
                float sim_time_ms, const STDPParams& p);
```

Update synaptic weights based on spike timing. For each synapse, compute timing difference between pre- and post-synaptic spikes and apply exponential potentiation or depression. If `p.dopamine_gated` is true, weight changes are scaled by the post-synaptic dopamine concentration. OpenMP parallelized over pre-synaptic neurons for >10k neurons (no write conflicts since CSR layout assigns each synapse to exactly one pre-neuron).

### NeuromodulatorUpdate
**Header**: `core/stdp.h`

```cpp
void NeuromodulatorUpdate(NeuronArray& neurons,
                          const SynapseTable& synapses, float dt_ms);
```

Decay existing neuromodulator concentrations and release from spiking neuromodulatory neurons. DAN neurons release dopamine to post-synaptic targets; fast-spiking interneurons release octopamine. All concentrations clamped to [0, 1].

### PropagateSpikes
**Header**: `core/synapse_table.h` (member of `SynapseTable`)

```cpp
void PropagateSpikes(const uint8_t* spiked, float* i_syn,
                     float weight_scale) const;
```

For each pre-synaptic neuron that spiked, traverse its outgoing CSR synapses and add `Sign(nt) * weight * weight_scale` to the post-synaptic `i_syn` accumulator. This is the computational hot loop. OpenMP parallelized over pre-synaptic neurons for >10k neurons, with `#pragma omp atomic` for i_syn accumulation.

### IzhikevichStepFast
**Header**: `core/izhikevich.h`

```cpp
void IzhikevichStepFast(NeuronArray& neurons, float dt_ms,
                        float sim_time_ms, const IzhikevichParams& p);
```

Dispatch function that selects AVX2-vectorized (`IzhikevichStepAVX2`, 8 neurons per iteration using `__m256` intrinsics with FMA) or scalar `IzhikevichStep` based on compile-time detection. ~87M neurons/sec on AVX2 hardware.

### ProprioMap
**Header**: `core/proprioception.h`

Assigns VNC neurons to proprioceptive sensory channels and injects body state as excitatory currents.

- `void Init(const NeuronArray& neurons, uint8_t vnc_region, float midline_x)`: Auto-assign first 30% of VNC neurons across 53 sensory channels (42 joint angle, 6 contact, 3 body velocity, 2 haltere L/R).
- `void Inject(NeuronArray& neurons, const ProprioState& state, const ProprioConfig& cfg) const`: Convert body state to currents and inject into assigned neurons. Joint angles use sigmoid activation; contacts are strong binary signals; haltere feedback is asymmetric L/R based on yaw rate.

Supporting types: `ProprioConfig` (gain parameters), `ProprioState` (42 joint angles, 42 velocities, 6 contacts, 3 body velocity), `ReadProprioFromMuJoCo<MjModel, MjData>()` (template function extracting state from MuJoCo qpos/qvel/contacts).

### CPGOscillator
**Header**: `core/cpg.h`

Central pattern generator injecting oscillatory current into VNC motor neurons for spontaneous rhythmic locomotion.

- `void Init(const NeuronArray& neurons, uint8_t vnc_region, float midline_x, float sensory_fraction)`: Split VNC motor neurons (after sensory fraction) into two anti-phase groups by x-coordinate.
- `void Step(NeuronArray& neurons, float dt_ms, float descending_drive)`: Advance phase at `frequency_hz` (default 8 Hz), inject `tonic_drive + sin(phase) * amplitude * drive_scale` to group A and anti-phase to group B. `drive_scale` smoothly tracks `descending_drive` with a 50ms time constant.

---

## Connectome I/O

### ConnectomeLoader
**Header**: `core/connectome_loader.h`

Static methods for loading binary connectome files.

- `static Result<size_t> LoadNeurons(const string& path, NeuronArray& neurons)`:Read `neurons.bin`. Returns neuron count or error. Validates count bounds (max 10M neurons).
- `static Result<size_t> LoadSynapses(const string& path, size_t n_neurons, SynapseTable& table)`:Read `synapses.bin`, validate index bounds, and build CSR. Returns synapse count or error.

```cpp
NeuronArray neurons;
SynapseTable synapses;
auto nr = ConnectomeLoader::LoadNeurons("data/neurons.bin", neurons);
auto sr = ConnectomeLoader::LoadSynapses("data/synapses.bin", neurons.n, synapses);
```

### ConnectomeExport
**Header**: `core/connectome_export.h`

Static methods for writing binary connectome files.

- `static Result<size_t> ExportNeurons(const string& path, const NeuronArray& neurons)`:Write `neurons.bin` from a NeuronArray.
- `static Result<size_t> ExportSynapses(const string& path, const SynapseTable& table)`:Reconstruct COO from CSR and write `synapses.bin`.

---

## Bridge System

### TwinBridge
**Header**: `bridge/twin_bridge.h`

Main bridge controller that orchestrates the read channel, digital twin simulation, shadow tracking, neuron replacement, and write channel. Key fields: `digital` (NeuronArray), `synapses` (SynapseTable), `shadow` (ShadowTracker), `replacer` (NeuronReplacer), `writer` (OptogeneticWriter), `read_channel` / `write_channel` (polymorphic I/O), `mode` (BridgeMode enum).

- `void Init(size_t n_neurons)`:Resize digital twin, initialize replacer and writer safety state.
- `void BuildAdjacency()`:Build post-synaptic adjacency list from CSR for adaptive boundary detection.
- `void Step()`:Execute one bridge loop iteration with hierarchical tick structure (fast/medium/slow).
- `void Run(float duration_ms, int metrics_interval)`:Run the bridge for a specified duration with periodic logging.

```cpp
TwinBridge bridge;
bridge.Init(10000);
bridge.mode = BridgeMode::kShadow;
bridge.dt_ms = 0.1f;
bridge.Run(5000.0f, 1000);
```

### ShadowTracker
**Header**: `bridge/shadow_tracker.h`

Measures divergence between digital predictions and biological observations. Maintains a history of `DriftSnapshot` records.

- `DriftSnapshot Measure(const NeuronArray& digital, const vector<BioReading>& bio, float sim_time_ms)`:Compute Pearson correlation, RMSE, voltage error, and false positive/negative counts. Appends to history.
- `void Resync(NeuronArray& digital, const vector<BioReading>& bio, float sim_time_ms)`:Copy biological state into the digital twin (voltage or spike injection).
- `bool DriftExceedsThreshold(float threshold) const`:Return true if the latest spike correlation is below threshold.

### NeuronReplacer
**Header**: `bridge/neuron_replacer.h`

Manages the four-state replacement machine (BIOLOGICAL, MONITORED, BRIDGED, REPLACED) with hysteresis, rollback, and adaptive boundaries.

- `void Init(size_t n)`:Initialize all neurons to BIOLOGICAL state.
- `void BeginMonitoring(const vector<uint32_t>& indices)`:Promote specified neurons from BIOLOGICAL to MONITORED.
- `void UpdateCorrelation(uint32_t idx, float correlation, float dt_ms)`:Update EMA correlation for a single neuron.
- `vector<uint32_t> TryAdvance()`:Attempt to promote neurons that meet threshold and observation time criteria. Returns indices of promoted neurons.
- `vector<uint32_t> RollbackDiverged(float threshold)`:Demote BRIDGED neurons with correlation below threshold to MONITORED.
- `vector<uint32_t> AutoPromoteNeighbors(const vector<vector<uint32_t>>& neighbors, float drift_threshold)`:Promote BIOLOGICAL neighbors of drifting neurons to MONITORED.
- `float ReplacementFraction() const`:Return fraction of neurons in REPLACED state.

### OptogeneticWriter
**Header**: `bridge/optogenetic_writer.h`

Converts digital spike decisions to stimulation commands with safety constraints (refractory period, thermal limit, SLM target cap, galvo-SLM split, predictive pre-staging). Optionally integrates opsin kinetics and tissue light model.

- `void InitSafety(size_t n_neurons)`:Initialize per-neuron safety state.
- `float PowerCurve(float voltage) const`:Compute nonlinear opsin activation (Hill equation).
- `vector<StimCommand> GenerateCommands(const NeuronArray& digital, const vector<BioReading>& bio_state, float sim_time_ms)`:Generate stimulation commands for the current timestep, respecting all safety limits.
- `GalvoSLMSplit SplitGalvoSLM(const vector<StimCommand>& commands) const`:Split commands into galvo (fast, up to 3) and SLM (batch) channels.
- `void PreStagePatterns(...)`:Pre-compute future hologram patterns using voltage-trend prediction.
- `void InitOpsinModel(size_t n, OpsinType excitatory, OpsinType inhibitory)`:Enable three-state opsin kinetics.
- `void InitLightModel(float laser_power_mw, float na)`:Enable tissue light attenuation model.
- `void ApplyOpsinStep(const vector<StimCommand>& commands, NeuronArray& neurons, float dt_ms)`:Translate commands into irradiance, step opsin kinetics, inject photocurrents.

### OpsinPopulation
**Header**: `bridge/opsin_model.h`

Three-state channelrhodopsin kinetic model (Closed, Open, Desensitized). Computes photocurrent as `I = g_max * open_fraction * (V - E_rev)`. Supports ChR2, ChRmine, and stGtACR2 via `OpsinType` enum.

- `void Init(size_t n_neurons, OpsinType type)`:Initialize with literature parameters.
- `void SetIrradiance(uint32_t idx, float power)`:Set light intensity for a neuron (mW/mm^2).
- `void Step(float dt_ms, const float* v, float* i_ext, size_t n)`:Advance kinetics, inject photocurrent.
- `float OpenFraction(uint32_t idx) const`:Fraction of channels in open state.
- `float DesensitizedFraction(uint32_t idx) const`:Fraction in desensitized state.

### LightModel
**Header**: `bridge/light_model.h`

Tissue optics for two-photon optogenetic stimulation. Beer-Lambert attenuation with Gaussian beam point spread function.

- `float IrradianceAt(float x, float y, float z, float lambda_nm) const`:Irradiance at a 3D position (mW/mm^2).
- `float LateralResolution(float lambda_nm) const`:Beam waist (um) for two-photon excitation.
- `float MaxDepth(float lambda_nm) const`:Maximum usable depth (um) before 1/e attenuation.
- `void ComputeMultiSpotIrradiance(...)`:Irradiance for holographic multi-spot with power splitting.

### ValidationEngine
**Header**: `bridge/validation.h`

Compares simulated vs recorded spike trains using multiple metrics.

- `NeuronValidation ValidateNeuron(const SpikeTrain& sim, const SpikeTrain& rec, float duration_ms) const`:Single-neuron comparison (rate, correlation, van Rossum distance, F1).
- `PopulationValidation ValidatePopulation(const vector<SpikeTrain>& sim, const vector<SpikeTrain>& rec, float duration_ms) const`:Population-level summary.
- `vector<WindowStats> SlidingWindowAnalysis(const SpikeTrain& sim, const SpikeTrain& rec, float duration_ms) const`:Temporal statistics.
- `static void RecordSpikes(const NeuronArray& neurons, float t, vector<SpikeTrain>& trains)`:Accumulate spike times from simulation.
- `static MatchResult MatchSpikes(const vector<float>& sim, const vector<float>& rec, float window_ms)`:Greedy spike matching with tolerance.

---

## Experiment Management

### ExperimentRunner
**Header**: `experiment_runner.h`

Top-level experiment orchestrator. Wires connectome loading, bridge configuration, stimulus protocols, recording, calibration, and checkpointing into a single `Run()` call.

- `int Run()`:Execute the full experiment from config. Returns 0 on success. Loads connectome, validates, configures bridge and channels, runs simulation loop, records data, saves final checkpoint.

```cpp
ExperimentRunner runner;
runner.config.name = "phase1";
runner.config.connectome_dir = "data";
runner.config.duration_ms = 10000;
runner.Run();
```

### ExperimentProtocol (StimulusController)
**Header**: `bridge/stimulus.h`

Applies timed stimulus events to neurons. Events are sorted by start time for efficient scanning.

- `void LoadProtocol(const vector<StimulusEvent>& events)`:Load and sort stimulus protocol.
- `void Apply(float sim_time_ms, NeuronArray& neurons)`:Inject external current from all active events at the given time. Current magnitude = `intensity * 15.0` (scaled to suprathreshold).
- `vector<const StimulusEvent*> ActiveAt(float sim_time_ms) const`:Query which events are currently active.

### Calibrator
**Header**: `bridge/calibrator.h`

Perturbation-based gradient-free weight optimizer for per-fly calibration.

- `void Init(size_t n_synapses)`:Allocate velocity and error accumulators.
- `void AccumulateError(const SynapseTable& synapses, const NeuronArray& digital, const vector<BioReading>& bio)`:For each synapse where the pre-neuron spiked, attribute prediction error at the post-neuron to that synapse.
- `void ApplyGradients(SynapseTable& synapses)`:Apply momentum SGD weight updates from accumulated error. Resets accumulators.
- `float MeanError(const NeuronArray& digital, const vector<BioReading>& bio) const`:Compute mean absolute prediction error across observed neurons.

---

## Parametric Generation

### ParametricGenerator
**Header**: `core/parametric_gen.h`

Generates NeuronArray and SynapseTable from a BrainSpec.

- `uint32_t Generate(const BrainSpec& spec, NeuronArray& neurons, SynapseTable& synapses, CellTypeManager& types)`:Create all neurons, assign cell types and regions, generate intra-region and inter-region synapses, build CSR, assign per-neuron Izhikevich parameters. Returns total neuron count.
- `vector<RegionRange> region_ranges`:After generation, maps region names to contiguous neuron index ranges.

```cpp
BrainSpec spec;
spec.name = "test";
spec.regions.push_back({"AL", 500, 0.1f, kACh, {{CellType::kPN_excitatory, 0.5f}}});

ParametricGenerator gen;
NeuronArray neurons;
SynapseTable synapses;
CellTypeManager types;
gen.Generate(spec, neurons, synapses, types);
```

### BrainSpecLoader
**Header**: `core/brain_spec_loader.h`

Parses `.brain` specification files into a `BrainSpec` struct. Supports region definitions, projections, stimuli, background noise (`background_mean`, `background_std`), and global parameters.

- `static Result<BrainSpec> Load(const string& path)`:Parse a `.brain` file and return a BrainSpec.

### ParamSweep
**Header**: `core/param_sweep.h`

Parameter sweep engine for auto-tuning Izhikevich parameters.

- `void GridSweep(CellType target, const NeuronArray& base, const SynapseTable& synapses, ScoreFn score_fn)`:Exhaustive grid search over (a, b, c, d) space.
- `void RandomSweep(...)`:Uniform random sampling of parameter space.
- `void Refine(CellType target, ..., int iterations, float step_size)`:Stochastic hill-climbing from the best point found.
- `IzhikevichParams BestParams() const`:Return the highest-scoring parameter set.

Built-in scoring functions (in `fwmc::scoring` namespace):
- `ScoreFn TargetFiringRate(float target_hz, float dt_ms)`:Score inversely proportional to firing rate error.
- `ScoreFn ActivityInRange(float min_fraction, float max_fraction)`:Score based on fraction of active neurons being within bounds.
- `ScoreFn RealisticCV(float target_cv)`:Score based on coefficient of variation of spike timing.

### ParametricSync
**Header**: `core/parametric_sync.h`

Adaptive sync engine for tuning a parametric model to match a reference brain.

- `void Init(size_t n_neurons, size_t n_synapses)`:Allocate per-neuron and per-synapse state.
- `void Step(NeuronArray& model, SynapseTable& synapses, const NeuronArray& ref, CellTypeManager& types)`:Run one sync step: corrective current (fast), weight error accumulation (medium), parameter nudges (slow).
- `bool HasConverged() const`:Return true if the target fraction of neurons have converged.
- `const SyncSnapshot& Latest() const`:Get the most recent sync metrics.

---

## Metrics and Plasticity

### RegionMetrics
**Header**: `core/region_metrics.h`

Per-region activity tracking during parametric brain simulation.

- `void Init(const ParametricGenerator& gen)`:Initialize with region ranges from the generator.
- `void Record(const NeuronArray& neurons, float sim_time_ms, float dt_ms, int window_steps)`:Record a snapshot of per-region spike counts, firing rates, fraction active, and mean voltage.
- `void LogLatest() const`:Log the most recent snapshot.
- `void LogSummary() const`:Log cumulative statistics (total spikes, peak rate, mean fraction active) for all regions.

### ApplyStimuli (free function)
**Header**: `core/region_metrics.h`

```cpp
void ApplyStimuli(const vector<StimulusSpec>& stimuli,
                  const vector<RegionRange>& regions,
                  NeuronArray& neurons, float sim_time_ms, uint32_t seed);
```

Apply parametric stimulus specifications to neurons based on region ranges. Only active stimuli (within their time window) inject current.

### StructuralPlasticity
**Header**: `core/structural_plasticity.h`

Synapse pruning and sprouting.

- `size_t PruneWeak(SynapseTable& syn)`:Zero out synapses with weight below threshold. Returns count pruned.
- `size_t SproutNew(SynapseTable& syn, NeuronArray& neurons, mt19937& rng)`:Create new excitatory synapses between co-active neurons. Rebuilds CSR. Returns count sprouted.
- `void Update(SynapseTable& syn, NeuronArray& neurons, int step, mt19937& rng)`:Called each step; only acts at update_interval boundaries.

Configuration via `StructuralPlasticity::Config`: `prune_threshold` (default 0.05), `sprout_rate` (default 0.001), `update_interval` (default 5000 steps), `max_synapses_per_neuron` (default 100).

### GapJunctionTable
**Header**: `core/gap_junctions.h`

Electrical gap junction storage. Bidirectional current `I = g * (Vb - Va)` between connected neurons.

- `void AddJunction(uint32_t a, uint32_t b, float g)`: Add a single gap junction with conductance `g`.
- `void PropagateGapCurrents(NeuronArray& neurons) const`: Inject gap currents into `i_ext`. OpenMP parallelized with atomic accumulation for >10k junctions.
- `void BuildFromRegion(const NeuronArray& neurons, uint8_t region, float density, float g_default, uint32_t seed)`: Connect all neuron pairs within a region with probability `density`.

### UpdateSTP
**Header**: `core/short_term_plasticity.h`

Tsodyks-Markram short-term plasticity update for SynapseTable.

- `void UpdateSTP(SynapseTable& synapses, const NeuronArray& neurons, float dt_ms)`: Relax u and x toward resting values, then apply spike updates for firing pre-neurons. Requires `SynapseTable::InitSTP()`.
- `void ResetSTP(SynapseTable& synapses)`: Restore resting state (u=U_se, x=1).
- `STPParams STPFacilitating()`, `STPDepressing()`, `STPCombined()`: Preset factories.
- `float MeanSTPUtilization(const SynapseTable&)`, `float MeanSTPResources(const SynapseTable&)`: Diagnostics.

### NWBExporter
**Header**: `core/nwb_export.h`

Lightweight NWB-compatible exporter producing CSV spike/voltage files and JSON session metadata.

- `bool BeginSession(const string& dir, const string& description, const NeuronArray& neurons)`: Create output directory, open CSVs, snapshot neuron metadata.
- `void SetVoltageSubset(const vector<uint32_t>& neuron_indices)`: Configure which neurons get voltage traces.
- `void RecordTimestep(float time_ms, const NeuronArray& neurons)`: Record spikes and voltages for one step.
- `void AddStimulus(float start_ms, float stop_ms, const string& name, const string& desc)`: Register stimulus for metadata.
- `void EndSession()`: Flush CSVs, write `session.nwb.json` with NWB 2.7 schema.

### CellTypeManager
**Header**: `core/cell_types.h`

Manages per-neuron Izhikevich parameters based on cell type assignments.

- `void AssignFromTypes(const NeuronArray& neurons)`:Populate `neuron_params` from each neuron's type field, applying overrides where set.
- `const IzhikevichParams& Get(size_t idx) const`:Get parameters for a single neuron.
- `void SetOverride(CellType ct, const IzhikevichParams& p)`:Override default parameters for a cell type. Call `AssignFromTypes()` after to apply.

---

## Checkpoint and Recording

### Checkpoint
**Header**: `core/checkpoint.h`

Binary checkpoint save/load for full simulation state.

- `static bool Save(const string& path, float sim_time_ms, int total_steps, int total_resyncs, const NeuronArray&, const SynapseTable&, const NeuronReplacer&, const ShadowTracker&)`:Save complete state. Creates parent directories. Returns true on success.
- `static bool Load(const string& path, float& sim_time_ms, int& total_steps, int& total_resyncs, NeuronArray&, SynapseTable&, NeuronReplacer&, ShadowTracker&)`:Load state. Validates magic number, version, and size consistency. Returns true on success.

### Recorder
**Header**: `core/recorder.h`

Binary and CSV recording to disk.

- `bool Open(const string& dir, uint32_t num_neurons)`:Create output directory and open output files (spikes.bin, voltages.bin, metrics.csv, per_neuron_error.bin) based on recording flags.
- `void RecordStep(float time_ms, const NeuronArray&, const DriftSnapshot*, int total_resyncs, float replaced_pct, const vector<float>* per_neuron_err)`:Write one timestep of data to all active output files.
- `void Close()`:Patch step counts in binary headers and close all files.

---

## Bridge Channels

### ReadChannel (abstract)
**Header**: `bridge/bridge_channel.h`

Interface for biological-to-digital data flow.

- `virtual vector<BioReading> ReadFrame(float sim_time_ms) = 0`
- `virtual size_t NumMonitored() const = 0`
- `virtual float SampleRateHz() const = 0`

### WriteChannel (abstract)
**Header**: `bridge/bridge_channel.h`

Interface for digital-to-biological commands.

- `virtual void WriteFrame(const vector<StimCommand>& commands) = 0`
- `virtual size_t MaxTargets() const = 0`
- `virtual float MinISI() const = 0`

### BioReading
**Header**: `bridge/bridge_channel.h`

A single neuron observation: `neuron_idx` (uint32), `spike_prob` (float, 0-1), `calcium_raw` (float, dF/F), `voltage_mv` (float, NaN if unavailable).

### StimCommand
**Header**: `bridge/bridge_channel.h`

A single stimulation command: `neuron_idx` (uint32), `intensity` (float, 0-1), `excitatory` (bool, CsChrimson vs GtACR1), `duration_ms` (float).

### CallbackReadChannel / CallbackWriteChannel
**Header**: `bridge/hardware_channel.h`

User-provided lambda-based channel implementations. Simplest integration path for custom hardware.

```cpp
auto read = CallbackReadChannel(
    [](float t) { return myHardware.GetReadings(t); },
    n_neurons, 1000.0f);
bridge.read_channel = std::make_unique<CallbackReadChannel>(std::move(read));
```

### SharedMemoryReadChannel / SharedMemoryWriteChannel
**Header**: `bridge/hardware_channel.h`

Read/write channels backed by a shared memory buffer. Layout: `[uint32_t count] [struct * count]`. Enables zero-copy IPC with external acquisition processes.

### RingBuffer\<T\>
**Header**: `bridge/hardware_channel.h`

Lock-free single-producer single-consumer ring buffer for streaming between acquisition and simulation threads.

- `void Init(size_t capacity)`:Allocate buffer.
- `bool Push(const T& item)`:Enqueue (returns false if full).
- `bool Pop(T& item)`:Dequeue (returns false if empty).
- `size_t Available() const`:Number of items ready to read.

### Hardware Config Structs
**Header**: `bridge/hardware_channel.h`

Configuration structs for common experimental platforms:
- `OpenEphysConfig`:Neuropixels spike-sorted data via ZMQ, unit-to-neuron mapping
- `ScanImageConfig`:Two-photon calcium imaging frame rate, ROI definitions
- `BonsaiConfig`:OSC addresses for reactive data flow

### SpikeDecoder
**Header**: `bridge/spike_decoder.h`

Multi-timescale calcium-to-spike deconvolution approximating CASCADE.

- `void Init(size_t n_neurons)`:Allocate per-neuron decoder state.
- `vector<BioReading> Decode(const vector<float>& raw_calcium, const vector<uint32_t>& neuron_indices, float dt_ms)`:Full multi-timescale deconvolution using three exponential kernels (fast 20ms, medium 100ms, slow 500ms) with adaptive thresholding.
- `vector<BioReading> DecodeSelective(const vector<float>& raw_calcium, const vector<uint32_t>& neuron_indices, float dt_ms, const vector<bool>& active_set)`:Full decode for active-set neurons; cheap threshold-only approximation for others.

---

## GPU Support

### GPUManager
**Header**: `cuda/gpu_manager.h` (requires `FWMC_CUDA` define)

Device lifecycle and memory management.

- `void Init(int device)`:Select CUDA device and create transfer stream.
- `void Shutdown()`:Destroy transfer stream.
- `NeuronArrayGPU AllocateNeurons(size_t n)`:Allocate device memory for neuron arrays.
- `SynapseTableGPU AllocateSynapses(size_t n_neurons, size_t n_synapses)`:Allocate device memory for CSR synapse table.
- `void UploadNeurons(const NeuronArray& cpu, NeuronArrayGPU& gpu)`:Async host-to-device transfer.
- `void DownloadNeurons(const NeuronArrayGPU& gpu, NeuronArray& cpu)`:Async device-to-host transfer.
- `void UploadSynapses(const SynapseTable& cpu, SynapseTableGPU& gpu)`:Async synapse upload.
- `void DownloadSynapseWeights(const SynapseTableGPU& gpu, SynapseTable& cpu)`:Download modified weights after STDP.
- `void SyncTransfers()`:Block until all async transfers complete.
