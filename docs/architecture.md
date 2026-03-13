# Software Architecture

## System Overview

```
+------------------------------------------------------------------+
|                         FWMC Simulator                            |
|                                                                   |
|  +---------------------+    +---------------------+              |
|  |    fwmc.cc (CLI)     |    | ExperimentRunner    |              |
|  |  --data / --param    |    |  --experiment .cfg  |              |
|  +----------+----------+    +----------+----------+              |
|             |                          |                          |
|             v                          v                          |
|  +--------------------------------------------------+            |
|  |              Simulation Loop                      |            |
|  |  clear I_syn -> propagate -> stimulus -> step ->  |            |
|  |  STDP -> neuromod -> record -> checkpoint         |            |
|  +--------------------------------------------------+            |
|             |                          |                          |
|     +-------+-------+         +-------+-------+                  |
|     v               v         v               v                  |
|  +---------+   +---------+  +---------+   +---------+            |
|  |  Core   |   | Bridge  |  |  CUDA   |   | Scripts |            |
|  +---------+   +---------+  +---------+   +---------+            |
|                                                                   |
+------------------------------------------------------------------+
```

## Module Dependency Graph

```
                        Experiment Layer
                  +---------------------------+
                  | ExperimentRunner           |
                  | fwmc.cc (CLI entry point)  |
                  +---------------------------+
                       |              |
          +------------+              +------------+
          v                                        v
    Bridge Layer                             Core Layer
+-------------------+                 +-------------------+
| TwinBridge        |                 | NeuronArray       |
| ShadowTracker     |-----depends--->| SynapseTable      |
| NeuronReplacer    |     on         | IzhikevichStep    |
| OptogeneticWriter |                | LIFStep           |
| OpsinModel        |                | STDPUpdate        |
| LightModel        |                | NeuromodulatorUpd |
| ValidationEngine  |                | CellTypeManager   |
| SpikeDecoder      |                | ConnectomeLoader  |
| Calibrator        |                | ConnectomeExport  |
| StimulusCtrl      |                | ConnectomeStats   |
| BridgeChannel     |                | ParametricGen     |
| HardwareChannel   |                | BrainSpecLoader   |
| FileReadChannel   |                | ParamSweep        |
+-------------------+                | ParametricSync    |
                                     | RegionMetrics     |
                                     | StructuralPlast.  |
                                     | Proprioception    |
                                     | CPGOscillator     |
                                     | MotorOutput       |
                                     | Checkpoint        |
                                     | Recorder          |
                                     | ExperimentConfig  |
                                     | ConfigLoader      |
                                     +-------------------+
                                              |
                                     +--------+--------+
                                     v                  v
                              +-----------+     +-----------+
                              | Platform  |     | Log/Error |
                              | (macros)  |     | Version   |
                              +-----------+     +-----------+

    GPU Layer (optional, behind FWMC_CUDA preprocessor guard)
+-------------------+
| GPUManager        |-----depends--->  Core Layer
| IzhikevichKernel  |                  (NeuronArray,
| SpikePropKernel   |                   SynapseTable)
| STDPKernel        |
+-------------------+
```

### Layer Responsibilities

**Core Layer**: Neuron state, synapse graph, dynamics integration, plasticity, connectome I/O, parametric generation, checkpointing, and recording. No knowledge of the bridge protocol. All data structures are flat arrays with no virtual dispatch on the hot path.

**Bridge Layer**: Bidirectional neural twinning protocol. Read/write channel abstractions (including hardware adapters for Open Ephys, ScanImage, and Bonsai), shadow tracking, neuron replacement state machine, optogenetic safety model with three-state opsin kinetics (ChR2, ChRmine, stGtACR2) and tissue light scattering, spike decoder, supervised calibration, and a validation engine for comparing simulated vs recorded spike trains. Depends on core data structures but adds the three-phase twinning logic.

**Experiment Layer**: Top-level orchestration. CLI argument parsing, config file loading, and the experiment runner that wires connectome loading, bridge configuration, stimulus protocols, recording, and calibration into a single run.

**GPU Layer**: CUDA kernels and device memory management. Mirrors core data structures on the GPU. Activated by the `FWMC_CUDA` preprocessor define. All kernels are functionally equivalent to their CPU counterparts.


## Data Flow Through Simulation Loop

```
                     +-------------------+
                     |  Load Connectome  |
                     | (neurons.bin,     |
                     |  synapses.bin)    |
                     |  OR               |
                     | Generate from     |
                     |  .brain spec      |
                     +---------+---------+
                               |
                               v
                     +---------+---------+
                     |   Initialize      |
                     |  NeuronArray (SoA) |
                     |  SynapseTable(CSR) |
                     |  CellTypeManager   |
                     +---------+---------+
                               |
               +===============+===============+  <-- main loop
               |                               |
               v                               |
     +---------+---------+                     |
     | 1. ClearSynaptic  |  i_syn[*] = 0      |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 2. PropagateSpikes|  CSR traversal:     |
     |    (hot loop)     |  for each spiked    |
     |                   |  pre, deliver w*s   |
     |                   |  to post i_syn      |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 3. Apply Stimulus |  i_ext injection    |
     |    (timed events) |  from protocol      |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 4. Step Neurons   |  Izhikevich or LIF  |
     |    (parallel)     |  v,u update + spike  |
     |                   |  detection + reset   |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 5. STDP + Neuromod|  Weight updates,    |
     |    (if enabled)   |  DA release/decay   |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 6. Bridge Step    |  Shadow tracking,   |
     |    (if active)    |  replacement FSM,   |
     |                   |  optogenetic cmds   |
     +---------+---------+                     |
               |                               |
               v                               |
     +---------+---------+                     |
     | 7. Record + Ckpt  |  Spikes, voltages,  |
     |                   |  metrics to disk    |
     +---------+---------+                     |
               |                               |
               +===============================+
```


## File Format Specifications

### neurons.bin

```
Offset  Size     Field
0       4        count (uint32)
4       21*N     neuron records:
  +0      8        root_id (uint64):FlyWire root ID
  +8      4        x (float32):position in nm
  +12     4        y (float32)
  +16     4        z (float32)
  +20     1        type (uint8):CellType enum index
```

### synapses.bin

```
Offset  Size     Field
0       4        count (uint32)
4       13*N     synapse records:
  +0      4        pre (uint32):pre-synaptic neuron index
  +4      4        post (uint32):post-synaptic neuron index
  +8      4        weight (float32)
  +12     1        nt_type (uint8):NTType enum index
```

### checkpoint.bin

```
Offset  Size     Field
0       4        magic (uint32) = 0x4B435746 ("FWCK")
4       4        version (uint32) = 1
8       4        sim_time_ms (float32)
12      4        total_steps (int32)
16      4        total_resyncs (int32)
20      4        n_neurons (uint32)
24      4        n_synapses (uint32)
------- Neuron state (all float32 arrays of length n_neurons) -------
28      4*N      v[]
...     4*N      u[]
...     4*N      i_syn[]
...     1*N      spiked[] (uint8)
...     4*N      dopamine[]
...     4*N      serotonin[]
...     4*N      octopamine[]
...     4*N      last_spike_time[]
------- Synapse weights -------
...     4*S      weight[] (float32, length n_synapses)
------- Replacer state -------
...     1*N      state[] (uint8)
...     4*N      running_correlation[]
...     4*N      time_in_state[]
...     4*N      min_correlation[]
...     4*N      rollback_count[] (int32)
------- Shadow tracker -------
...     4        last_resync_time (float32)
...     4        n_history (uint32)
...     28*H     history[] (DriftSnapshot structs)
```

### spikes.bin (recording output)

```
Offset  Size     Field
0       4        n_neurons (uint32)
4       4        n_steps (uint32):patched on close
8       ...      per-step records:
  +0      4        time_ms (float32)
  +4      1*N      spiked[] (uint8)
```

### voltages.bin (recording output)

Same header as spikes.bin; per-step records contain `[time_ms:f32] [v[0..n]:f32]`.

### per_neuron_error.bin (recording output)

Same header as spikes.bin; per-step records contain `[time_ms:f32] [error[0..n]:f32]`.

### metrics.csv (recording output)

CSV with columns: `time_ms, spike_count, correlation, rmse, mean_v_error, false_pos, false_neg, resyncs, replaced_pct`.


## Build System and Platform Support

### Build System

FWMC uses CMake (minimum version 3.14). The project is header-only for the core and bridge layers, with a single translation unit (`fwmc.cc`) and three test executables.

```bash
# Linux / macOS
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Windows (MSVC)
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Platform Support

| Platform | Compiler | Status |
|----------|----------|--------|
| Linux x86_64 | GCC 13+, Clang 18+ | Supported |
| macOS arm64 | Apple Clang 15+ | Supported |
| Windows x64 | MSVC 19.35+ (VS 2022) | Supported |
| CUDA | nvcc 12.0+ | Optional (`-DFWMC_CUDA=ON`) |

The `platform.h` header provides portability macros:
- `FWMC_RESTRICT`: Maps to `__restrict` (MSVC) or `__restrict__` (GCC/Clang) for pointer aliasing hints
- Preprocessor guards for OpenMP availability
- CUDA host/device annotations when GPU support is active

### Dependencies

None required. The project uses only the C++ standard library (C++23), optional OpenMP, and optional CUDA.


## Performance Characteristics

### CPU Performance (single core, MSVC 19.44, Release)

| Operation | Scale | Throughput |
|-----------|-------|------------|
| IzhikevichStep | 100K neurons | ~89M neurons/sec |
| LIFStep | 100K neurons | ~170M neurons/sec (estimated) |
| Spike propagation (OpenMP) | 1K neurons, 5% spikes | 0.006 ms/iteration |
| Full bridge step | 10K neurons, 500K synapses | 0.73x real-time |

### GPU Performance (estimated, CUDA)

| Operation | Scale | Expected Speedup |
|-----------|-------|-----------------|
| IzhikevichStep | 100K neurons | 10-50x over single CPU core |
| Spike propagation | 140K neurons, 50M synapses | 5-20x (atomic contention limited) |
| STDP update | 50M synapses | 10-30x |

### Memory Requirements

| Component | 140K neurons | 50M synapses |
|-----------|-------------|-------------|
| NeuronArray (SoA) | ~8 MB | - |
| SynapseTable (CSR) | - | ~450 MB |
| Checkpoint file | ~460 MB | - |
| Spike recording (10K steps) | ~1.4 GB | - |
| Voltage recording (10K steps) | ~5.6 GB | - |

### Scaling

- Neuron stepping scales linearly with neuron count (embarrassingly parallel)
- Spike propagation scales with (active neurons) x (mean out-degree)
- STDP scales with synapse count
- Structural plasticity (sprouting) scales with (active neurons)^2


## Extension Points

### Adding a New Neuron Model

1. Define a parameter struct (e.g., `AdExParams`) in a new header under `src/core/`
2. Implement a step function with the same signature pattern as `IzhikevichStep`: takes `NeuronArray&`, `float dt_ms`, `float sim_time_ms`, and the params struct
3. Operate on the existing `NeuronArray` fields (`v`, `u`, `i_syn`, `i_ext`, `spiked`, `last_spike_time`)
4. Add OpenMP parallelization for neuron counts > 10K
5. Include NaN/Inf guards for numerical safety
6. Optionally add a CUDA kernel following the pattern in `src/cuda/izhikevich_kernel.cu`

### Adding a New Bridge Channel

1. Implement the `ReadChannel` interface (for biological-to-digital data flow):
   - `ReadFrame(float sim_time_ms) -> vector<BioReading>`
   - `NumMonitored() -> size_t`
   - `SampleRateHz() -> float`

2. Implement the `WriteChannel` interface (for digital-to-biological commands):
   - `WriteFrame(const vector<StimCommand>& commands)`
   - `MaxTargets() -> size_t`
   - `MinISI() -> float`

3. Wire the new channel into `TwinBridge` via `read_channel` / `write_channel` unique pointers

Existing implementations include `SimulatedRead`/`SimulatedWrite` (for testing) and `FileReadChannel` (for replay of pre-recorded data).

### Adding a New Scoring Function

Parameter sweep scoring functions follow the `ScoreFn` signature:

```cpp
using ScoreFn = std::function<float(const NeuronArray& neurons, float sim_time_ms)>;
```

Add new functions to the `fwmc::scoring` namespace in `param_sweep.h`. Higher return values indicate better fitness.

### Adding a New Cell Type

1. Add an entry to the `CellType` enum in `experiment_config.h`
2. Add Izhikevich parameters to the `ParamsForCellType()` switch in the same file
3. Add name mapping to `BrainSpecLoader::ParseCellTypeName()` in `brain_spec_loader.h`
4. Add display name to `ParamSweep::CellTypeName()` in `param_sweep.h`
