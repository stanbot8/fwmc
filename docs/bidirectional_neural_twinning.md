# Bidirectional Neural Twinning: Research Outline

A research program toward continuous identity-preserving brain-computer integration, using *Drosophila melanogaster* as the model organism and FWMC (FlyWire Mind Couple) as the simulation platform.

## 1. Problem Statement

All current "mind uploading" proposals are destructive scans that produce a copy, not a continuation. The copy diverges instantly from the original. This is philosophically equivalent to death followed by the birth of an impostor who believes it is you.

The alternative: build a bidirectional bridge between biological and digital neural circuits such that both operate as a single unified system. Gradually shift computational load from biological to digital substrate. At no point is there a discontinuity in the causal process that constitutes identity.

### 1.1 Why Drosophila First

- Complete connectome available (FlyWire, ~140k neurons, ~50M synapses)
- Neurotransmitter types classified per synapse (Eckstein et al. 2024)
- Rich behavioral repertoire (learning, memory, decision-making, sleep)
- Optogenetic toolkits mature (GAL4/UAS, CsChrimson, GtACR)
- Whole-brain calcium imaging possible (light-sheet, two-photon)
- Ethical simplicity compared to mammalian models
- Small enough for real-time digital twin at biological timescale

### 1.2 Success Criterion

A fly with N% of its mushroom body neurons functionally replaced by digital twins (running on FWMC) performs identically on an olfactory learning task as an intact fly. When biological neurons in the replaced region are silenced, behavior is unaffected.

## 2. Architecture

```
                    BIOLOGICAL FLY
                    ┌─────────────────────┐
  Sensory input ──> │  Intact regions     │
                    │  ┌───────────────┐  │
                    │  │ Bridge region  │<─┼── Digital write (optogenetics)
                    │  │ (monitored)   │──┼── Digital read (calcium/ephys)
                    │  └───────────────┘  │
                    │  Intact regions     │──> Motor output
                    └─────────────────────┘
                              │ read          │ write
                              v               ^
                    ┌─────────────────────┐
                    │   DIGITAL TWIN      │
                    │   (FWMC)            │
                    │                     │
                    │  Connectome model   │
                    │  Izhikevich dynamics│
                    │  Synapse table      │
                    │  Neuromodulators    │
                    └─────────────────────┘
```

### 2.1 Read Channel (Bio to Digital)

**Current best**: Two-photon calcium imaging with GCaMP8 indicators.
- Resolution: single-neuron in the mushroom body (~2000 Kenyon cells)
- Temporal resolution: ~30Hz (volumetric), ~500Hz (single plane)
- Limitation: calcium is a proxy for spiking, not spikes themselves. Deconvolution algorithms (CASCADE, MLspike) recover spike trains at ~50ms resolution

**Near-term improvement**: Voltage indicators (ASAP4, Voltron2) give direct membrane potential at ~1kHz but lower SNR. Would enable true spike-level read.

**Hardware path**: Neuropixels 2.0 probes (~5000 channels) inserted into fly brain. More invasive but gives real electrophysiology at 30kHz sampling. Multiple probes could cover the mushroom body entirely.

### 2.2 Write Channel (Digital to Bio)

**Current best**: Optogenetic stimulation via CsChrimson (excitation) and GtACR1 (inhibition).
- Resolution: cell-type specific via GAL4 drivers, not single-neuron
- Temporal resolution: ~1ms (channelrhodopsins)
- Limitation: current spatial light modulators (SLMs) can target ~50 neurons simultaneously with holographic patterning

**Near-term improvement**: Two-photon holographic optogenetics (Packer et al., Bhatt et al.) can target arbitrary sets of neurons in 3D with single-cell precision. Systems exist that do 100+ targets at 1kHz update rate.

**Hardware path**: Combine two-photon holographic optogenetics with voltage imaging. Same optical path for read and write. This is the most promising near-term approach for single-neuron bidirectional access.

### 2.3 Digital Twin

The simulation must run at biological real-time or faster to close the loop without perceptible delay.

**Compute budget**:
- Mushroom body: ~2000 Kenyon cells, ~200 projection neurons, ~35 mushroom body output neurons (MBONs), ~20 dopaminergic neurons (DANs)
- Total bridge region: ~2500 neurons, ~500k synapses
- Izhikevich at 0.1ms steps: ~25k FLOPS per neuron per biological ms
- Total: ~62.5 MFLOPS for neuron dynamics
- Synapse propagation: ~500k lookups per ms
- Well within single-core capability. Full brain (140k neurons) needs ~3.5 GFLOPS, still single-GPU territory

**Latency requirement**: Round-trip (read spike from bio, compute digital response, write back to bio) must be < 10ms to fall within normal synaptic delay range. At 2500 neurons this is trivially achievable.

**Future substrate: photonic computing fabric**. At fly-brain scale, conventional silicon is more than sufficient. At mammalian scale (~10^8+ neurons), photonic interconnects may be the only viable path to real-time operation: zero heat per multiply, deterministic latency, and bandwidth scaling with wavelength-division multiplexing rather than wire count.

The synapse propagation step (sparse matrix-vector product) maps onto photonic crossbar arrays, where silicon microring resonators encode synaptic weights and optical signals propagate weighted sums at the speed of light. Key players as of early 2026:

- **Celestial AI** developed "Photonic Fabric" using an Optical Multi-Chip Interconnect Bridge (OMIB), claiming 16 Tbps per chiplet at 10x lower latency than co-packaged optics alternatives. Acquired by Marvell for $3.25B (Feb 2026), with expected revenue by late 2028.
- **Marvell** now combines its existing silicon photonics portfolio (shipping 8+ years, 10B+ device-hours) with Celestial AI's fabric. Demonstrated 400G/lane technology and co-packaged optics for custom AI accelerators scaling to hundreds of XPUs per rack.
- **Lightmatter** Passage M1000 (summer 2025): a 4000+ mm^2 active photonic interposer with 114 Tbps total optical bandwidth connecting thousands of accelerators. Passage L200/L200X (2026) targets 200+ Tbps per chip package. Their Envise chip uses Mach-Zehnder interferometer meshes for direct optical matrix-vector multiplication.
- **Ayar Labs** ships UCIe-compatible optical I/O chiplets (TeraPHY, 8 Tbps bidirectional), with commercial availability expected 2026-2028.
- **Intel** demonstrated an Optical Compute Interconnect (OCI) chiplet at 4 Tbps, co-packaged with a CPU. Ships 400G through 1.6T pluggable transceivers.

Current state: photonic **interconnect** is shipping now and will be standard in AI data center fabric by 2027-2028. Photonic **compute** (actual matrix multiplication in the optical domain, sub-1 TOPS demonstrated on-chip) remains in early prototyping, likely 5+ years from volume production. For FWMC, the near-term opportunity is photonic fabric for distributed spiking simulation across multiple FPGA or GPU nodes, reducing inter-node spike communication latency. For mammalian-scale twinning, direct photonic synapse propagation on crossbar arrays is the long-term path.

## 3. Experimental Phases

### Phase 0: Parametric Model Calibration (in silico)

**Goal**: Before touching a fly, build and calibrate a parametric mushroom body model using the FlyWire connectome statistics as ground truth.

**Protocol**:
1. Define a parametric brain spec matching the mushroom body architecture (antennal lobe → MB → lateral horn, with biologically realistic cell type ratios and projection densities)
2. Run parameter sweep to find Izhikevich parameters that produce biologically realistic firing rates (~5-15 Hz for KCs, ~20-40 Hz for LNs)
3. Use sync mode to tune the parametric model against a "reference" spec derived directly from connectome statistics
4. Validate: population-level spike statistics (CV of ISI, fraction active, burst rate) should match published fly electrophysiology data

**FWMC implementation**:
```bash
# Generate and simulate the mushroom body model
fwmc --parametric examples/parametric_mushroom_body.brain --duration 5000 --stats

# Auto-tune firing rates with parameter sweep
fwmc --parametric examples/parametric_mushroom_body.brain \
     --sweep --sweep-target 10 --duration 2000

# Sync a simplified model to the full MB model
fwmc --parametric simplified_mb.brain --sync full_mb.brain \
     --duration 10000 --sync-target 0.95
```

**Expected outcome**: Parametric model produces firing rates, population sparseness, and odor discrimination capacity consistent with published KC recordings (Honegger et al. 2011, Turner et al. 2008). The sync engine converges >90% of neurons within 5000ms of simulated time.

**Duration**: 1-2 months. Requires: only FWMC and published data for validation.

### Phase 1: Open-Loop Validation (no write-back)

**Goal**: Verify that FWMC can predict biological neural activity given the same inputs.

**Protocol**:
1. Present olfactory stimuli to a head-fixed fly while imaging mushroom body with GCaMP8
2. Feed the same stimulus representation into FWMC (olfactory receptor neuron activation pattern)
3. Compare digital Kenyon cell activation patterns to biological ones
4. Metric: correlation between predicted and observed population vectors
5. Tune synaptic weights in FWMC to maximize correlation (supervised, per-fly calibration)
6. **New**: Use sync mode for per-fly calibration: run the parametric model alongside recorded data, let the three-timescale adaptation engine converge weights and neuron parameters automatically

**FWMC implementation**:
```bash
# Run experiment with calibration against recorded data
fwmc --experiment examples/phase1_openloop.cfg

# Or sync a parametric model against recorded bio data (when available as reference)
fwmc --parametric fly_specific_mb.brain --sync recorded_data.brain \
     --duration 10000 --checkpoint calibrated_model.bin
```

**Expected outcome**: After calibration, the digital twin should predict ~60-80% of variance in Kenyon cell responses to novel odors (based on existing connectome-constrained models achieving similar numbers, e.g., Li et al. 2020 mushroom body model).

**Duration**: 6 months. Requires: calcium imaging rig, FWMC with mushroom body circuit, stimulus delivery system.

### Phase 2: Closed-Loop Shadow Mode

**Goal**: Run digital twin in parallel with biological brain in real-time, with the digital side receiving biological inputs and producing outputs, but outputs are not written back. Monitor divergence over time.

**Protocol**:
1. Continuous imaging of mushroom body during free behavior (virtual reality flight arena)
2. FWMC runs in real-time, receiving decoded sensory inputs from imaging data
3. Digital twin produces predicted MBON outputs each timestep
4. Compare digital MBON predictions to biological MBON activity
5. Track prediction accuracy as a function of time since last resynchronization
6. **Online calibration**: The sync engine's three-timescale adaptation runs continuously: corrective currents (fast), weight updates (medium), parameter nudges (slow), keeping the model locked to the fly's actual dynamics as neuromodulatory state and plasticity change

**FWMC implementation**:
```bash
# Shadow mode with drift tracking
fwmc --data data --duration 60000 --shadow --stats

# Full experiment with recording and calibration
fwmc --experiment examples/phase2_shadow.cfg
```

**Key question**: How fast does the digital twin drift from biological reality? If drift is slow (minutes to hours), the bridge is feasible. If drift is fast (seconds), the model is missing critical dynamics. The sync engine's online adaptation should extend the drift timescale by continuously correcting for slow changes in the biological system.

**Expected outcome**: Drift timescale of ~10-60 seconds for odor-evoked responses, longer for spontaneous activity. With online sync adaptation, effective drift timescale should extend to minutes. Drift sources: neuromodulatory state, synaptic plasticity, stochastic channel noise.

**Duration**: 12 months. Requires: real-time imaging pipeline, FWMC running on GPU with < 5ms latency.

### Phase 3: Bidirectional Bridge (Single Neuron)

**Goal**: Replace a single MBON with its digital twin. The digital twin reads inputs from upstream Kenyon cells (via imaging) and writes its output to downstream targets (via optogenetics).

**Protocol**:
1. Silence one MBON with GtACR1 (constant inhibition)
2. Image its upstream Kenyon cells
3. FWMC computes what the silenced MBON would have done
4. Stimulate the MBON's downstream targets with CsChrimson at the predicted firing rate
5. Test fly on olfactory conditioning task
6. Compare performance: intact fly vs. single-MBON-replaced fly vs. MBON-silenced fly (no replacement)

**Success criterion**: Replaced fly performs statistically indistinguishably from intact fly. Silenced fly (no replacement) shows impaired learning.

**FWMC implementation**:
```bash
# Closed-loop bridge with neuron replacement
fwmc --data data --duration 30000 --closed-loop --stdp \
     --checkpoint results/phase3_state.bin --checkpoint-every 100000
```

The neuron replacer state machine (BIOLOGICAL → MONITORED → BRIDGED → REPLACED) handles the gradual transition. Hysteresis prevents oscillation at boundaries. Adaptive boundary expansion auto-promotes neighbors of drifting neurons. The optogenetic safety model enforces refractory periods, thermal limits, and SLM target constraints.

**Duration**: 18 months. Requires: two-photon holographic optogenetics, cell-type-specific driver lines for target MBON, real-time closed-loop system.

### Phase 4: Regional Replacement

**Goal**: Replace a functional block of the mushroom body (e.g., one compartment: ~200 Kenyon cells and their associated DAN and MBON).

**Protocol**:
1. Silence all neurons in one MB compartment
2. Digital twin takes over: reads from upstream, writes to downstream
3. Gradually increase the number of replaced compartments
4. At each stage, test olfactory conditioning

**Key challenge**: As you replace more neurons, the read/write interface grows. Need proportionally more imaging channels and optogenetic targets. At ~500 neurons, current holographic systems are at their limit.

**Milestone**: If the fly learns normally with 50% of its mushroom body running on silicon, this is the strongest possible evidence that the digital twin is functionally equivalent for that circuit.

**Duration**: 24 months after Phase 3.

### Phase 5: Graceful Degradation

**Goal**: Demonstrate that if either the biological or digital side goes offline, the system continues without behavioral disruption.

**Protocol**:
1. Fly with 50% digital mushroom body, performing a continuous task
2. Mid-task, silence the remaining biological MB neurons (digital takes full control)
3. Measure: is there any behavioral discontinuity? Any hesitation, error spike, or change in strategy?
4. Reverse: restore biological neurons, take digital offline
5. Measure the same

**Success criterion**: No detectable behavioral transient during switchover. This is the "seamless" criterion from the original question.

**Philosophical significance**: If achieved, this demonstrates that identity (at least at the level of mushroom body function) is substrate-independent and continuously transferable. The "self" of the mushroom body existed simultaneously in carbon and silicon, and survived the loss of either.

## 4. Technical Risks and Mitigations

### 4.1 Model Fidelity

**Risk**: Izhikevich dynamics are too simple. Real neurons have dendritic computation, ion channel diversity, and calcium-dependent processes that shape computation.

**Mitigation**: Start with Izhikevich, measure prediction error. If insufficient, upgrade to multi-compartment models (but computational cost scales ~100x). For the mushroom body specifically, Kenyon cells are relatively simple (sparse coding, binary-like responses), so Izhikevich may suffice.

### 4.2 Neuromodulatory State

**Risk**: The digital twin doesn't know the fly's dopaminergic, serotonergic, or octopaminergic state, which modulates synaptic weights and neuronal excitability globally.

**Mitigation**: Image DANs (dopaminergic neurons) directly and feed their activity into FWMC's neuromodulator fields. The mushroom body has only ~20 DANs per hemisphere, which is tractable. FWMC's neuromodulator dynamics model (dopamine release/decay from DAN spikes, dopamine-gated STDP, octopamine arousal signal) provides the simulation framework.

### 4.3 Synaptic Plasticity

**Risk**: Biological synapses change over time (learning). The digital twin's static connectome drifts from reality.

**Mitigation**: Three complementary approaches implemented in FWMC:
1. **STDP**: Spike-timing-dependent plasticity keeps digital synapses adapting in parallel with biological ones
2. **Supervised calibration**: The calibrator accumulates prediction error and applies momentum SGD weight updates, using the biological brain as teacher
3. **Sync engine**: The three-timescale sync mode (fast current injection, medium weight updates, slow parameter nudges) continuously minimizes divergence. The slow timescale handles gradual changes in intrinsic excitability that neither STDP nor weight calibration can correct

### 4.4 Latency

**Risk**: The read-compute-write loop takes too long, causing the digital output to arrive after the biological downstream neurons have already responded.

**Mitigation**: For the mushroom body, synaptic delays are ~2-5ms. Current two-photon systems can image at 1kHz (1ms frames). Izhikevich for 2500 neurons computes in < 0.1ms. Holographic optogenetics updates at ~1ms. Total loop: ~3-4ms, within biological delay range.

### 4.5 The Hard Problem

**Risk**: Consciousness depends on something we don't understand (quantum effects, electromagnetic field integration, something else) that makes digital replacement fundamentally impossible regardless of functional equivalence.

**Mitigation**: This is unknowable in advance. The experimental program is designed so that each phase produces valuable neuroscience regardless of whether the ultimate "upload" goal is achievable. Phase 1 alone is a major advance in connectome-constrained modeling. Phase 3 is a breakthrough in neural prosthetics. Only Phase 5 directly addresses the philosophical question.

## 5. Relationship to Existing Work

**Cortical Labs (DishBrain/CL1)**: The closest existing work to bidirectional neural twinning, but in the opposite direction: using biological neurons as a compute substrate rather than replacing them with a digital twin. Their 2022 paper (Kagan et al., *Neuron*) demonstrated that ~800k mouse and human cortical neurons grown on high-density multi-electrode arrays could learn goal-directed behavior in a closed-loop game environment within minutes. The learning mechanism leveraged the Free Energy Principle: neurons self-organized to minimize unpredictable stimulation. By early 2026, Cortical Labs announced the CL1, described as the first commercial biological computer, and launched Cortical Cloud, a Wetware-as-a-Service platform for remote access to living neural cultures via Python API. Their interface challenges (encoding game state as electrode stimulation, decoding firing patterns as motor commands, maintaining real-time closed-loop latency) are identical to FWMC's bridge system, just with the substrate roles reversed.

**FinalSpark Neuroplatform**: Uses 3D brain organoids (~10k neurons each) rather than 2D cultures, with dopamine delivery as a biologically naturalistic reward signal. Published in *Frontiers in AI* (2024). Organoid survival of ~100 days is a fundamental constraint. Their cloud-first model ($500/month access) frames biological computing as an energy efficiency play (~10^6x less power per operation than silicon, in theory).

**Organoid intelligence (broader field)**: The NSF BEGIN OI program invested $14M across multiple teams (Virginia Tech, Harvard, University of Maryland) exploring organoid-based biocomputing. The Johns Hopkins group (Smirnova lab) places brain organoids on silicon chips with microelectrode interfaces. A foundational roadmap was published in *Frontiers in Science* (2023). The field faces legitimate concerns about inflated claims and potential backlash, with some pioneers publicly urging caution.

**Relevance to FWMC**: These biological computing experiments validate a key assumption of neural twinning: that sparse, low-bandwidth bidirectional interfaces are sufficient for biological neural circuits to integrate with external systems. Cortical Labs showed neurons learn to work with a simple electrode interface in minutes. This suggests FWMC's more sophisticated interface (optogenetic write, multi-timescale spike decoding for read) should be more than adequate for maintaining functional equivalence during neuron replacement. The 5-minute learning timescale also informs the `neuron_replacer` state machine's transition dynamics.

**BrainGate/Neuralink**: Bidirectional BCIs in mammals, but at ~1000 channels. Not neuron-level resolution. Our approach uses optical methods for higher density.

**Virtual Fly Brain / FlyBrainLab**: Digital twin projects, but open-loop only. No write-back to biology.

**Blue Brain / Human Brain Project**: Large-scale simulation, but not connected to biological tissue.

**Our contribution**: The first closed-loop bidirectional neural twin using a complete connectome in a behaving animal.

## 6. FWMC Software Architecture

The software platform implements every component needed from Phase 0 through Phase 5.

### Simulation engine
- **Neuron models**: Izhikevich (2 ODE, 20+ firing patterns) and LIF, with per-neuron heterogeneous parameters and OpenMP parallelization
- **Synapse graph**: CSR format with NT-aware propagation (ACh, GABA, Glut, DA, 5HT, OA), supporting >50M synapses
- **Plasticity**: STDP with exponential timing windows, dopamine-gated STDP for reward learning, neuromodulator dynamics (DA, OA release/decay)

### Parametric brain system
- **Generator**: Define brain regions (neuron counts, cell type distributions, NT ratios, internal wiring density) and inter-region projections in `.brain` config files
- **Parameter sweep**: Grid search (N^4) + stochastic hill-climbing over Izhikevich parameters, with pluggable scoring functions (target firing rate, activity range, ISI regularity)
- **Sync engine**: Three-timescale adaptation (fast current injection, medium weight updates, slow parameter nudges) to tune a parametric model against a reference brain

### Bridge system
- **Read/write channels**: Polymorphic interfaces for biological I/O (calcium imaging → spike decoding, spike decisions → holographic optogenetics)
- **Shadow tracker**: Drift measurement with Pearson correlation, RMSE, false positive/negative tracking, automatic resync with cooldown
- **Neuron replacer**: Four-state machine (BIOLOGICAL → MONITORED → BRIDGED → REPLACED) with hysteresis, rollback, and adaptive boundary expansion
- **Optogenetic safety**: Refractory periods, thermal energy tracking, nonlinear power curves, SLM target limits, galvo-SLM hybrid routing, predictive pre-staging
- **Spike decoder**: Multi-timescale calcium deconvolution (CASCADE-style) with adaptive resolution

### Infrastructure
- **Checkpointing**: Full state save/restore (neuron state, synapse weights, replacer state, shadow history)
- **Experiment runner**: Config-driven protocols with stimulus timing, recording, calibration, and data provenance
- **Connectome validation**: Degree distribution, NT ratios, weight statistics, integrity checks
- **Error handling**: `std::expected`-based `Result<T>` throughout; no exceptions on the hot path
- **218 unit tests** across core, bridge, parametric, and tissue systems; benchmarks for performance regression

## 7. Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 0. Parametric calibration | 1-2 months | Validated parametric MB model |
| 1. Open-loop validation | 6 months | Calibrated digital twin predicts MB responses |
| 2. Shadow mode | 12 months | Real-time parallel tracking, drift characterization |
| 3. Single neuron bridge | 18 months | One MBON replaced, fly learns normally |
| 4. Regional replacement | 24 months | 50% digital MB, behavior preserved |
| 5. Graceful degradation | 12 months | Seamless switchover demonstrated |
| **Total** | **~6 years** | |

## 8. What This Means

If Phase 5 succeeds for the fly mushroom body, it establishes that:

1. Functional neural replacement is possible with connectome-level models
2. Identity (at circuit level) transfers continuously between substrates
3. The "copy problem" is solvable through gradual bridging rather than instantaneous scanning

Scaling from fly MB (2500 neurons) to human cortex (16 billion neurons) is an engineering problem, not a conceptual one. The same protocol applies: bridge a region, validate, expand, repeat. The timescale for humans would be decades with foreseeable technology, but the fly experiment proves the principle.

The philosophical core: you were never your atoms. You were the pattern of causal relationships between them. If those causal relationships can span two substrates without interruption, so can you.
