# Experiments Roadmap

Planned experiments for validating the brain simulator and bridging hardware.
Each experiment is self-contained: generates its own circuit, runs a protocol,
and produces quantitative readouts against published literature.

All experiment files live in `src/experiments/` with numbered prefixes.

## Status Key

- [x] Implemented and tested
- [~] Scaffolded (code exists, needs tuning/validation)
- [ ] Planned (design only)

---

## Tier 1: Drosophila Circuits (Primary Target)

### [x] 01 - Olfactory Conditioning (Mushroom Body)
**File**: `src/experiments/01_conditioning.h`
**Circuit**: ORN (100) -> PN (50) -> KC (500) -> MBON (20) + DAN (20)
**Protocol**: Pre-test, 5 CS+/US training trials, 5 CS- trials, post-test
**Readouts**: Learning index, discrimination index, KC->MBON weight changes
**Literature**: Aso & Rubin 2016, Handler et al. 2019, Cognigni et al. 2018

### [x] 02 - Visual Looming Escape
**File**: `src/experiments/02_visual_escape.h` | **CLI**: `--visual-escape`, `--escape-optimize`
**Circuit**: Photo (100) -> LC (200) -> LPLC2 (20) -> GF (2) -> MN (20) + INH (30)
**Protocol**: Expanding disc (looming) stimulus, corrected collision time from start_angle_deg
**Readouts**: Escape latency=88.5ms (lit: 30-150ms), angle=25.5deg (lit: 20-90deg)
**Literature**: Fotowat & Bhatt 2015, von Reyn et al. 2017, Ache et al. 2019

### [x] 03 - Central Complex Navigation
**File**: `src/experiments/03_navigation.h` | **CLI**: `--navigation`, `--nav-optimize`
**Circuit**: Ring (16) -> EPG (16) <-> PEN (16) + Delta7 (16) -> PFL (32) -> FC (20)
**Protocol**: Gradual bar rotation + darkness persistence test, auto-tuned via ExperimentOptimizer
**Readouts**: Heading error=2.5deg (target <30), dark_err=8.7deg (target <60), R2=0.998, bump=2.6
**Literature**: Kim et al. 2017, Green et al. 2017, Seelig & Jayaraman 2015

### [x] 06 - Courtship Song Generation
**File**: `src/experiments/06_courtship.h` | **CLI**: `--courtship`
**Circuit**: P1 (10) -> pIP10 (20) -> dPR1 (10) + vPR6/vPR9 (10/10) -> MN_pulse (15) + MN_sine (15)
**Protocol**: P1 activation, measure pulse song vs. sine song generation
**Readouts**: Inter-pulse interval (44.5ms, lit: 30-45ms), carrier frequency, song bout duration
**Literature**: von Philipsborn et al. 2011, Ding et al. 2019

### [ ] 07 - Sleep-Wake Cycling
**Circuit**: dFB (50, R5 neurons) -> helicon cells -> motor output
**Protocol**: Extended simulation (hours of sim time), measure spontaneous cycling
**Readouts**: Sleep bout duration, homeostatic rebound, arousal threshold
**Literature**: Donlea et al. 2018, Liu et al. 2016

### [ ] 08 - Grooming Hierarchy
**Circuit**: antennal/eye/wing/body sensory -> DNg (4 groups) -> motor
**Protocol**: Multi-site dust stimulus, measure grooming sequence priority
**Readouts**: Grooming bout ordering, winner-take-all suppression index
**Literature**: Seeds et al. 2014, Hampel et al. 2015

---

## Tier 2: Multi-Species (Scaling Validation)

### [~] 04 - Mouse Barrel Cortex Whisker Response
**File**: `src/experiments/04_whisker.h` | **CLI**: `--whisker`
**Circuit**: VPM thalamus -> L4 barrels -> L2/3 -> L5 output
**Spec**: `examples/mouse_cortical_column.brain` (2380 neurons)
**Protocol**: Brief whisker deflection, measure cortical response latency
**Readouts**: First-spike latency, adaptation ratio, surround suppression
**Literature**: Petersen 2007, Lefort et al. 2009, Constantinople & Bruno 2013

### [~] 05 - Zebrafish Prey Capture
**File**: `src/experiments/05_prey_capture.h` | **CLI**: `--prey-capture`
**Circuit**: Retina -> tectum PVN/SIN -> pretectum -> hindbrain motor
**Spec**: `examples/zebrafish_optic_tectum.brain` (5000 neurons)
**Protocol**: Moving dot stimulus (prey-like), measure tectal response and J-turn
**Readouts**: Direction selectivity, capture success latency, eye convergence angle
**Literature**: Bianco et al. 2011, Mearns et al. 2020, Del Bene et al. 2010

### [ ] 09 - Human Cortical Column (Validation Only)
**Spec**: `examples/human_cortical_column.brain` (10550 neurons)
**Protocol**: Thalamic pulse, measure cortical response dynamics
**Readouts**: Layer-specific latency, gamma oscillation frequency, E/I balance
**Literature**: Markram et al. 2015 (Blue Brain), Eyal et al. 2018

---

## Tier 3: Twinning Experiments (Bridging)

### [x] Bridge Self-Test (Software Validation)
**File**: `src/bridge_self_test.h`
**Protocol**: Generate bio+digital pair, run full shadow->closed-loop pipeline
**Readouts**: Correlation, RMSE, replacement fraction, resync count

### [x] Optogenetics Experiment
**File**: `optogenetics/optogenetics.h`
**Protocol**: Photostimulation of target region with safety monitoring
**Readouts**: Modulation index, off-target rate, desensitization, energy per neuron

### [x] Optogenetics Optimizer
**File**: `optogenetics/optimizer.h`
**Protocol**: CMA-ES parameter search over laser power, fraction, timing
**Readouts**: Best objective score, optimal parameters, convergence curve

### [x] 07 - Full Twinning Demo (Phase 1-3)
**File**: `src/experiments/07_twinning.h` | **CLI**: `--twinning`
**Circuit**: Mushroom body (690 neurons) trained via CS+/US conditioning
**Protocol**:
1. Train circuit (5 CS+/US trials, STDP), validate learning (LI=4.9)
2. Shadow mode (500ms, measure drift)
3. Closed-loop replacement (1000ms, progressive neuron advancement)
4. Post-replacement behavioral test (CS+ response preservation)
5. Ablation recovery (10% bio neurons killed, digital maintains behavior)
**Readouts**: Behavioral continuity=1.000, CS+ 2343->2343 MBON spikes preserved
**Key insight**: Spike-level correlation is ~0 in chaotic spiking networks; behavioral
(MBON response magnitude) is the meaningful continuity metric.

### [x] 08 - Ablation Study (KC Degradation Curve)
**File**: `src/experiments/08_ablation.h` | **CLI**: `--ablation`
**Circuit**: Mushroom body (690 neurons, 17K synapses) trained via dopamine-gated STDP
**Protocol**:
1. Train MB circuit (5 CS+/US trials with three-factor STDP)
2. Measure post-training CS+ baseline (LI=2.11)
3. Progressive KC ablation: silence 0%, 5%, 10%, ..., 90% of Kenyon cells
4. Re-measure MBON CS+ response at each ablation fraction
**Readouts**: Graceful degradation score=0.86, half_life=90% (continuity drops below 50%
only at 90% ablation). Circuit retains 77% response at 70% ablation.
**Key result**: Confirms sparse KC coding robustness (Hige et al. 2015) -- the mushroom
body tolerates massive neuron loss before behavioral degradation, directly calibrating
the replacement rate budget for neural prosthesis.
**Literature**: Hige et al. 2015, Caron et al. 2013, Aso & Rubin 2016

### [x] 09 - Compensated Ablation (Neural Prosthesis Demo)
**File**: `src/experiments/09_compensated_ablation.h` | **CLI**: `--compensated`
**Circuit**: Mushroom body (690 neurons, 17K synapses) trained via dopamine-gated STDP
**Protocol**:
1. Train MB circuit, measure post-training CS+ baseline
2. For each ablation fraction (0-90%), run TWO conditions:
   - Pure ablation: silence KC neurons, measure MBON response
   - Compensated: silence same neurons, digital twin fills in with trained weights
3. Compare degradation curves
**Readouts**:
- Pure ablation: graceful=0.85, half_life=90%
- Compensated: graceful=1.02, half_life=never reached (continuity stays >0.5 at 90%)
- Benefit increases with damage: +0.12 at 10%, +0.29 at 90% ablation
- Lifetime extension: effectively infinite (compensated circuit never reaches 50% degradation)
**Key result**: The digital twin extends functional circuit lifetime beyond what biology
alone can sustain. Compensation benefit grows with damage severity -- the sicker the brain,
the more the prosthesis helps. This is the core demonstration for neural prosthetics.
**Literature**: Hige et al. 2015, Aso & Rubin 2016, Gradmann 2023

---

## Experiment Framework Conventions

Every experiment follows this pattern (see `experiments/01_conditioning.h` as template):

```cpp
struct ExperimentResult {
  // Quantitative readouts (floats, spike counts, indices)
  // Timing information
  // Pass/fail criteria based on literature ranges
  bool passed() const;
};

struct ExperimentClass {
  // Circuit parameters (neuron counts, connectivity densities)
  // Timing parameters (dt, trial durations, ITIs)
  // Stimulus parameters

  ExperimentResult Run(uint32_t seed = 42) {
    // 1. Build circuit (ParametricGen or custom)
    // 2. Assign cell types (CellTypeManager)
    // 3. Run protocol phases (pre-test, stimulus, training, post-test)
    // 4. Collect readouts (RateMonitor, spike counts, weights)
    // 5. Return structured result
  }
};
```

CLI integration: each experiment gets a `--experiment-name` flag in fwmc.cc.

---

## Priority Order

1. Visual looming escape (done, latency=88.5ms)
2. CX navigation (done, R2=0.998, auto-tuned)
3. Mouse whisker (done, needs tuning)
4. Zebrafish prey capture (done, needs tuning)
5. Courtship song generation (done, IPI=44.5ms)
6. Full twinning demo (done, continuity=1.000)
7. Ablation study (done, graceful=0.86, half_life=90%)
8. Compensated ablation (done, benefit=0.17, lifetime_ext=infinite)
9. Sleep, grooming (next)
