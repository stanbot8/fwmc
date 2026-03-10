#ifndef FWMC_STDP_H_
#define FWMC_STDP_H_

#include <algorithm>
#include <cmath>
#include "core/experiment_config.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"

namespace fwmc {

// Spike-timing-dependent plasticity.
// Updates synaptic weights based on relative timing of pre and post spikes.
// dw = A+ * exp(-dt/tau+) if pre fires before post (potentiation)
// dw = -A- * exp(dt/tau-) if post fires before pre (depression)
struct STDPParams {
  float a_plus = 0.01f;
  float a_minus = 0.012f;
  float tau_plus = 20.0f;   // ms
  float tau_minus = 20.0f;  // ms
  float w_min = 0.0f;
  float w_max = 10.0f;

  // Dopamine modulation (Izhikevich 2007, reward-modulated STDP)
  // When enabled, STDP weight changes are gated by local dopamine:
  //   dw_effective = dw * (1 + da_scale * dopamine[post])
  // High dopamine -> enhanced potentiation (reward learning in MB)
  // Low dopamine -> default STDP rates
  bool dopamine_gated = false;
  float da_scale = 5.0f;  // dopamine modulation strength

  // Eligibility traces (Izhikevich 2007, three-factor learning rule)
  // When enabled with dopamine_gated, spike pairs set an eligibility
  // trace that decays over tau_eligibility_ms. The actual weight change
  // happens later when dopamine arrives, allowing reward signals to
  // arrive seconds after the causal spike pair.
  // Without traces, dopamine must arrive within a single timestep.
  bool use_eligibility_traces = false;
  float tau_eligibility_ms = 1000.0f;  // eligibility trace decay (~1s)

  // Timing window multiplier: spikes beyond tau * window_factor are ignored
  float window_factor = 5.0f;
};

// Neuromodulator dynamics parameters (configurable)
struct NeuromodulatorParams {
  float da_release = 0.2f;        // dopamine released per DAN spike
  float da_decay = 0.005f;        // exponential decay rate per ms
  float da_autoreceptor = 0.5f;   // fraction of da_release for autoreceptor feedback
  float serotonin_decay = 0.002f; // serotonin decay rate per ms
  float oa_release = 0.1f;        // octopamine per spike (arousal signal)
  float oa_decay = 0.003f;        // octopamine decay rate per ms
};

// Update neuromodulator concentrations based on DAN spike activity.
// DANs release dopamine when they spike. Dopamine propagates through
// synaptic connections and decays exponentially.
//
// In the mushroom body, DANs encode reward/punishment signals that gate
// Kenyon cell -> MBON plasticity. This is the core mechanism for
// associative olfactory learning in Drosophila (Aso & Rubin 2016).
inline void NeuromodulatorUpdate(NeuronArray& neurons, const SynapseTable& synapses,
                                  float dt_ms,
                                  const NeuromodulatorParams& nmp = {}) {

  // Decay existing concentrations (clamp factor to [0,1] for large dt)
  float da_factor = std::max(0.0f, 1.0f - nmp.da_decay * dt_ms);
  float ser_factor = std::max(0.0f, 1.0f - nmp.serotonin_decay * dt_ms);
  float oa_factor = std::max(0.0f, 1.0f - nmp.oa_decay * dt_ms);
  for (size_t i = 0; i < neurons.n; ++i) {
    neurons.dopamine[i] *= da_factor;
    neurons.serotonin[i] *= ser_factor;
    neurons.octopamine[i] *= oa_factor;
  }

  // Release from spiking neuromodulatory neurons
  constexpr uint8_t kDAN_PPL1 = static_cast<uint8_t>(CellType::kDAN_PPL1);
  constexpr uint8_t kDAN_PAM  = static_cast<uint8_t>(CellType::kDAN_PAM);
  constexpr uint8_t kFS       = static_cast<uint8_t>(CellType::kFastSpiking);

  for (size_t i = 0; i < neurons.n; ++i) {
    if (!neurons.spiked[i]) continue;
    uint8_t ct = neurons.type[i];

    // DAN spike -> release dopamine to post-synaptic targets
    if (ct == kDAN_PPL1 || ct == kDAN_PAM) {
      uint32_t start = synapses.row_ptr[i];
      uint32_t end = synapses.row_ptr[i + 1];
      for (uint32_t s = start; s < end; ++s) {
        uint32_t post = synapses.post[s];
        neurons.dopamine[post] += nmp.da_release;
      }
      // Autoreceptor feedback
      neurons.dopamine[i] += nmp.da_release * nmp.da_autoreceptor;
    }

    // Octopamine from fast-spiking interneurons (arousal)
    if (ct == kFS) {
      neurons.octopamine[i] += nmp.oa_release;
    }
  }

  // Clamp to [0, 1]
  for (size_t i = 0; i < neurons.n; ++i) {
    neurons.dopamine[i] = std::min(1.0f, std::max(0.0f, neurons.dopamine[i]));
    neurons.serotonin[i] = std::min(1.0f, std::max(0.0f, neurons.serotonin[i]));
    neurons.octopamine[i] = std::min(1.0f, std::max(0.0f, neurons.octopamine[i]));
  }
}

// last_spike_time is updated in IzhikevichStep/LIFStep,
// so this function only reads neuron state.
// Each synapse is owned by exactly one pre-neuron in CSR layout,
// so the outer loop is embarrassingly parallel (no write conflicts).
//
// Two modes depending on STDPParams:
//   1. Direct mode (default): spike pairs immediately update weights,
//      optionally scaled by instantaneous dopamine.
//   2. Eligibility trace mode (use_eligibility_traces=true, dopamine_gated=true):
//      spike pairs write to eligibility traces; actual weight change is
//      deferred until dopamine arrives (three-factor learning rule).
inline void STDPUpdate(SynapseTable& synapses, const NeuronArray& neurons,
                       float sim_time_ms, const STDPParams& p) {
  const bool use_traces = p.use_eligibility_traces && p.dopamine_gated
                          && synapses.HasEligibilityTraces();
  const int n = static_cast<int>(synapses.n_neurons);
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
  #endif
  for (int pre = 0; pre < n; ++pre) {
    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];

    for (uint32_t s = start; s < end; ++s) {
      uint32_t post_idx = synapses.post[s];
      float dw = 0.0f;

      // Pre spiked this step
      if (neurons.spiked[pre]) {
        float dt = sim_time_ms - neurons.last_spike_time[post_idx];
        if (dt > 0 && dt < p.window_factor * p.tau_minus) {
          // Post fired before pre: depression
          dw += -p.a_minus * std::exp(-dt / p.tau_minus);
        }
      }

      // Post spiked this step
      if (neurons.spiked[post_idx]) {
        float dt = sim_time_ms - neurons.last_spike_time[pre];
        if (dt > 0 && dt < p.window_factor * p.tau_plus) {
          // Pre fired before post: potentiation
          dw += p.a_plus * std::exp(-dt / p.tau_plus);
        }
      }

      if (dw == 0.0f) continue;

      if (use_traces) {
        // Eligibility trace mode: accumulate trace, defer weight change
        synapses.eligibility_trace[s] += dw;
      } else {
        // Direct mode: apply weight change now, with optional DA scaling
        float da_mod = 1.0f;
        if (p.dopamine_gated) {
          da_mod = 1.0f + p.da_scale * neurons.dopamine[post_idx];
        }
        synapses.weight[s] = std::clamp(
            synapses.weight[s] + dw * da_mod, p.w_min, p.w_max);
      }
    }
  }
}

// Convert eligibility traces into weight changes using local dopamine.
// Call this after NeuromodulatorUpdate and STDPUpdate each timestep.
// The trace decays exponentially; dopamine arriving within the decay
// window converts accumulated spike-pair credit into actual learning.
//
// Three-factor rule (Izhikevich 2007):
//   dw/dt = eligibility_trace * dopamine * da_scale
//   d(trace)/dt = -trace / tau_eligibility
inline void EligibilityTraceUpdate(SynapseTable& synapses,
                                    const NeuronArray& neurons,
                                    float dt_ms, const STDPParams& p) {
  if (!synapses.HasEligibilityTraces()) return;

  float decay = std::exp(-dt_ms / p.tau_eligibility_ms);
  const int n = static_cast<int>(synapses.n_neurons);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
  #endif
  for (int pre = 0; pre < n; ++pre) {
    uint32_t start = synapses.row_ptr[pre];
    uint32_t end = synapses.row_ptr[pre + 1];

    for (uint32_t s = start; s < end; ++s) {
      float& trace = synapses.eligibility_trace[s];
      if (trace == 0.0f) continue;

      // Convert trace to weight change scaled by postsynaptic dopamine
      uint32_t post_idx = synapses.post[s];
      float da = neurons.dopamine[post_idx];
      if (da > 0.0f) {
        float dw = trace * da * p.da_scale * dt_ms;
        synapses.weight[s] = std::clamp(
            synapses.weight[s] + dw, p.w_min, p.w_max);
      }

      // Decay the trace
      trace *= decay;

      // Zero out negligible traces to avoid floating point dust
      if (std::abs(trace) < 1e-7f) trace = 0.0f;
    }
  }
}

// Multiplicative synaptic scaling (Turrigiano 2008).
// Homeostatic mechanism that prevents STDP from driving all weights
// to their bounds. Each neuron tracks its running firing rate and
// rescales its incoming synaptic weights to maintain a target rate.
//
// scaling_factor = (target_rate / actual_rate) ^ alpha
// w_new = w_old * scaling_factor
//
// alpha < 1 for stability (default 0.1 = gentle correction).
// Call periodically (every ~1000 steps), not every timestep.
struct SynapticScaling {
  float target_rate_hz = 5.0f;   // target firing rate
  float alpha = 0.1f;            // scaling exponent (smaller = gentler)
  float min_scale = 0.5f;        // prevent catastrophic down-scaling
  float max_scale = 2.0f;        // prevent runaway up-scaling

  // Per-neuron running firing rate estimate
  std::vector<float> firing_rate;
  std::vector<int> spike_count;
  float window_ms = 0.0f;  // accumulation window

  void Init(size_t n_neurons) {
    firing_rate.assign(n_neurons, 0.0f);
    spike_count.assign(n_neurons, 0);
    window_ms = 0.0f;
  }

  // Accumulate spike counts each timestep
  void AccumulateSpikes(const NeuronArray& neurons, float dt_ms) {
    for (size_t i = 0; i < neurons.n; ++i) {
      spike_count[i] += neurons.spiked[i];
    }
    window_ms += dt_ms;
  }

  // Apply scaling: compute rates, rescale incoming weights, reset counters.
  // Call every scaling_interval steps (e.g. every 100ms of sim time).
  // Uses the CSR transpose trick: iterate by pre-neuron but scale by
  // post-neuron's rate. This requires iterating all synapses.
  size_t Apply(SynapseTable& synapses, const STDPParams& p) {
    if (window_ms <= 0.0f) return 0;

    // Compute firing rates (Hz)
    float window_s = window_ms / 1000.0f;
    for (size_t i = 0; i < firing_rate.size(); ++i) {
      firing_rate[i] = static_cast<float>(spike_count[i]) / window_s;
    }

    // Compute per-neuron scaling factor
    std::vector<float> scale(firing_rate.size(), 1.0f);
    for (size_t i = 0; i < firing_rate.size(); ++i) {
      float rate = std::max(0.1f, firing_rate[i]);  // floor to avoid div by 0
      float ratio = target_rate_hz / rate;
      float s = std::pow(ratio, alpha);
      scale[i] = std::clamp(s, min_scale, max_scale);
    }

    // Scale incoming weights by postsynaptic neuron's scaling factor
    size_t n_scaled = 0;
    for (size_t pre = 0; pre < synapses.n_neurons; ++pre) {
      uint32_t start = synapses.row_ptr[pre];
      uint32_t end = synapses.row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        uint32_t post_idx = synapses.post[s];
        if (scale[post_idx] != 1.0f) {
          synapses.weight[s] = std::clamp(
              synapses.weight[s] * scale[post_idx], p.w_min, p.w_max);
          n_scaled++;
        }
      }
    }

    // Reset accumulators
    std::fill(spike_count.begin(), spike_count.end(), 0);
    window_ms = 0.0f;

    return n_scaled;
  }
};

}  // namespace fwmc

#endif  // FWMC_STDP_H_
