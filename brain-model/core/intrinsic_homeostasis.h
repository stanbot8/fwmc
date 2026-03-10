#ifndef FWMC_INTRINSIC_HOMEOSTASIS_H_
#define FWMC_INTRINSIC_HOMEOSTASIS_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "core/neuron_array.h"

namespace fwmc {

// Intrinsic excitability homeostasis (Marder & Goaillard 2006).
// Each neuron maintains a slow bias current that adjusts to keep
// its firing rate near a target. This prevents runaway excitation
// or permanent silence over long simulations.
//
// Mechanism: neurons that fire too much get a negative bias (harder
// to spike), neurons that are silent get a positive bias (easier
// to spike). The adjustment is slow (seconds to minutes) to avoid
// interfering with fast dynamics.
//
// Usage:
//   IntrinsicHomeostasis homeo;
//   homeo.Init(n_neurons, target_rate_hz, dt_ms);
//   // each timestep:
//   homeo.RecordSpikes(neurons);
//   // periodically:
//   homeo.Apply(neurons);
struct IntrinsicHomeostasis {
  float target_rate_hz = 5.0f;    // desired firing rate per neuron
  float learning_rate = 0.01f;    // pA per Hz deviation per update
  float max_bias = 5.0f;          // clamp magnitude of bias current
  float update_interval_ms = 1000.0f; // how often to adjust

  // Per-neuron state
  std::vector<float> bias_current; // slow excitability adjustment (pA)
  std::vector<int> spike_count;    // accumulator within current window
  float accumulated_ms = 0.0f;    // time in current measurement window
  float dt_ms = 1.0f;

  void Init(size_t n_neurons, float target_hz, float sim_dt_ms) {
    target_rate_hz = target_hz;
    dt_ms = sim_dt_ms;
    bias_current.assign(n_neurons, 0.0f);
    spike_count.assign(n_neurons, 0);
    accumulated_ms = 0.0f;
  }

  // Call each timestep to accumulate spike counts.
  void RecordSpikes(const NeuronArray& neurons) {
    for (size_t i = 0; i < neurons.n; ++i) {
      spike_count[i] += neurons.spiked[i];
    }
    accumulated_ms += dt_ms;
  }

  // Check if enough time has elapsed and apply adjustment.
  // Returns true if an update was performed.
  bool MaybeApply(NeuronArray& neurons) {
    if (accumulated_ms < update_interval_ms) return false;
    Apply(neurons);
    return true;
  }

  // Apply the homeostatic adjustment now.
  void Apply(NeuronArray& neurons) {
    if (accumulated_ms <= 0.0f) return;
    float window_s = accumulated_ms / 1000.0f;

    for (size_t i = 0; i < neurons.n; ++i) {
      float rate = static_cast<float>(spike_count[i]) / window_s;
      float error = target_rate_hz - rate;

      // Adjust bias: positive error (too silent) -> increase bias
      //              negative error (too active) -> decrease bias
      bias_current[i] += learning_rate * error;
      bias_current[i] = std::clamp(bias_current[i], -max_bias, max_bias);

      // Inject bias as external current
      neurons.i_ext[i] += bias_current[i];
    }

    // Reset accumulators
    std::fill(spike_count.begin(), spike_count.end(), 0);
    accumulated_ms = 0.0f;
  }

  // Get the mean bias current across all neurons.
  float MeanBias() const {
    if (bias_current.empty()) return 0.0f;
    float sum = 0.0f;
    for (float b : bias_current) sum += b;
    return sum / static_cast<float>(bias_current.size());
  }

  // Get the fraction of neurons with positive bias (excitability up).
  float FractionExcited() const {
    if (bias_current.empty()) return 0.0f;
    int count = 0;
    for (float b : bias_current) {
      if (b > 0.0f) count++;
    }
    return static_cast<float>(count) / static_cast<float>(bias_current.size());
  }
};

}  // namespace fwmc

#endif  // FWMC_INTRINSIC_HOMEOSTASIS_H_
