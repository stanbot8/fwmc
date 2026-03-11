#ifndef FWMC_SHADOW_TRACKER_H_
#define FWMC_SHADOW_TRACKER_H_

#include <cmath>
#include <deque>
#include <numeric>
#include <vector>
#include "bridge/bridge_channel.h"
#include "core/neuron_array.h"

namespace fwmc {

// Shadow mode: digital twin runs in parallel with biological brain,
// receiving the same inputs but NOT writing back. Measures how
// quickly the digital prediction diverges from biological reality.
// Phase 2 of the twinning protocol.
struct ShadowTracker {
  struct DriftSnapshot {
    float time_ms;
    float spike_correlation;
    float population_rmse;
    float mean_v_error;
    int n_false_positive;
    int n_false_negative;
    float time_since_resync;
  };

  std::deque<DriftSnapshot> history;
  size_t max_history_size = 10000;  // cap to prevent unbounded growth
  float last_resync_time = 0.0f;

  DriftSnapshot Measure(const NeuronArray& digital,
                        const std::vector<BioReading>& bio,
                        float sim_time_ms) {
    DriftSnapshot snap{};
    snap.time_ms = sim_time_ms;
    snap.time_since_resync = sim_time_ms - last_resync_time;

    if (bio.empty()) {
      history.push_back(snap);
      while (history.size() > max_history_size) history.pop_front();
      return snap;
    }

    std::vector<float> predicted, observed;
    float v_error_sum = 0.0f;
    int v_count = 0;

    for (const auto& b : bio) {
      if (b.neuron_idx >= digital.n) continue;
      float p = digital.spiked[b.neuron_idx] ? 1.0f : 0.0f;
      float o = b.spike_prob;
      predicted.push_back(p);
      observed.push_back(o);

      if (p > 0.5f && o < 0.3f) snap.n_false_positive++;
      if (p < 0.5f && o > 0.7f) snap.n_false_negative++;

      if (!std::isnan(b.voltage_mv)) {
        v_error_sum += std::abs(digital.v[b.neuron_idx] - b.voltage_mv);
        v_count++;
      }
    }

    size_t nn = predicted.size();
    if (nn == 0) {
      history.push_back(snap);
      while (history.size() > max_history_size) history.pop_front();
      return snap;
    }

    // Pearson correlation
    float mean_p = std::accumulate(predicted.begin(), predicted.end(), 0.0f) / nn;
    float mean_o = std::accumulate(observed.begin(), observed.end(), 0.0f) / nn;
    float cov = 0, var_p = 0, var_o = 0;
    for (size_t i = 0; i < nn; ++i) {
      float dp = predicted[i] - mean_p;
      float d_o = observed[i] - mean_o;
      cov += dp * d_o;
      var_p += dp * dp;
      var_o += d_o * d_o;
    }
    float denom = std::sqrt(var_p * var_o);
    snap.spike_correlation = (denom > 1e-9f) ? (cov / denom) : 0.0f;

    // RMSE
    float mse = 0;
    for (size_t i = 0; i < nn; ++i) {
      float diff = predicted[i] - observed[i];
      mse += diff * diff;
    }
    snap.population_rmse = std::sqrt(mse / nn);
    snap.mean_v_error = (v_count > 0) ? (v_error_sum / v_count) : 0.0f;

    history.push_back(snap);
    while (history.size() > max_history_size) {
      history.pop_front();
    }
    return snap;
  }

  // Resynchronize: copy biological state into digital twin
  void Resync(NeuronArray& digital, const std::vector<BioReading>& bio,
              float sim_time_ms) {
    for (const auto& b : bio) {
      if (b.neuron_idx >= digital.n) continue;
      if (!std::isnan(b.voltage_mv)) {
        digital.v[b.neuron_idx] = b.voltage_mv;
      } else if (b.spike_prob > 0.7f) {
        digital.v[b.neuron_idx] = 30.0f;
        digital.spiked[b.neuron_idx] = 1;
      }
    }
    last_resync_time = sim_time_ms;
  }

  bool DriftExceedsThreshold(float threshold = 0.5f) const {
    if (history.empty()) return false;
    return history.back().spike_correlation < threshold;
  }
};

}  // namespace fwmc

#endif  // FWMC_SHADOW_TRACKER_H_
