#ifndef FWMC_MULTI_TRIAL_H_
#define FWMC_MULTI_TRIAL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "conditioning_experiment.h"
#include "core/log.h"

namespace fwmc {

// Aggregate statistics from multiple conditioning trials.
struct MultiTrialStats {
  int n_trials = 0;

  // Learning index
  float learning_mean = 0.0f;
  float learning_std = 0.0f;
  float learning_min = 0.0f;
  float learning_max = 0.0f;

  // Discrimination index
  float discrimination_mean = 0.0f;
  float discrimination_std = 0.0f;

  // Behavioral learning
  float behavioral_mean = 0.0f;
  float behavioral_std = 0.0f;

  // Weight change ratio
  float weight_ratio_mean = 0.0f;
  float weight_ratio_std = 0.0f;

  // Firing rate validation
  float regions_in_range_mean = 0.0f;

  // How many trials showed learning (abs(learning_index) > 0.05)
  int n_learned = 0;
  float success_rate = 0.0f;

  // Timing
  double total_seconds = 0.0;
};

// Run N conditioning trials with different seeds and compute statistics.
//
// Usage:
//   MultiTrialRunner runner;
//   runner.n_trials = 20;
//   runner.base_config.n_training_trials = 5;
//   auto stats = runner.Run();
struct MultiTrialRunner {
  int n_trials = 10;
  ConditioningExperiment base_config;

  MultiTrialStats Run() {
    Log(LogLevel::kInfo, "=== Multi-Trial Analysis: %d trials ===", n_trials);

    std::vector<ConditioningResult> results;
    results.reserve(n_trials);

    for (int i = 0; i < n_trials; ++i) {
      ConditioningExperiment exp = base_config;
      exp.seed = base_config.seed + static_cast<uint32_t>(i);

      Log(LogLevel::kInfo, "--- Trial %d/%d (seed=%u) ---",
          i + 1, n_trials, exp.seed);

      results.push_back(exp.Run());
    }

    return ComputeStats(results);
  }

  static MultiTrialStats ComputeStats(
      const std::vector<ConditioningResult>& results) {
    MultiTrialStats stats;
    stats.n_trials = static_cast<int>(results.size());
    if (results.empty()) return stats;

    // Collect per-trial values
    std::vector<float> learning, disc, behav, wratio;
    float sum_regions = 0.0f;

    for (const auto& r : results) {
      learning.push_back(r.learning_index);
      disc.push_back(r.discrimination_index);
      behav.push_back(r.behavioral_learning);
      wratio.push_back(r.weight_change_ratio);
      sum_regions += static_cast<float>(r.regions_in_range);
      stats.total_seconds += r.elapsed_seconds;
      if (r.learned()) stats.n_learned++;
    }

    auto mean_std = [](const std::vector<float>& v) -> std::pair<float, float> {
      float sum = 0.0f;
      for (float x : v) sum += x;
      float m = sum / static_cast<float>(v.size());
      float ss = 0.0f;
      for (float x : v) ss += (x - m) * (x - m);
      float s = std::sqrt(ss / static_cast<float>(v.size()));
      return {m, s};
    };

    auto [lm, ls] = mean_std(learning);
    stats.learning_mean = lm;
    stats.learning_std = ls;
    stats.learning_min = *std::min_element(learning.begin(), learning.end());
    stats.learning_max = *std::max_element(learning.begin(), learning.end());

    auto [dm, ds] = mean_std(disc);
    stats.discrimination_mean = dm;
    stats.discrimination_std = ds;

    auto [bm, bs] = mean_std(behav);
    stats.behavioral_mean = bm;
    stats.behavioral_std = bs;

    auto [wm, ws] = mean_std(wratio);
    stats.weight_ratio_mean = wm;
    stats.weight_ratio_std = ws;

    stats.regions_in_range_mean =
        sum_regions / static_cast<float>(stats.n_trials);
    stats.success_rate =
        static_cast<float>(stats.n_learned) / static_cast<float>(stats.n_trials);

    // Report
    Log(LogLevel::kInfo, "=== Multi-Trial Results (%d trials) ===",
        stats.n_trials);
    Log(LogLevel::kInfo, "Learning index:   %.3f +/- %.3f  [%.3f, %.3f]",
        stats.learning_mean, stats.learning_std,
        stats.learning_min, stats.learning_max);
    Log(LogLevel::kInfo, "Discrimination:   %.3f +/- %.3f",
        stats.discrimination_mean, stats.discrimination_std);
    Log(LogLevel::kInfo, "Behavioral:       %.3f +/- %.3f",
        stats.behavioral_mean, stats.behavioral_std);
    Log(LogLevel::kInfo, "Weight ratio:     %.3f +/- %.3f",
        stats.weight_ratio_mean, stats.weight_ratio_std);
    Log(LogLevel::kInfo, "Regions in range: %.1f",
        stats.regions_in_range_mean);
    Log(LogLevel::kInfo, "Success rate:     %d/%d (%.0f%%)",
        stats.n_learned, stats.n_trials, stats.success_rate * 100.0f);
    Log(LogLevel::kInfo, "Total time:       %.2fs", stats.total_seconds);

    return stats;
  }
};

}  // namespace fwmc

#endif  // FWMC_MULTI_TRIAL_H_
