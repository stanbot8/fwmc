#ifndef FWMC_EXPERIMENT_RUNNER_H_
#define FWMC_EXPERIMENT_RUNNER_H_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "bridge/bridge_channel.h"
#include "bridge/bridge_checkpoint.h"
#include "bridge/calibrator.h"
#include "bridge/file_read_channel.h"
#include "bridge/stimulus.h"
#include "bridge/twin_bridge.h"
#include "core/checkpoint.h"
#include "core/connectome_loader.h"
#include "core/connectome_stats.h"
#include "core/experiment_config.h"
#include "core/log.h"
#include "core/intrinsic_homeostasis.h"
#include "core/motor_output.h"
#include "core/rate_monitor.h"
#include "core/recorder.h"
#include "core/stdp.h"

namespace fwmc {

// Runs a complete experiment from config to recorded results.
// This is the top-level orchestrator that wires together:
//   Connectome → Bridge → Stimulus → Recorder → Analysis
struct ExperimentRunner {
  ExperimentConfig config;
  TwinBridge bridge;
  StimulusController stimulus;
  Recorder recorder;
  Calibrator calibrator;
  STDPParams stdp_params;
  SynapticScaling scaling;
  int scaling_interval = 1000;  // apply scaling every N steps
  RateMonitor rate_monitor;
  MotorOutput motor;
  IntrinsicHomeostasis homeostasis;
  int rate_report_interval = 10000;  // log firing rates every N steps

  // Run the full experiment. Returns 0 on success.
  int Run() {
    Log(LogLevel::kInfo, "=== Experiment: %s ===", config.name.c_str());
    Log(LogLevel::kInfo, "Fly: %s  Date: %s", config.fly_strain.c_str(), config.date.c_str());

    // 1. Load connectome
    NeuronArray neurons;
    SynapseTable synapses;

    std::string neuron_path = config.connectome_dir + "/neurons.bin";
    std::string synapse_path = config.connectome_dir + "/synapses.bin";

    auto nr = ConnectomeLoader::LoadNeurons(neuron_path, neurons);
    if (!nr) {
      Log(LogLevel::kError, "%s", nr.error().message.c_str());
      return 1;
    }
    auto sr = ConnectomeLoader::LoadSynapses(synapse_path, neurons.n, synapses);
    if (!sr) {
      Log(LogLevel::kError, "%s", sr.error().message.c_str());
      return 1;
    }

    Log(LogLevel::kInfo, "%zu neurons, %zu synapses loaded", neurons.n, synapses.Size());

    // 1b. Validate and log connectome statistics
    ConnectomeStats stats;
    if (!stats.Compute(synapses, neurons)) {
      Log(LogLevel::kError, "Connectome validation failed, aborting");
      return 1;
    }
    stats.LogSummary();

    // 2. Apply per-cell-type parameters
    ApplyCellTypeParams(neurons);

    // 3. Configure bridge
    bridge.digital = std::move(neurons);
    bridge.synapses = std::move(synapses);
    bridge.dt_ms = config.dt_ms;
    bridge.weight_scale = config.weight_scale;

    switch (config.bridge_mode) {
      case 1: bridge.mode = BridgeMode::kShadow; break;
      case 2: bridge.mode = BridgeMode::kClosedLoop; break;
      default: bridge.mode = BridgeMode::kOpenLoop; break;
    }

    bridge.resync_threshold = config.resync_threshold;

    // 4. Configure I/O channels
    if (bridge.mode != BridgeMode::kOpenLoop) {
      if (!config.recording_input.empty()) {
        auto file_reader = std::make_unique<FileReadChannel>();
        if (!file_reader->Open(config.recording_input)) return 1;
        bridge.read_channel = std::move(file_reader);
      } else {
        bridge.read_channel = std::make_unique<SimulatedRead>();
      }
      bridge.write_channel = std::make_unique<SimulatedWrite>();

      bridge.replacer.Init(bridge.digital.n);
      bridge.replacer.monitor_threshold = config.monitor_threshold;
      bridge.replacer.bridge_threshold = config.bridge_threshold;
      bridge.replacer.min_observation_ms = config.min_observation_ms;

      if (!config.monitor_neurons.empty()) {
        bridge.replacer.BeginMonitoring(config.monitor_neurons);
        Log(LogLevel::kInfo, "Monitoring %zu neurons for replacement",
            config.monitor_neurons.size());
      }
    }

    // 5. Load stimulus protocol
    stimulus.LoadProtocol(config.stimulus_protocol);
    if (!config.stimulus_protocol.empty()) {
      Log(LogLevel::kInfo, "%zu stimulus events loaded",
          config.stimulus_protocol.size());
    }

    // 6. Initialize recorder
    recorder.record_spikes = config.record_spikes;
    recorder.record_voltages = config.record_voltages;
    recorder.record_shadow_metrics = config.record_shadow_metrics;
    recorder.record_per_neuron_error = config.record_per_neuron_error;
    recorder.recording_interval = config.recording_interval;

    if (!recorder.Open(config.output_dir,
                       static_cast<uint32_t>(bridge.digital.n))) {
      return 1;
    }

    // 7. Initialize calibrator
    if (bridge.mode != BridgeMode::kOpenLoop &&
        config.calibration_interval > 0) {
      calibrator.Init(bridge.synapses.Size());
      calibrator.learning_rate = config.calibration_lr;
      Log(LogLevel::kInfo, "Calibrator enabled: lr=%.4f interval=%d steps",
          config.calibration_lr, config.calibration_interval);
    }

    // 7b. Initialize synaptic scaling if STDP is enabled
    if (config.enable_stdp) {
      scaling.Init(bridge.digital.n);
      if (stdp_params.use_eligibility_traces && stdp_params.dopamine_gated) {
        bridge.synapses.InitEligibilityTraces();
        Log(LogLevel::kInfo, "Eligibility traces enabled (tau=%.0fms)",
            stdp_params.tau_eligibility_ms);
      }
    }

    // 7c. Initialize rate monitor and motor output
    rate_monitor.Init(bridge.digital, config.dt_ms);
    // Motor output: auto-detect SEZ (region 12) and MBON regions
    motor.InitFromRegions(bridge.digital, 12, 3);
    if (motor.HasMotorNeurons()) {
      Log(LogLevel::kInfo, "Motor output: %zu neurons (L=%zu R=%zu approach=%zu avoid=%zu)",
          motor.TotalNeurons(), motor.descending_left.size(),
          motor.descending_right.size(), motor.approach_neurons.size(),
          motor.avoid_neurons.size());
    }

    // 7c2. Initialize intrinsic homeostasis
    homeostasis.Init(bridge.digital.n, 5.0f, config.dt_ms);

    // 7d. Copy config to output directory for reproducibility
    SaveConfigCopy();

    // 8. Run simulation
    auto t0 = std::chrono::high_resolution_clock::now();
    int n_steps = static_cast<int>(config.duration_ms / config.dt_ms);

    // Pre-allocate per-neuron error buffer outside the loop
    std::vector<float> per_neuron_err;
    if (config.record_per_neuron_error && bridge.read_channel) {
      per_neuron_err.resize(bridge.digital.n, 0.0f);
    }

    for (int step = 0; step < n_steps; ++step) {
      // Clear transient external input, then apply stimulus protocol
      bridge.digital.ClearExternalInput();
      stimulus.Apply(bridge.sim_time_ms, bridge.digital);

      // Step the bridge
      bridge.Step();

      // Neuromodulator dynamics and STDP
      if (config.enable_stdp) {
        NeuromodulatorUpdate(bridge.digital, bridge.synapses, config.dt_ms);
        STDPUpdate(bridge.synapses, bridge.digital, bridge.sim_time_ms,
                   stdp_params);
        // Three-factor learning: convert eligibility traces using dopamine
        if (stdp_params.use_eligibility_traces) {
          EligibilityTraceUpdate(bridge.synapses, bridge.digital,
                                  config.dt_ms, stdp_params);
        }
        // Synaptic scaling homeostasis
        scaling.AccumulateSpikes(bridge.digital, config.dt_ms);
        if (step > 0 && step % scaling_interval == 0) {
          scaling.Apply(bridge.synapses, stdp_params);
        }
      }

      // Record data
      if (step % recorder.recording_interval == 0) {
        DriftMetrics drift_metrics;
        const DriftMetrics* drift = nullptr;
        if (!bridge.shadow.history.empty()) {
          const auto& snap = bridge.shadow.history.back();
          drift_metrics.spike_correlation = snap.spike_correlation;
          drift_metrics.population_rmse = snap.population_rmse;
          drift_metrics.mean_v_error = snap.mean_v_error;
          drift_metrics.n_false_positive = snap.n_false_positive;
          drift_metrics.n_false_negative = snap.n_false_negative;
          drift = &drift_metrics;
        }

        // Per-neuron error computation (reuse pre-allocated buffer)
        bool has_per_neuron = false;
        if (config.record_per_neuron_error && bridge.read_channel) {
          auto bio = bridge.read_channel->ReadFrame(bridge.sim_time_ms);
          std::fill(per_neuron_err.begin(), per_neuron_err.end(), 0.0f);
          for (const auto& b : bio) {
            if (b.neuron_idx < bridge.digital.n) {
              float pred = bridge.digital.spiked[b.neuron_idx] ? 1.0f : 0.0f;
              per_neuron_err[b.neuron_idx] = std::abs(pred - b.spike_prob);
            }
          }
          has_per_neuron = true;
        }

        recorder.RecordStep(bridge.sim_time_ms, bridge.digital,
                            drift, bridge.total_resyncs,
                            bridge.replacer.ReplacementFraction() * 100.0f,
                            has_per_neuron ? &per_neuron_err : nullptr);
      }

      // Calibration: accumulate error and periodically apply gradients
      if (bridge.mode != BridgeMode::kOpenLoop && bridge.read_channel &&
          config.calibration_interval > 0) {
        auto bio = bridge.read_channel->ReadFrame(bridge.sim_time_ms);
        calibrator.AccumulateError(bridge.synapses, bridge.digital, bio);

        if (step > 0 && step % config.calibration_interval == 0) {
          calibrator.ApplyGradients(bridge.synapses);
        }
      }

      // Periodic checkpoint (every 10% of simulation)
      int checkpoint_every = n_steps / 10;
      if (checkpoint_every > 0 && step > 0 && step % checkpoint_every == 0) {
        std::string ckpt_path = config.output_dir + "/checkpoint.bin";
        auto ext = BridgeCheckpoint::Serialize(bridge.replacer, bridge.shadow);
        Checkpoint::Save(ckpt_path, bridge.sim_time_ms, bridge.total_steps,
                         bridge.total_resyncs, bridge.digital, bridge.synapses,
                         ext);
      }

      // Rate monitor, motor output, and homeostasis
      rate_monitor.RecordStep(bridge.digital);
      motor.Update(bridge.digital, config.dt_ms);
      homeostasis.RecordSpikes(bridge.digital);
      homeostasis.MaybeApply(bridge.digital);

      // Log progress
      if (step % config.metrics_interval == 0) {
        int spikes = bridge.digital.CountSpikes();
        if (!bridge.shadow.history.empty()) {
          Log(LogLevel::kInfo, "t=%.1fms  spikes=%d  corr=%.3f  resyncs=%d",
              bridge.sim_time_ms, spikes,
              bridge.shadow.history.back().spike_correlation,
              bridge.total_resyncs);
        } else {
          Log(LogLevel::kInfo, "t=%.1fms  spikes=%d",
              bridge.sim_time_ms, spikes);
        }
      }

      // Periodic rate report with literature comparison
      if (rate_report_interval > 0 && step > 0 &&
          step % rate_report_interval == 0) {
        auto rates = rate_monitor.ComputeRates();
        Log(LogLevel::kInfo, "--- Firing rate check (t=%.1fms) ---",
            bridge.sim_time_ms);
        RateMonitor::LogRates(rates);
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double realtime_ratio = (config.duration_ms / 1000.0) / elapsed;

    // 9. Close recorder and save final checkpoint
    recorder.Close();

    std::string final_ckpt = config.output_dir + "/checkpoint.bin";
    auto final_ext = BridgeCheckpoint::Serialize(bridge.replacer, bridge.shadow);
    Checkpoint::Save(final_ckpt, bridge.sim_time_ms, bridge.total_steps,
                     bridge.total_resyncs, bridge.digital, bridge.synapses,
                     final_ext);

    // 10. Report summary
    Log(LogLevel::kInfo, "=== Experiment complete ===");
    Log(LogLevel::kInfo, "%.1fms simulated in %.3fs (%.1fx real-time)",
        config.duration_ms, elapsed, realtime_ratio);
    Log(LogLevel::kInfo, "Results saved to: %s", config.output_dir.c_str());

    if (bridge.mode != BridgeMode::kOpenLoop && !bridge.shadow.history.empty()) {
      auto& last = bridge.shadow.history.back();
      Log(LogLevel::kInfo, "Final correlation: %.4f  RMSE: %.4f",
          last.spike_correlation, last.population_rmse);
      Log(LogLevel::kInfo, "Total resyncs: %d  Replaced: %.1f%%",
          bridge.total_resyncs, bridge.replacer.ReplacementFraction() * 100.0f);
    }

    // Final firing rate report
    auto final_rates = rate_monitor.ComputeRates();
    if (!final_rates.empty()) {
      int in_range = RateMonitor::CountInRange(final_rates);
      Log(LogLevel::kInfo, "Firing rates: %d/%zu regions in biological range",
          in_range, final_rates.size());
      RateMonitor::LogRates(final_rates);
    }

    // Motor output summary
    if (motor.HasMotorNeurons()) {
      auto& cmd = motor.Command();
      Log(LogLevel::kInfo, "Motor: fwd=%.2f mm/s  turn=%.2f rad/s  "
          "approach=%.2f  freeze=%.1f",
          cmd.forward_velocity, cmd.angular_velocity,
          cmd.approach_drive, cmd.freeze);
    }

    return 0;
  }

 private:
  void SaveConfigCopy() {
    std::string path = config.output_dir + "/experiment.cfg";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "# Auto-saved experiment config (reproducibility)\n");
    fprintf(f, "name = %s\n", config.name.c_str());
    fprintf(f, "fly_strain = %s\n", config.fly_strain.c_str());
    fprintf(f, "date = %s\n", config.date.c_str());
    fprintf(f, "notes = %s\n", config.notes.c_str());
    fprintf(f, "dt_ms = %g\n", config.dt_ms);
    fprintf(f, "duration_ms = %g\n", config.duration_ms);
    fprintf(f, "weight_scale = %g\n", config.weight_scale);
    fprintf(f, "metrics_interval = %d\n", config.metrics_interval);
    fprintf(f, "enable_stdp = %s\n", config.enable_stdp ? "true" : "false");
    fprintf(f, "bridge_mode = %d\n", config.bridge_mode);
    fprintf(f, "monitor_threshold = %g\n", config.monitor_threshold);
    fprintf(f, "bridge_threshold = %g\n", config.bridge_threshold);
    fprintf(f, "resync_threshold = %g\n", config.resync_threshold);
    fprintf(f, "min_observation_ms = %g\n", config.min_observation_ms);
    fprintf(f, "calibration_interval = %d\n", config.calibration_interval);
    fprintf(f, "calibration_lr = %g\n", config.calibration_lr);
    fprintf(f, "connectome_dir = %s\n", config.connectome_dir.c_str());
    if (!config.recording_input.empty())
      fprintf(f, "recording_input = %s\n", config.recording_input.c_str());
    fprintf(f, "output_dir = %s\n", config.output_dir.c_str());
    fprintf(f, "record_spikes = %s\n", config.record_spikes ? "true" : "false");
    fprintf(f, "record_voltages = %s\n", config.record_voltages ? "true" : "false");
    fprintf(f, "record_shadow_metrics = %s\n", config.record_shadow_metrics ? "true" : "false");
    fprintf(f, "record_per_neuron_error = %s\n", config.record_per_neuron_error ? "true" : "false");
    fprintf(f, "recording_interval = %d\n", config.recording_interval);
    for (const auto& ev : config.stimulus_protocol) {
      fprintf(f, "stimulus: %s %g %g %g", ev.label.c_str(),
              ev.start_ms, ev.end_ms, ev.intensity);
      for (size_t j = 0; j < ev.target_neurons.size(); ++j) {
        fprintf(f, "%s%u", j == 0 ? " " : ",", ev.target_neurons[j]);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  }

  void ApplyCellTypeParams(NeuronArray& neurons) {
    if (config.neuron_types.empty()) return;

    int typed = 0;
    for (const auto& [idx, ct] : config.neuron_types) {
      if (idx < neurons.n) {
        neurons.type[idx] = static_cast<uint8_t>(ct);
        typed++;
      }
    }
    Log(LogLevel::kInfo, "Applied cell type parameters to %d neurons", typed);
  }
};

}  // namespace fwmc

#endif  // FWMC_EXPERIMENT_RUNNER_H_
