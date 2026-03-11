#include <charconv>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include "core/log.h"
#include "core/version.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/connectome_loader.h"
#include "core/cell_types.h"
#include "core/connectome_stats.h"
#include "core/checkpoint.h"
#include "core/config_loader.h"
#include "core/izhikevich.h"
#include "core/param_sweep.h"
#include "core/parametric_gen.h"
#include "core/brain_spec_loader.h"
#include "core/parametric_sync.h"
#include "core/connectome_export.h"
#include "core/recorder.h"
#include "core/region_metrics.h"
#include "core/stdp.h"
#include "core/structural_plasticity.h"
#include "bridge/twin_bridge.h"
#include "bridge/bridge_checkpoint.h"
#include "experiment_runner.h"
#include "conditioning_experiment.h"
#include "multi_trial.h"
#include "bridge_self_test.h"

namespace fwmc {

// Built-in profiles map to brain spec files in examples/.
// The default profile ("flywire") uses the full Drosophila brain.
struct Profile {
  const char* name;
  const char* brain_spec;  // relative path to .brain file
  const char* description;
};

static constexpr Profile kProfiles[] = {
  {"flywire",  "examples/drosophila_full.brain",         "Full Drosophila brain (~139k neurons, FlyWire-scale)"},
  {"mb",       "examples/parametric_mushroom_body.brain", "Mushroom body circuit (2500 neurons)"},
  {"al",       "examples/antennal_lobe.brain",            "Antennal lobe olfactory circuit (1300 neurons)"},
  {"cx",       "examples/central_complex.brain",          "Central complex navigation circuit"},
};
static constexpr size_t kNumProfiles = sizeof(kProfiles) / sizeof(kProfiles[0]);

static const Profile* FindProfile(std::string_view name) {
  for (size_t i = 0; i < kNumProfiles; ++i) {
    if (name == kProfiles[i].name) return &kProfiles[i];
  }
  return nullptr;
}

struct Config {
  const char* data_dir = "data";
  const char* experiment_config = nullptr;
  const char* checkpoint_path = nullptr;   // --checkpoint: save state here
  const char* resume_path = nullptr;       // --resume: restore state from here
  float dt_ms = 0.1f;
  float duration_ms = 1000.0f;
  float weight_scale = 1.0f;
  int metrics_interval = 1000;
  int checkpoint_interval = 0;            // steps between checkpoints (0=disabled)
  bool stdp = false;
  bool stats = false;                     // --stats: print connectome statistics
  bool show_help = false;
  bool show_version = false;
  const char* parametric_brain = nullptr; // --parametric: generate from brain spec
  const char* profile = nullptr;         // --profile: named brain profile (default: flywire)
  bool sweep = false;                     // --sweep: run parameter sweep
  float sweep_target_hz = 10.0f;          // --sweep-target: target firing rate
  const char* sync_ref = nullptr;        // --sync: reference brain spec for sync mode
  float sync_convergence = 0.95f;        // --sync-target: convergence threshold
  const char* export_dir = nullptr;      // --export: export parametric brain to binary
  bool structural_plasticity = false;    // --plasticity: enable synapse pruning/sprouting
  bool stochastic = false;              // --stochastic: enable Monte Carlo synaptic release
  bool stp = false;                     // --stp: enable short-term plasticity (Tsodyks-Markram)
  const char* output_dir = nullptr;     // --output: record spikes to directory
  int recording_interval = 10;          // --record-interval: steps between recordings
  BridgeMode bridge_mode = BridgeMode::kOpenLoop;
  bool conditioning = false;            // --conditioning: run olfactory conditioning demo
  bool bridge_test = false;             // --bridge-test: run bridge self-test
  uint32_t seed = 42;                  // --seed: deterministic random seed
  int multi_trial = 0;                // --multi-trial N: run N conditioning trials
  bool explicit_data = false;          // true if --data was explicitly passed
};

static float ParseFloat(const char* s) {
  float val = 0;
  std::from_chars(s, s + std::string_view(s).size(), val);
  return val;
}

static int ParseInt(const char* s) {
  int val = 0;
  std::from_chars(s, s + std::string_view(s).size(), val);
  return val;
}

static void PrintUsage() {
  Log(LogLevel::kInfo,
      "Usage: fwmc [OPTIONS]\n\n"
      "  --profile NAME        Brain profile (default: flywire)\n"
      "                        Available: flywire, mb, al, cx\n"
      "  --experiment CONFIG   Run from experiment config file\n"
      "  --data DIR            Connectome data directory (default: data)\n"
      "  --dt MS               Timestep in ms (default: 0.1)\n"
      "  --duration MS         Simulation duration in ms (default: 1000)\n"
      "  --weight-scale F      Global synaptic weight multiplier (default: 1.0)\n"
      "  --metrics N           Print metrics every N steps (default: 1000)\n"
      "  --stdp                Enable spike-timing-dependent plasticity\n"
      "  --shadow              Shadow mode (read bio, measure drift)\n"
      "  --closed-loop         Full bidirectional bridge\n"
      "  --stats               Print connectome statistics after loading\n"
      "  --checkpoint PATH     Save checkpoint to PATH at end of run\n"
      "  --checkpoint-every N  Save checkpoint every N steps (0=disabled)\n"
      "  --resume PATH         Resume simulation from checkpoint file\n"
      "  --parametric FILE     Generate connectome from brain spec file\n"
      "  --sweep               Run parameter sweep (with --parametric)\n"
      "  --sweep-target HZ     Target firing rate for sweep (default: 10)\n"
      "  --sync FILE           Sync parametric brain to reference brain spec\n"
      "  --sync-target F       Sync convergence threshold 0-1 (default: 0.95)\n"
      "  --export DIR          Export parametric brain to binary files\n"
      "  --plasticity          Enable structural plasticity (pruning/sprouting)\n"
      "  --stochastic          Enable Monte Carlo synaptic release\n"
      "  --stp                 Enable short-term plasticity (Tsodyks-Markram)\n"
      "  --output DIR          Record spikes to output directory\n"
      "  --record-interval N   Steps between spike recordings (default: 10)\n"
      "  --conditioning        Run olfactory conditioning demo\n"
      "  --multi-trial N       Run N conditioning trials with statistics\n"
      "  --bridge-test         Run bridge self-test (no hardware needed)\n"
      "  --seed N              Random seed for reproducibility (default: 42)\n"
      "  --help                Show this help message\n"
      "  --version             Show version information");
}

Config ParseArgs(int argc, const char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--experiment" && i + 1 < argc)
      cfg.experiment_config = argv[++i];
    else if (arg == "--data" && i + 1 < argc) {
      cfg.data_dir = argv[++i];
      cfg.explicit_data = true;
    }
    else if (arg == "--profile" && i + 1 < argc)
      cfg.profile = argv[++i];
    else if (arg == "--dt" && i + 1 < argc)
      cfg.dt_ms = ParseFloat(argv[++i]);
    else if (arg == "--duration" && i + 1 < argc)
      cfg.duration_ms = ParseFloat(argv[++i]);
    else if (arg == "--weight-scale" && i + 1 < argc)
      cfg.weight_scale = ParseFloat(argv[++i]);
    else if (arg == "--metrics" && i + 1 < argc)
      cfg.metrics_interval = ParseInt(argv[++i]);
    else if (arg == "--stdp")
      cfg.stdp = true;
    else if (arg == "--shadow")
      cfg.bridge_mode = BridgeMode::kShadow;
    else if (arg == "--closed-loop")
      cfg.bridge_mode = BridgeMode::kClosedLoop;
    else if (arg == "--stats")
      cfg.stats = true;
    else if (arg == "--checkpoint" && i + 1 < argc)
      cfg.checkpoint_path = argv[++i];
    else if (arg == "--checkpoint-every" && i + 1 < argc)
      cfg.checkpoint_interval = ParseInt(argv[++i]);
    else if (arg == "--resume" && i + 1 < argc)
      cfg.resume_path = argv[++i];
    else if (arg == "--parametric" && i + 1 < argc)
      cfg.parametric_brain = argv[++i];
    else if (arg == "--sweep")
      cfg.sweep = true;
    else if (arg == "--sweep-target" && i + 1 < argc)
      cfg.sweep_target_hz = ParseFloat(argv[++i]);
    else if (arg == "--sync" && i + 1 < argc)
      cfg.sync_ref = argv[++i];
    else if (arg == "--sync-target" && i + 1 < argc)
      cfg.sync_convergence = ParseFloat(argv[++i]);
    else if (arg == "--export" && i + 1 < argc)
      cfg.export_dir = argv[++i];
    else if (arg == "--plasticity")
      cfg.structural_plasticity = true;
    else if (arg == "--stochastic")
      cfg.stochastic = true;
    else if (arg == "--stp")
      cfg.stp = true;
    else if (arg == "--output" && i + 1 < argc)
      cfg.output_dir = argv[++i];
    else if (arg == "--record-interval" && i + 1 < argc)
      cfg.recording_interval = ParseInt(argv[++i]);
    else if (arg == "--conditioning")
      cfg.conditioning = true;
    else if (arg == "--multi-trial" && i + 1 < argc)
      cfg.multi_trial = ParseInt(argv[++i]);
    else if (arg == "--bridge-test")
      cfg.bridge_test = true;
    else if (arg == "--seed" && i + 1 < argc)
      cfg.seed = static_cast<uint32_t>(ParseInt(argv[++i]));
    else if (arg == "--help" || arg == "-h")
      cfg.show_help = true;
    else if (arg == "--version")
      cfg.show_version = true;
    else {
      Log(LogLevel::kError, "Unknown option: %s", argv[i]);
      PrintUsage();
      exit(1);
    }
  }
  return cfg;
}

int RunExperiment(const char* config_path) {
  auto result = ConfigLoader::Load(config_path);
  if (!result) {
    Log(LogLevel::kError, "%s", result.error().message.c_str());
    return 1;
  }

  ExperimentRunner runner;
  runner.config = std::move(*result);
  return runner.Run();
}

int RunParametric(const Config& cfg) {
  Log(LogLevel::kInfo, "FlyWire Mind Couple v1.0 - Parametric brain generator");

  auto spec_result = BrainSpecLoader::Load(cfg.parametric_brain);
  if (!spec_result) {
    Log(LogLevel::kError, "%s", spec_result.error().message.c_str());
    return 1;
  }

  auto& spec = *spec_result;
  Log(LogLevel::kInfo, "Brain spec: %s (%zu regions, %zu projections)",
      spec.name.c_str(), spec.regions.size(), spec.projections.size());

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);

  if (cfg.stats) {
    ConnectomeStats cstats;
    cstats.Compute(synapses, neurons);
    cstats.LogSummary();
  }

  // Parameter sweep mode
  if (cfg.sweep) {
    Log(LogLevel::kInfo, "Running parameter sweep (target %.1f Hz)...",
        cfg.sweep_target_hz);

    // Inject baseline current for sweep evaluation
    for (size_t i = 0; i < neurons.n; ++i) neurons.i_ext[i] = 8.0f;

    ParamSweep sweep;
    sweep.grid_steps = 4;
    sweep.sim_duration_ms = std::min(500.0f, cfg.duration_ms);
    sweep.dt_ms = cfg.dt_ms;
    sweep.weight_scale = cfg.weight_scale;

    // Sweep for each cell type present
    std::vector<uint8_t> seen_types;
    for (size_t i = 0; i < total; ++i) {
      uint8_t t = neurons.type[i];
      bool found = false;
      for (auto s : seen_types) { if (s == t) { found = true; break; } }
      if (!found) seen_types.push_back(t);
    }

    for (uint8_t t : seen_types) {
      auto ct = static_cast<CellType>(t);
      sweep.GridSweep(ct, neurons, synapses,
                      scoring::TargetFiringRate(cfg.sweep_target_hz, cfg.dt_ms));
      sweep.Refine(ct, neurons, synapses,
                   scoring::TargetFiringRate(cfg.sweep_target_hz, cfg.dt_ms),
                   30, 0.5f);

      // Apply best params
      types.SetOverride(ct, sweep.BestParams());
    }
    types.AssignFromTypes(neurons);
    Log(LogLevel::kInfo, "Sweep complete. Running simulation with optimized params.");
  }

  // Initialize region metrics
  RegionMetrics region_metrics;
  region_metrics.Init(gen);

  bool has_stimuli = !spec.stimuli.empty();
  if (has_stimuli) {
    Log(LogLevel::kInfo, "%zu stimuli defined", spec.stimuli.size());
  }

  // Structural plasticity
  StructuralPlasticity plasticity;
  std::mt19937 plasticity_rng(spec.seed + 7);

  // Background noise generator (models tonic synaptic bombardment)
  bool has_background = (spec.background_current_mean != 0.0f ||
                         spec.background_current_std != 0.0f);
  std::mt19937 noise_rng(spec.seed + 13);
  std::normal_distribution<float> noise_dist(spec.background_current_mean,
                                              spec.background_current_std);

  // Initialize stochastic release if requested (and brain spec has p_release < 1)
  // The parametric generator already populates p_release from brain spec.
  // If --stochastic is set but no p_release was specified, default to 0.5.
  std::mt19937 release_rng(spec.seed + 7);
  if (cfg.stochastic && !synapses.HasStochasticRelease()) {
    synapses.InitReleaseProbability(0.5f);
  }

  // Initialize short-term plasticity if requested
  if (cfg.stp) {
    STPParams stp_params;
    stp_params.U_se = 0.5f;
    stp_params.tau_d = 200.0f;
    stp_params.tau_f = 50.0f;
    synapses.InitSTP(stp_params);
  }

  // Set up recording if output dir specified
  Recorder recorder;
  if (cfg.output_dir) {
    recorder.record_spikes = true;
    recorder.record_voltages = false;
    recorder.record_shadow_metrics = false;
    recorder.record_per_neuron_error = false;
    recorder.recording_interval = cfg.recording_interval;
    recorder.Open(cfg.output_dir, static_cast<uint32_t>(neurons.n));
  }

  // Run simulation with heterogeneous dynamics
  auto t0 = std::chrono::high_resolution_clock::now();
  int n_steps = static_cast<int>(cfg.duration_ms / cfg.dt_ms);
  STDPParams stdp_params;
  float sim_time = 0.0f;

  for (int step = 0; step < n_steps; ++step) {
    // Clear transient external input, then apply timed stimuli
    neurons.ClearExternalInput();

    // Inject background noise current (tonic drive + variability)
    if (has_background) {
      for (size_t i = 0; i < neurons.n; ++i) {
        neurons.i_ext[i] += noise_dist(noise_rng);
      }
    }

    if (has_stimuli) {
      ApplyStimuli(spec.stimuli, gen.region_ranges, neurons, sim_time,
                   spec.seed);
    }

    neurons.ClearSynapticInput();
    if (synapses.HasStochasticRelease()) {
      synapses.PropagateSpikesMonteCarlo(neurons.spiked.data(),
                                          neurons.i_syn.data(),
                                          cfg.weight_scale, release_rng);
    } else {
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(),
                               cfg.weight_scale);
    }
    if (synapses.HasSTP()) {
      synapses.RecoverSTP(cfg.dt_ms);
    }
    IzhikevichStepHeterogeneous(neurons, cfg.dt_ms, sim_time, types);

    if (cfg.stdp) {
      STDPUpdate(synapses, neurons, sim_time, stdp_params);
    }

    if (cfg.structural_plasticity) {
      plasticity.Update(synapses, neurons, step, plasticity_rng);
    }

    sim_time += cfg.dt_ms;

    if (cfg.output_dir && step % recorder.recording_interval == 0) {
      recorder.RecordStep(sim_time, neurons, nullptr, 0, 0.0f, nullptr);
    }

    if (step % cfg.metrics_interval == 0) {
      region_metrics.Record(neurons, sim_time, cfg.dt_ms,
                            cfg.metrics_interval);
      int spikes = neurons.CountSpikes();
      Log(LogLevel::kInfo, "t=%.1fms  spikes=%d", sim_time, spikes);
      region_metrics.LogLatest();
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  double realtime_ratio = (cfg.duration_ms / 1000.0) / elapsed;

  if (cfg.output_dir) recorder.Close();

  region_metrics.LogSummary();
  Log(LogLevel::kInfo, "Done: %u neurons, %.1fms simulated in %.3fs (%.1fx real-time)",
      total, cfg.duration_ms, elapsed, realtime_ratio);

  // Export parametric brain to binary format
  if (cfg.export_dir) {
    std::string npath = std::string(cfg.export_dir) + "/neurons.bin";
    std::string spath = std::string(cfg.export_dir) + "/synapses.bin";
    auto nr = ConnectomeExport::ExportNeurons(npath, neurons);
    auto sr = ConnectomeExport::ExportSynapses(spath, synapses);
    if (!nr) Log(LogLevel::kError, "%s", nr.error().message.c_str());
    if (!sr) Log(LogLevel::kError, "%s", sr.error().message.c_str());
  }

  if (cfg.checkpoint_path) {
    Checkpoint::Save(cfg.checkpoint_path, sim_time, n_steps, 0,
                     neurons, synapses);
  }

  return 0;
}

int RunParametricSync(const Config& cfg) {
  Log(LogLevel::kInfo, "FlyWire Mind Couple v1.0 - Parametric sync mode");

  // Load model brain spec
  auto model_result = BrainSpecLoader::Load(cfg.parametric_brain);
  if (!model_result) {
    Log(LogLevel::kError, "%s", model_result.error().message.c_str());
    return 1;
  }

  // Load reference brain spec
  auto ref_result = BrainSpecLoader::Load(cfg.sync_ref);
  if (!ref_result) {
    Log(LogLevel::kError, "%s", ref_result.error().message.c_str());
    return 1;
  }

  // Generate model brain
  NeuronArray model_neurons;
  SynapseTable model_synapses;
  CellTypeManager model_types;
  ParametricGenerator model_gen;
  uint32_t model_n = model_gen.Generate(*model_result, model_neurons,
                                         model_synapses, model_types);

  // Generate reference brain
  NeuronArray ref_neurons;
  SynapseTable ref_synapses;
  CellTypeManager ref_types;
  ParametricGenerator ref_gen;
  uint32_t ref_n = ref_gen.Generate(*ref_result, ref_neurons,
                                     ref_synapses, ref_types);

  if (model_n != ref_n) {
    Log(LogLevel::kError, "Neuron count mismatch: model=%u ref=%u", model_n, ref_n);
    return 1;
  }

  Log(LogLevel::kInfo, "Sync: %u neurons, model=%zu synapses, ref=%zu synapses",
      model_n, model_synapses.Size(), ref_synapses.Size());

  // Inject baseline current
  for (size_t i = 0; i < model_n; ++i) {
    model_neurons.i_ext[i] = 8.0f;
    ref_neurons.i_ext[i] = 8.0f;
  }

  // Initialize sync engine
  ParametricSync sync;
  sync.dt_ms = cfg.dt_ms;
  sync.weight_scale = cfg.weight_scale;
  sync.target_convergence = cfg.sync_convergence;
  sync.Init(model_n, model_synapses.Size());

  auto t0 = std::chrono::high_resolution_clock::now();
  int n_steps = static_cast<int>(cfg.duration_ms / cfg.dt_ms);
  float ref_time = 0.0f;

  for (int step = 0; step < n_steps; ++step) {
    // Step reference brain independently
    ref_neurons.ClearSynapticInput();
    ref_synapses.PropagateSpikes(ref_neurons.spiked.data(),
                                  ref_neurons.i_syn.data(), cfg.weight_scale);
    IzhikevichStepHeterogeneous(ref_neurons, cfg.dt_ms, ref_time, ref_types);
    ref_time += cfg.dt_ms;

    // Step model with sync adaptation
    sync.Step(model_neurons, model_synapses, ref_neurons, model_types);

    // Early termination on convergence
    if (sync.HasConverged()) {
      Log(LogLevel::kInfo, "Converged at t=%.1fms (%.1f%% neurons synced)",
          sync.sim_time_ms, sync.Latest().fraction_converged * 100.0f);
      break;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  if (!sync.history.empty()) {
    const auto& final_snap = sync.Latest();
    Log(LogLevel::kInfo,
        "Sync complete: corr=%.3f  rmse=%.2f  converged=%.1f%%  (%.3fs)",
        final_snap.global_correlation, final_snap.global_rmse,
        final_snap.fraction_converged * 100.0f, elapsed);
  }

  if (cfg.checkpoint_path) {
    Checkpoint::Save(cfg.checkpoint_path, sync.sim_time_ms, sync.total_steps, 0,
                     model_neurons, model_synapses);
  }

  return 0;
}

int RunDirect(const Config& cfg) {
  Log(LogLevel::kInfo, "FlyWire Mind Couple v1.0 - Spiking network simulator");
  Log(LogLevel::kInfo, "dt=%.2fms  duration=%.0fms  mode=%s",
      cfg.dt_ms, cfg.duration_ms,
      cfg.bridge_mode == BridgeMode::kOpenLoop ? "open-loop" :
      cfg.bridge_mode == BridgeMode::kShadow ? "shadow" : "closed-loop");

  NeuronArray neurons;
  SynapseTable synapses;

  std::string neuron_path = std::string(cfg.data_dir) + "/neurons.bin";
  std::string synapse_path = std::string(cfg.data_dir) + "/synapses.bin";

  auto neuron_result = ConnectomeLoader::LoadNeurons(neuron_path, neurons);
  if (!neuron_result) {
    Log(LogLevel::kError, "%s", neuron_result.error().message.c_str());
    Log(LogLevel::kError, "Run: python3 scripts/import_connectome.py --test");
    return 1;
  }

  auto synapse_result = ConnectomeLoader::LoadSynapses(
      synapse_path, neurons.n, synapses);
  if (!synapse_result) {
    Log(LogLevel::kError, "%s", synapse_result.error().message.c_str());
    return 1;
  }

  Log(LogLevel::kInfo, "%zu neurons, %zu synapses (CSR)",
      neurons.n, synapses.Size());

  // Connectome validation and statistics
  if (cfg.stats) {
    ConnectomeStats stats;
    if (!stats.Compute(synapses, neurons)) {
      Log(LogLevel::kError, "Connectome validation failed, aborting");
      return 1;
    }
    stats.LogSummary();
  }

  TwinBridge bridge;
  bridge.digital = std::move(neurons);
  bridge.synapses = std::move(synapses);
  bridge.dt_ms = cfg.dt_ms;
  bridge.weight_scale = cfg.weight_scale;
  bridge.mode = cfg.bridge_mode;

  if (cfg.bridge_mode != BridgeMode::kOpenLoop) {
    bridge.read_channel = std::make_unique<SimulatedRead>();
    bridge.write_channel = std::make_unique<SimulatedWrite>();
    bridge.replacer.Init(bridge.digital.n);
  }

  // Resume from checkpoint if requested
  if (cfg.resume_path) {
    std::vector<uint8_t> ext;
    if (!Checkpoint::Load(cfg.resume_path,
                          bridge.sim_time_ms, bridge.total_steps,
                          bridge.total_resyncs,
                          bridge.digital, bridge.synapses, ext)) {
      return 1;
    }
    BridgeCheckpoint::Deserialize(ext, bridge.replacer, bridge.shadow);
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  int n_steps = static_cast<int>(cfg.duration_ms / cfg.dt_ms);
  STDPParams stdp_params;

  for (int step = 0; step < n_steps; ++step) {
    bridge.Step();

    if (cfg.stdp) {
      STDPUpdate(bridge.synapses, bridge.digital, bridge.sim_time_ms,
                 stdp_params);
    }

    if (step % cfg.metrics_interval == 0) {
      int spikes = bridge.digital.CountSpikes();
      if (cfg.bridge_mode != BridgeMode::kOpenLoop &&
          !bridge.shadow.history.empty()) {
        Log(LogLevel::kInfo, "t=%.1fms  spikes=%d  corr=%.3f",
            bridge.sim_time_ms, spikes,
            bridge.shadow.history.back().spike_correlation);
      } else {
        Log(LogLevel::kInfo, "t=%.1fms  spikes=%d",
            bridge.sim_time_ms, spikes);
      }
    }

    // Periodic checkpointing
    if (cfg.checkpoint_interval > 0 && cfg.checkpoint_path &&
        step > 0 && step % cfg.checkpoint_interval == 0) {
      auto ext = BridgeCheckpoint::Serialize(bridge.replacer, bridge.shadow);
      Checkpoint::Save(cfg.checkpoint_path,
                       bridge.sim_time_ms, bridge.total_steps,
                       bridge.total_resyncs,
                       bridge.digital, bridge.synapses, ext);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  double realtime_ratio = (cfg.duration_ms / 1000.0) / elapsed;

  Log(LogLevel::kInfo, "Done: %.1fms simulated in %.3fs (%.1fx real-time)",
      cfg.duration_ms, elapsed, realtime_ratio);

  // Final checkpoint
  if (cfg.checkpoint_path) {
    auto ext = BridgeCheckpoint::Serialize(bridge.replacer, bridge.shadow);
    Checkpoint::Save(cfg.checkpoint_path,
                     bridge.sim_time_ms, bridge.total_steps,
                     bridge.total_resyncs,
                     bridge.digital, bridge.synapses, ext);
  }

  return 0;
}

int RunConditioning(uint32_t seed) {
  ConditioningExperiment exp;
  exp.seed = seed;
  auto result = exp.Run();
  return result.learned() ? 0 : 1;
}

int RunMultiTrialAnalysis(int n_trials, uint32_t seed) {
  MultiTrialRunner runner;
  runner.n_trials = n_trials;
  runner.base_config.seed = seed;
  auto stats = runner.Run();
  return (stats.success_rate > 0.5f) ? 0 : 1;
}

int RunBridgeTest() {
  BridgeSelfTest test;
  auto result = test.Run();
  return result.passed() ? 0 : 1;
}

int Run(int argc, const char** argv) {
  Config cfg = ParseArgs(argc, argv);

  if (cfg.show_version) {
    Log(LogLevel::kInfo, "fwmc v{}", kVersionString);
    Log(LogLevel::kInfo, "{}", kProjectDescription);
    return 0;
  }

  if (cfg.show_help) {
    PrintUsage();
    return 0;
  }

  if (cfg.multi_trial > 0) {
    return RunMultiTrialAnalysis(cfg.multi_trial, cfg.seed);
  }
  if (cfg.conditioning) {
    return RunConditioning(cfg.seed);
  }
  if (cfg.bridge_test) {
    return RunBridgeTest();
  }
  if (cfg.experiment_config) {
    return RunExperiment(cfg.experiment_config);
  }
  if (cfg.parametric_brain && cfg.sync_ref) {
    return RunParametricSync(cfg);
  }
  if (cfg.parametric_brain) {
    return RunParametric(cfg);
  }

  // Default: resolve profile to a parametric brain spec.
  // If --data was explicitly passed, use RunDirect (binary connectome).
  // Otherwise, use the flywire profile as default.
  if (!cfg.explicit_data) {
    const char* profile_name = cfg.profile ? cfg.profile : "flywire";
    auto* prof = FindProfile(profile_name);
    if (!prof) {
      Log(LogLevel::kError, "Unknown profile: %s", profile_name);
      Log(LogLevel::kInfo, "Available profiles:");
      for (size_t i = 0; i < kNumProfiles; ++i) {
        Log(LogLevel::kInfo, "  %-10s  %s", kProfiles[i].name, kProfiles[i].description);
      }
      return 1;
    }
    Log(LogLevel::kInfo, "Profile: %s (%s)", prof->name, prof->description);
    Config pcfg = cfg;
    pcfg.parametric_brain = prof->brain_spec;
    return RunParametric(pcfg);
  }

  return RunDirect(cfg);
}

}  // namespace fwmc

int main(int argc, const char** argv) {
  return fwmc::Run(argc, argv);
}
