#include <charconv>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <unordered_map>

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
#include "core/spike_frequency_adaptation.h"
#include "core/temperature.h"
#include "core/gap_junctions.h"
#include "core/intrinsic_homeostasis.h"
#include "core/inhibitory_plasticity.h"
#include "core/neuromodulator_effects.h"
#include "core/nmda.h"
#include "core/calcium_plasticity.h"
#include "tissue/lod_transition.h"
#include "tissue/neuromodulator_field.h"
#include "bridge/twin_bridge.h"
#include "bridge/bridge_checkpoint.h"
#include "experiment_runner.h"
#include "experiments/01_conditioning.h"
#include "experiments/02_visual_escape.h"
#include "experiments/03_navigation.h"
#include "experiments/04_whisker.h"
#include "experiments/05_prey_capture.h"
#include "experiments/06_courtship.h"
#include "experiments/07_twinning.h"
#include "experiments/08_ablation.h"
#include "experiments/09_compensated_ablation.h"
#include "multi_trial.h"
#include "bridge_self_test.h"
#include "optogenetics/optogenetics.h"
#include "optogenetics/optimizer.h"
#include "optogenetics/opto_io.h"

namespace fwmc {
using namespace mechabrain;

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
  {"mouse",    "examples/mouse_cortical_column.brain",    "Mouse barrel cortex column (2380 neurons)"},
  {"human",    "examples/human_cortical_column.brain",    "Human cortical column with thalamocortical loop (10550 neurons)"},
  {"zebrafish","examples/zebrafish_optic_tectum.brain",   "Zebrafish larval optic tectum (5000 neurons)"},
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
  bool visual_escape = false;           // --visual-escape: run looming escape experiment
  bool escape_optimize = false;          // --escape-optimize: auto-tune escape params
  bool navigation = false;              // --navigation: run CX heading experiment
  bool nav_optimize = false;             // --nav-optimize: auto-tune navigation params
  bool whisker = false;                 // --whisker: run mouse barrel cortex experiment
  bool prey_capture = false;             // --prey-capture: run zebrafish prey capture experiment
  bool courtship = false;                // --courtship: run courtship song experiment
  bool twinning = false;                 // --twinning: run full neural twinning experiment
  bool ablation = false;                  // --ablation: run progressive ablation study
  bool compensated = false;               // --compensated: run compensated ablation (prosthesis demo)
  bool test_all = false;                 // --test-all: run all experiments as regression suite
  uint32_t seed = 42;                  // --seed: deterministic random seed
  int multi_trial = 0;                // --multi-trial N: run N conditioning trials
  bool explicit_data = false;          // true if --data was explicitly passed
  bool sfa = true;                    // spike-frequency adaptation (default on)
  float temperature = 25.0f;         // simulation temperature in Celsius
  float ref_temperature = 22.0f;     // reference temperature for Q10 scaling
  bool multiscale = false;           // --multiscale: LOD transition engine mode
  float grid_spacing = 10.0f;       // --grid-spacing: voxel size in um (default 10)
  float focus_x = 250.0f;           // --focus: LOD focus point (x,y,z in um)
  float focus_y = 150.0f;
  float focus_z = 100.0f;
  bool gap_junctions = true;        // gap junctions (default on; --no-gap-junctions)
  bool homeostasis = true;          // intrinsic homeostasis (default on; --no-homeostasis)
  bool eligibility_traces = false;  // three-factor learning (--eligibility-traces)
  bool opto = false;                // --opto: run optogenetics experiment
  bool opto_optimize = false;       // --opto-optimize: run optogenetics parameter search
  const char* opto_config = nullptr;  // --opto-config: load config from file
  const char* opto_output = nullptr;  // --opto-output: write results to file
  float opto_power = 10.0f;         // --opto-power: laser power in mW
  float opto_fraction = 0.2f;       // --opto-fraction: target expression fraction
  const char* opto_region = nullptr; // --opto-region: target region name
  int opto_trials = 50;            // --opto-trials: optimization budget
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
      "  --visual-escape       Run visual looming escape experiment\n"
      "  --navigation          Run CX heading/navigation experiment\n"
      "  --nav-optimize        Auto-tune navigation circuit parameters\n"
      "  --whisker             Run mouse barrel cortex whisker experiment\n"
      "  --prey-capture        Run zebrafish prey capture experiment\n"
      "  --courtship           Run courtship song generation experiment\n"
      "  --twinning            Run full neural twinning experiment\n"
      "  --test-all            Run all experiments as regression test suite\n"
      "  --seed N              Random seed for reproducibility (default: 42)\n"
      "  --no-sfa              Disable spike-frequency adaptation\n"
      "  --temperature C       Simulation temperature in Celsius (default: 25)\n"
      "  --ref-temperature C   Reference temperature for Q10 scaling (default: 22)\n"
      "  --multiscale          Run multiscale LOD simulation (field + spiking)\n"
      "  --grid-spacing UM     Voxel size for multiscale grid (default: 10)\n"
      "  --focus X Y Z         LOD focus point in um (default: 250 150 100)\n"
      "  --opto                Run optogenetics experiment\n"
      "  --opto-optimize       Run optogenetics parameter optimization\n"
      "  --opto-config FILE    Load optogenetics config from file\n"
      "  --opto-output FILE    Write results to file (default: stdout)\n"
      "  --opto-power MW       Laser power in mW (default: 10)\n"
      "  --opto-fraction F     Target expression fraction (default: 0.2)\n"
      "  --opto-region NAME    Target region name\n"
      "  --opto-trials N       Optimization trial budget (default: 50)\n"
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
    else if (arg == "--visual-escape")
      cfg.visual_escape = true;
    else if (arg == "--escape-optimize")
      cfg.escape_optimize = true;
    else if (arg == "--navigation")
      cfg.navigation = true;
    else if (arg == "--nav-optimize")
      cfg.nav_optimize = true;
    else if (arg == "--whisker")
      cfg.whisker = true;
    else if (arg == "--prey-capture")
      cfg.prey_capture = true;
    else if (arg == "--courtship")
      cfg.courtship = true;
    else if (arg == "--twinning")
      cfg.twinning = true;
    else if (arg == "--ablation")
      cfg.ablation = true;
    else if (arg == "--compensated")
      cfg.compensated = true;
    else if (arg == "--test-all")
      cfg.test_all = true;
    else if (arg == "--seed" && i + 1 < argc)
      cfg.seed = static_cast<uint32_t>(ParseInt(argv[++i]));
    else if (arg == "--no-sfa")
      cfg.sfa = false;
    else if (arg == "--temperature" && i + 1 < argc)
      cfg.temperature = ParseFloat(argv[++i]);
    else if (arg == "--ref-temperature" && i + 1 < argc)
      cfg.ref_temperature = ParseFloat(argv[++i]);
    else if (arg == "--multiscale")
      cfg.multiscale = true;
    else if (arg == "--grid-spacing" && i + 1 < argc)
      cfg.grid_spacing = ParseFloat(argv[++i]);
    else if (arg == "--focus" && i + 3 < argc) {
      cfg.focus_x = ParseFloat(argv[++i]);
      cfg.focus_y = ParseFloat(argv[++i]);
      cfg.focus_z = ParseFloat(argv[++i]);
    }
    else if (arg == "--opto")
      cfg.opto = true;
    else if (arg == "--opto-optimize")
      cfg.opto_optimize = true;
    else if (arg == "--opto-config" && i + 1 < argc)
      cfg.opto_config = argv[++i];
    else if (arg == "--opto-output" && i + 1 < argc)
      cfg.opto_output = argv[++i];
    else if (arg == "--opto-power" && i + 1 < argc)
      cfg.opto_power = ParseFloat(argv[++i]);
    else if (arg == "--opto-fraction" && i + 1 < argc)
      cfg.opto_fraction = ParseFloat(argv[++i]);
    else if (arg == "--opto-region" && i + 1 < argc)
      cfg.opto_region = argv[++i];
    else if (arg == "--opto-trials" && i + 1 < argc)
      cfg.opto_trials = ParseInt(argv[++i]);
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
  auto species_defaults = spec.GetDefaults();
  Log(LogLevel::kInfo, "Brain spec: %s (%zu regions, %zu projections, species=%s)",
      spec.name.c_str(), spec.regions.size(), spec.projections.size(),
      SpeciesName(spec.species));

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);
  synapses.AssignPerNeuronTau(neurons);  // per-NT synaptic time constants

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
  std::vector<ResolvedStimulus> resolved_stimuli;
  if (has_stimuli) {
    Log(LogLevel::kInfo, "%zu stimuli defined", spec.stimuli.size());
    resolved_stimuli = ResolveStimuli(spec.stimuli, gen.region_ranges);
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

  // Initialize short-term plasticity from species defaults.
  if (cfg.stp) {
    STPParams stp_params{species_defaults.stp_U_se,
                         species_defaults.stp_tau_d,
                         species_defaults.stp_tau_f};
    synapses.InitSTP(stp_params);
    Log(LogLevel::kInfo, "STP: U=%.2f, tau_d=%.0fms, tau_f=%.0fms",
        stp_params.U_se, stp_params.tau_d, stp_params.tau_f);
  }

  // Spike-frequency adaptation (calcium-based sAHP)
  SpikeFrequencyAdaptation sfa;
  if (cfg.sfa) {
    sfa.Init(neurons.n);
    Log(LogLevel::kInfo, "SFA enabled (tau_ca=%.0fms, g_sahp=%.2f)",
        sfa.tau_calcium_ms, sfa.g_sahp);
  }

  // Temperature-dependent rate scaling (Q10 model).
  // Use species-appropriate reference temperature unless overridden by CLI.
  TemperatureModel temperature;
  temperature.reference_temp_c = cfg.ref_temperature;
  temperature.current_temp_c = cfg.temperature;
  // Auto-select species ref temp if user hasn't overridden (22C = Drosophila default)
  if (cfg.ref_temperature == 22.0f && species_defaults.ref_temperature_c != 22.0f) {
    temperature.reference_temp_c = species_defaults.ref_temperature_c;
  }
  temperature.enabled = (temperature.current_temp_c != temperature.reference_temp_c);
  if (temperature.enabled) {
    Log(LogLevel::kInfo, "Temperature scaling: %.1fC (ref %.1fC), channel=%.2fx synapse=%.2fx",
        temperature.current_temp_c, temperature.reference_temp_c,
        temperature.ChannelScale(), temperature.SynapseScale());
  }

  // Distance-dependent axonal conduction delays.
  // Velocity from species defaults (um/ms = m/s * 1000).
  float conduction_v = species_defaults.conduction_velocity_m_s * 1000.0f;
  synapses.InitDistanceDelay(neurons, conduction_v, cfg.dt_ms);
  if (synapses.HasDelays()) {
    Log(LogLevel::kInfo, "Conduction delays: ring_size=%zu, dt=%.2fms",
        synapses.ring_size, cfg.dt_ms);
  }

  // NMDA receptor dynamics: slow excitatory conductance with Mg2+ block.
  // Provides coincidence detection for Hebbian associative learning.
  NMDAReceptor nmda;
  nmda.Init(neurons.n);
  Log(LogLevel::kInfo, "NMDA receptors enabled (tau=%.0fms, Mg=%.1fmM, gain=%.2f)",
      nmda.tau_nmda_ms, nmda.mg_conc_mM, nmda.nmda_gain);

  // Calcium-dependent plasticity: NMDA Ca2+ -> LTP/LTD via Omega function
  CalciumPlasticity cadp;
  cadp.Init(neurons.n);
  Log(LogLevel::kInfo, "Calcium plasticity enabled (theta_d=%.3f, theta_p=%.3f, BCM=%s)",
      cadp.params.theta_d, cadp.params.theta_p,
      cadp.params.bcm_sliding ? "on" : "off");

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
  stdp_params.a_plus = species_defaults.stdp_a_plus;
  stdp_params.a_minus = species_defaults.stdp_a_minus;
  stdp_params.tau_plus = species_defaults.stdp_tau_plus;
  stdp_params.tau_minus = species_defaults.stdp_tau_minus;
  SynapticScaling synaptic_scaling;
  synaptic_scaling.Init(neurons.n);
  int scaling_step_count = 0;
  constexpr int kScalingSteps = 1000;  // apply every 1000 steps

  // Inhibitory plasticity: Vogels iSTDP for E/I balance
  InhibitorySTDP istdp;
  if (cfg.stdp) {
    istdp.Init(synapses.post.size(), neurons.n);
  }

  // Neuromodulatory excitability effects
  NeuromodulatorEffects neuromod_effects;

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
      ApplyResolvedStimuli(resolved_stimuli, neurons, sim_time);
    }

    // Deliver delayed synaptic currents from previous timesteps
    synapses.DeliverDelayed(neurons.i_syn.data());

    // Exponential synaptic current decay (realistic temporal integration).
    // New spikes add on top of decaying residual from previous timesteps.
    // Base tau from species defaults, scaled by temperature if enabled.
    float eff_tau = temperature.ScaledTauSyn(species_defaults.tau_syn_excitatory);
    neurons.DecaySynapticInput(cfg.dt_ms, eff_tau);
    if (synapses.HasStochasticRelease()) {
      synapses.PropagateSpikesMonteCarlo(neurons.spiked.data(),
                                          neurons.i_syn.data(),
                                          cfg.weight_scale, release_rng);
    } else {
      synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(),
                               cfg.weight_scale);
    }
    synapses.AdvanceDelayRing();
    if (synapses.HasSTP()) {
      synapses.RecoverSTP(cfg.dt_ms);
    }

    // NMDA: accumulate slow excitatory conductance from ACh spikes, then
    // apply voltage-dependent Mg2+ block and inject current into neurons.
    // Must run after PropagateSpikes (needs spiked[]) and before Izhikevich
    // (so NMDA current affects membrane potential this timestep).
    nmda.AccumulateFromSpikes(synapses, neurons.spiked.data(), cfg.weight_scale);
    nmda.Step(neurons, cfg.dt_ms);

    IzhikevichStepHeterogeneousFast(neurons, cfg.dt_ms, sim_time, types);

    // Spike-frequency adaptation: slow calcium-gated K+ current
    if (cfg.sfa) {
      sfa.Update(neurons, cfg.dt_ms);
    }

    // Neuromodulator dynamics: DA/5HT/OA release, diffusion, decay
    NeuromodulatorUpdate(neurons, synapses, cfg.dt_ms);

    // Neuromodulatory excitability: DA/5HT/OA modulate membrane currents
    neuromod_effects.Apply(neurons);

    // Calcium-dependent plasticity: Omega(Ca_NMDA) -> LTP/LTD
    // Must run after NMDA Step (ca_nmda current) and IzhikevichStep (spiked[] set)
    if (nmda.initialized && cadp.initialized) {
      cadp.UpdateBCM(neurons, cfg.dt_ms);
      cadp.Apply(synapses, neurons, nmda.ca_nmda.data());
    }

    if (cfg.stdp) {
      STDPUpdate(synapses, neurons, sim_time, stdp_params);

      // Inhibitory STDP: Vogels rule for E/I balance
      if (istdp.IsInitialized()) {
        InhibitorySTDPUpdate(synapses, neurons, cfg.dt_ms, istdp);
      }

      // Three-factor learning: convert eligibility traces to weight changes
      if (stdp_params.use_eligibility_traces && stdp_params.dopamine_gated) {
        EligibilityTraceUpdate(synapses, neurons, cfg.dt_ms, stdp_params);
      }

      // Synaptic scaling: homeostatic weight normalization
      synaptic_scaling.AccumulateSpikes(neurons, cfg.dt_ms);
      scaling_step_count++;
      if (scaling_step_count >= kScalingSteps) {
        synaptic_scaling.Apply(synapses, stdp_params);
        scaling_step_count = 0;
      }
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
    ref_neurons.DecaySynapticInput(cfg.dt_ms, 3.0f);
    ref_synapses.PropagateSpikes(ref_neurons.spiked.data(),
                                  ref_neurons.i_syn.data(), cfg.weight_scale);
    IzhikevichStepHeterogeneousFast(ref_neurons, cfg.dt_ms, ref_time, ref_types);
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

int RunMultiscale(const Config& cfg) {
  Log(LogLevel::kInfo, "FlyWire Mind Couple v1.0 - Multiscale brain simulation");
  Log(LogLevel::kInfo, "dt=%.2fms  duration=%.0fms  grid=%.0fum  focus=(%.0f,%.0f,%.0f)",
      cfg.dt_ms, cfg.duration_ms, cfg.grid_spacing,
      cfg.focus_x, cfg.focus_y, cfg.focus_z);

  // Build multiscale engine with Drosophila brain anatomy.
  // Neural field (Wilson-Cowan) runs everywhere as a global backbone.
  // Spiking neurons (Izhikevich) spawn dynamically near the focus point.
  LODTransitionEngine engine;
  engine.brain_sdf.InitDrosophila();

  // Configure LOD zones: 4-level hierarchy.
  // Drosophila brain is ~500x300x200 um. Zones define concentric shells
  // around the focus point, each at a different resolution level.
  // LOD 3 (compartmental): closest to focus, full biophysical detail
  // LOD 2 (spiking): individual Izhikevich neurons with CSR synapses
  // LOD 1 (population): MPR exact mean-field, 6 ODEs per region
  // LOD 0 (field): Wilson-Cowan continuum, cheapest
  engine.lod.zones = {
    {30.0f,  LODLevel::kCompartmental},  // biophysical detail within 30um
    {100.0f, LODLevel::kNeuron},         // spiking neurons within 100um
    {200.0f, LODLevel::kRegion},         // population mean-field within 200um
  };
  engine.lod.default_level = LODLevel::kContinuum;
  engine.lod.SetFocus(cfg.focus_x, cfg.focus_y, cfg.focus_z);

  // Engine configuration
  engine.config.neurons_per_voxel = 10.0f;
  engine.config.synapse_density = 0.05f;
  engine.config.boundary_width_um = 20.0f;
  engine.config.field_to_neuron_scale = 10.0f;
  engine.config.neuron_to_field_scale = 0.01f;
  engine.config.weight_scale = cfg.weight_scale;
  engine.config.seed = cfg.seed;

  // --- Inter-region projections from Drosophila connectomics ---
  // Maps brain spec projections to SDF bilateral primitive names.
  // When both endpoints escalate to LOD 2+, cross-chunk synapses
  // are instantiated automatically by the LODTransitionEngine.
  auto AddBilateral = [&](const char* from_base, const char* to_base,
                          float density, uint8_t nt,
                          float w_mean, float w_std) {
    for (const char* side : {"_L", "_R"}) {
      LODProjection p;
      p.from_region = std::string(from_base) + side;
      p.to_region = std::string(to_base) + side;
      p.density = density;
      p.nt_type = nt;
      p.weight_mean = w_mean;
      p.weight_std = w_std;
      engine.projections.push_back(p);
    }
  };

  auto AddToMidline = [&](const char* from_base, const char* to,
                          float density, uint8_t nt,
                          float w_mean, float w_std) {
    for (const char* side : {"_L", "_R"}) {
      LODProjection p;
      p.from_region = std::string(from_base) + side;
      p.to_region = to;
      p.density = density;
      p.nt_type = nt;
      p.weight_mean = w_mean;
      p.weight_std = w_std;
      engine.projections.push_back(p);
    }
  };

  auto AddProjection = [&](const char* from, const char* to,
                           float density, uint8_t nt,
                           float w_mean, float w_std) {
    LODProjection p;
    p.from_region = from;
    p.to_region = to;
    p.density = density;
    p.nt_type = nt;
    p.weight_mean = w_mean;
    p.weight_std = w_std;
    engine.projections.push_back(p);
  };

  // AL -> MB calyx: PN axons to KC dendrites (Caron et al. 2013)
  AddBilateral("antennal_lobe", "mb_calyx", 0.003f, kACh, 1.5f, 0.4f);
  // AL -> LH: direct innate pathway (Jefferis et al. 2007)
  AddBilateral("antennal_lobe", "lateral_horn", 0.015f, kACh, 1.3f, 0.3f);
  // MB lobe -> LH: MBON output modulates innate behavior (Aso et al. 2014)
  AddBilateral("mb_lobe", "lateral_horn", 0.001f, kACh, 1.0f, 0.2f);
  // MB lobe -> CX: learned behavior influences motor planning
  AddToMidline("mb_lobe", "central_complex", 0.0005f, kACh, 0.8f, 0.2f);
  // LH -> CX: innate olfactory drives navigation
  AddToMidline("lateral_horn", "central_complex", 0.005f, kACh, 1.0f, 0.3f);
  // OL -> CX: visual input to central complex (landmark navigation)
  AddToMidline("optic_lobe", "central_complex", 0.0003f, kACh, 1.2f, 0.3f);
  // CX -> SEZ: descending motor commands (Namiki et al. 2018)
  AddProjection("central_complex", "sez", 0.002f, kACh, 1.5f, 0.4f);
  // LH -> SEZ: direct innate motor responses
  AddToMidline("lateral_horn", "sez", 0.002f, kACh, 1.2f, 0.3f);
  // Central brain -> CX: descending modulation
  AddProjection("central_brain", "central_complex", 0.003f, kACh, 1.0f, 0.3f);
  // Central brain -> SEZ: descending neurons
  AddProjection("central_brain", "sez", 0.002f, kACh, 1.3f, 0.4f);
  // SEZ -> central brain: ascending sensory feedback
  AddProjection("sez", "central_brain", 0.0005f, kACh, 0.8f, 0.2f);
  // MB calyx -> lobe: internal MB axonal projection (KC pedunculus)
  AddBilateral("mb_calyx", "mb_lobe", 0.01f, kACh, 1.2f, 0.3f);

  Log(LogLevel::kInfo, "Defined %zu inter-region projections",
      engine.projections.size());

  engine.Init(cfg.grid_spacing);

  Log(LogLevel::kInfo, "Grid: %ux%ux%u voxels (%.0fum spacing), %zu regions",
      engine.grid.nx, engine.grid.ny, engine.grid.nz,
      cfg.grid_spacing, engine.brain_sdf.primitives.size());

  // Per-chunk biological subsystem state, indexed by region_idx.
  // Created when a chunk escalates, destroyed when it de-escalates.
  std::unordered_map<uint32_t, SpikeFrequencyAdaptation> chunk_sfa;
  NeuromodulatorEffects neuromod_effects;

  // Background noise: tonic drive + variability (Drosophila calibrated)
  std::mt19937 noise_rng(cfg.seed + 99);
  std::normal_distribution<float> noise_dist(8.0f, 3.0f);

  // Temperature scaling for synaptic time constants
  TemperatureModel temperature;
  temperature.reference_temp_c = cfg.ref_temperature;
  temperature.current_temp_c = cfg.temperature;
  temperature.enabled = (cfg.temperature != cfg.ref_temperature);
  float eff_tau = temperature.ScaledTauSyn(3.0f);

  // STDP state (when enabled)
  STDPParams stdp_params;

  // Run simulation with biological subsystems on escalated chunks.
  // Manually orchestrates the LODTransitionEngine phases so that
  // SFA, neuromodulation, synaptic decay, and background noise
  // are applied per-chunk between the field and neuron steps.
  auto t0 = std::chrono::high_resolution_clock::now();
  int n_steps = static_cast<int>(cfg.duration_ms / cfg.dt_ms);
  float sim_time = 0.0f;

  for (int step = 0; step < n_steps; ++step) {
    // --- Phase 1: LOD transitions ---
    int transitions = engine.UpdateLOD();

    // Initialize SFA for newly escalated chunks
    for (auto& chunk : engine.chunks) {
      if (chunk_sfa.find(chunk.region_idx) == chunk_sfa.end()) {
        SpikeFrequencyAdaptation sfa;
        sfa.Init(chunk.neurons.n);
        chunk_sfa[chunk.region_idx] = std::move(sfa);
      }
    }
    // Clean up SFA state for de-escalated regions
    for (auto it = chunk_sfa.begin(); it != chunk_sfa.end(); ) {
      if (!engine.IsRegionActive(it->first))
        it = chunk_sfa.erase(it);
      else
        ++it;
    }

    // --- Phase 2: Neural field backbone (Wilson-Cowan on all voxels) ---
    engine.field.Step(engine.grid, cfg.dt_ms);

    // --- Phase 3: Boundary coupling (field -> spiking neurons) ---
    engine.CoupleFieldToNeurons();

    // --- Phase 4: Per-chunk synaptic processing ---
    for (auto& chunk : engine.chunks) {
      // Background noise (tonic drive + variability)
      for (size_t i = 0; i < chunk.neurons.n; ++i) {
        chunk.neurons.i_ext[i] += noise_dist(noise_rng);
      }

      // Exponential synaptic current decay (temperature-scaled).
      // Residual from previous step decays, new spikes add on top,
      // producing alpha-function-like postsynaptic currents.
      chunk.neurons.DecaySynapticInput(cfg.dt_ms, eff_tau);

      // Intra-chunk spike propagation
      chunk.synapses.PropagateSpikes(
          chunk.neurons.spiked.data(),
          chunk.neurons.i_syn.data(),
          cfg.weight_scale);

      // STP recovery (Tsodyks-Markram facilitation/depression)
      if (chunk.synapses.HasSTP()) {
        chunk.synapses.RecoverSTP(cfg.dt_ms);
      }
    }

    // Compartmental chunks: basic synaptic processing
    for (auto& comp : engine.comp_chunks) {
      comp.neurons.ClearSynapticInput();
      comp.synapses.PropagateSpikes(
          comp.neurons.spiked.data(),
          comp.neurons.i_syn_soma.data(),
          cfg.weight_scale);
    }

    // Cross-chunk spike propagation (inter-region axonal projections)
    engine.PropagateCrossChunkSpikes(cfg.weight_scale);

    // --- Phase 5: Neuron dynamics + biological subsystems ---
    for (auto& chunk : engine.chunks) {
      // Izhikevich spiking dynamics
      IzhikevichStep(chunk.neurons, cfg.dt_ms, sim_time, chunk.params);

      // Spike-frequency adaptation: slow calcium-gated K+ current
      auto sfa_it = chunk_sfa.find(chunk.region_idx);
      if (sfa_it != chunk_sfa.end()) {
        sfa_it->second.Update(chunk.neurons, cfg.dt_ms);
      }

      // Neuromodulator dynamics: DA/5HT/OA release, diffusion, decay
      NeuromodulatorUpdate(chunk.neurons, chunk.synapses, cfg.dt_ms);

      // Neuromodulatory excitability: DA/5HT/OA modulate membrane currents
      neuromod_effects.Apply(chunk.neurons);

      // STDP: spike-timing-dependent plasticity (if enabled)
      if (cfg.stdp) {
        STDPUpdate(chunk.synapses, chunk.neurons, sim_time, stdp_params);
      }

      chunk.AccumulateSpikes();
    }

    // Step compartmental chunks
    for (auto& comp : engine.comp_chunks) {
      CompartmentalStep(comp.neurons, cfg.dt_ms, sim_time, comp.params);
      comp.AccumulateSpikes();
    }

    // --- Phase 6: Neuron -> field coupling ---
    engine.CoupleNeuronsToField(cfg.dt_ms);

    // Clear external input for next step
    for (auto& chunk : engine.chunks) chunk.neurons.ClearExternalInput();
    for (auto& comp : engine.comp_chunks) comp.neurons.ClearExternalInput();

    sim_time += cfg.dt_ms;

    if (transitions > 0) {
      Log(LogLevel::kInfo,
          "t=%.1fms  LOD transitions: %d  chunks=%zu  neurons=%zu  cross_links=%zu",
          sim_time, transitions, engine.ActiveChunks(),
          engine.TotalActiveNeurons(), engine.ActiveCrossLinks());
    }

    if (step % cfg.metrics_interval == 0) {
      int spikes = engine.TotalSpikes();
      Log(LogLevel::kInfo,
          "t=%.1fms  spikes=%d  chunks=%zu  neurons=%zu  cross_links=%zu (%zu synapses)",
          sim_time, spikes, engine.ActiveChunks(),
          engine.TotalActiveNeurons(),
          engine.ActiveCrossLinks(), engine.TotalCrossChunkSynapses());

      for (const auto& chunk : engine.chunks) {
        int chunk_spikes = chunk.CountSpikes();
        Log(LogLevel::kInfo,
            "  chunk '%s': %zu neurons, %zu synapses, %d spikes, %zu boundary",
            chunk.name.c_str(), chunk.neurons.n, chunk.synapses.Size(),
            chunk_spikes, chunk.boundary_indices.size());
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  double realtime_ratio = (cfg.duration_ms / 1000.0) / elapsed;

  // Sample field activity at a few key points
  Log(LogLevel::kInfo, "Field activity at key locations:");
  const char* probe_names[] = {"central_brain", "optic_lobe_L", "antennal_lobe_L", "mb_calyx_L"};
  float probe_x[] = {250, 80, 200, 180};
  float probe_y[] = {150, 150, 90, 200};
  float probe_z[] = {100, 100, 60, 120};
  for (int i = 0; i < 4; ++i) {
    float E = engine.ReadActivity(probe_x[i], probe_y[i], probe_z[i]);
    Log(LogLevel::kInfo, "  %s: E=%.3f", probe_names[i], E);
  }

  Log(LogLevel::kInfo,
      "Done: %.1fms simulated in %.3fs (%.1fx real-time), %zu final chunks, %zu final neurons, %zu cross-links",
      cfg.duration_ms, elapsed, realtime_ratio,
      engine.ActiveChunks(), engine.TotalActiveNeurons(),
      engine.ActiveCrossLinks());

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
  synapses.AssignPerNeuronTau(neurons);  // per-NT synaptic time constants

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

int RunOpto(const Config& cfg) {
  OptogeneticsConfig opto_cfg;

  // Load from file or build from CLI args
  if (cfg.opto_config) {
    if (!ReadOptoConfig(cfg.opto_config, opto_cfg)) {
      Log(LogLevel::kError, "Failed to load opto config: %s", cfg.opto_config);
      return 1;
    }
  }

  // CLI overrides
  if (cfg.parametric_brain) opto_cfg.brain_spec_path = cfg.parametric_brain;
  opto_cfg.laser_power_mw = cfg.opto_power;
  opto_cfg.target_fraction = cfg.opto_fraction;
  if (cfg.opto_region) opto_cfg.target_region = cfg.opto_region;
  opto_cfg.seed = cfg.seed;
  opto_cfg.dt_ms = cfg.dt_ms;
  if (cfg.duration_ms != 1000.0f) {
    // User specified duration: override protocol timing
    opto_cfg.baseline_ms = cfg.duration_ms * 0.2f;
    opto_cfg.post_stim_ms = cfg.duration_ms * 0.2f;
  }
  opto_cfg.enable_stdp = cfg.stdp;

  OptogeneticsExperiment experiment;
  auto result = experiment.Run(opto_cfg);

  // Output results
  if (cfg.opto_output) {
    WriteOptoResult(cfg.opto_output, result);
    Log(LogLevel::kInfo, "Results written to %s", cfg.opto_output);
  }

  Log(LogLevel::kInfo, "Modulation index: %.4f (baseline=%.1f Hz, evoked=%.1f Hz)",
      result.modulation_index, result.baseline_rate_hz, result.evoked_rate_hz);

  return result.has_effect() ? 0 : 1;
}

int RunOptoOptimize(const Config& cfg) {
  OptogeneticsOptimizer optimizer;
  optimizer.param_space = OptogeneticsOptimizer::DefaultParamSpace();

  // Base config from CLI
  if (cfg.parametric_brain) optimizer.base_config.brain_spec_path = cfg.parametric_brain;
  if (cfg.opto_region) optimizer.base_config.target_region = cfg.opto_region;
  optimizer.base_config.seed = cfg.seed;
  optimizer.base_config.dt_ms = cfg.dt_ms;
  optimizer.base_config.enable_stdp = cfg.stdp;

  OptimizerConfig opt_cfg;
  opt_cfg.max_trials = cfg.opto_trials;
  opt_cfg.seed = cfg.seed;
  opt_cfg.verbose = true;

  auto result = optimizer.Optimize(OptoObjectives::MaxExcitation, opt_cfg);

  if (cfg.opto_output) {
    WriteOptimizerResult(cfg.opto_output, optimizer.param_space, result);
    Log(LogLevel::kInfo, "Optimizer results written to %s", cfg.opto_output);
  }

  Log(LogLevel::kInfo, "Best score: %.4f (%d trials, %.1fs)",
      result.best.score, result.n_trials, result.total_seconds);

  return result.converged() ? 0 : 1;
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

  if (cfg.opto_optimize) {
    return RunOptoOptimize(cfg);
  }
  if (cfg.opto) {
    return RunOpto(cfg);
  }
  if (cfg.multi_trial > 0) {
    return RunMultiTrialAnalysis(cfg.multi_trial, cfg.seed);
  }
  // ---- Regression test suite: run ALL experiments, report pass/fail ----
  if (cfg.test_all) {
    Log(LogLevel::kInfo, "=== Regression Test Suite ===");
    auto t_suite_start = std::chrono::steady_clock::now();
    int n_pass = 0, n_fail = 0;

    auto report = [&](const char* name, bool passed, const char* detail) {
      if (passed) {
        n_pass++;
        Log(LogLevel::kInfo, "  PASS  %-25s %s", name, detail);
      } else {
        n_fail++;
        Log(LogLevel::kInfo, "  FAIL  %-25s %s", name, detail);
      }
    };

    // 01 - Olfactory conditioning
    {
      ConditioningExperiment exp;
      exp.seed = cfg.seed;
      auto r = exp.Run();
      char buf[128];
      snprintf(buf, sizeof(buf), "LI=%.2f, DI=%.2f", r.learning_index, r.discrimination_index);
      report("01_conditioning", r.learned(), buf);
    }

    // 02 - Visual looming escape
    {
      VisualEscapeExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "latency=%.1fms, angle=%.1fdeg", r.escape_latency_ms, r.angle_at_escape_deg);
      report("02_visual_escape", r.timing_plausible(), buf);
    }

    // 03 - CX navigation
    {
      NavigationExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "bump=%.2f, err=%.1fdeg", r.bump_amplitude, r.heading_error_deg);
      report("03_navigation", r.heading_plausible(), buf);
    }

    // 04 - Mouse whisker
    {
      WhiskerExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "L4=%.1fms, L23/L4=%.2f", r.l4_first_spike_ms, r.l23_l4_ratio);
      report("04_whisker", r.responded(), buf);
    }

    // 05 - Zebrafish prey capture
    {
      PreyCaptureExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "tectal=%.1fms, motor=%.1fms", r.tectal_onset_ms, r.capture_latency_ms);
      report("05_prey_capture", r.responded(), buf);
    }

    // 06 - Courtship song
    {
      CourtshipExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "pulses=%d, IPI=%.1fms", r.n_pulses, r.mean_ipi_ms);
      report("06_courtship", r.produced_pulses(), buf);
    }

    // 07 - Full twinning
    {
      TwinningExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "continuity=%.2f, replaced=%.0f%%",
               r.behavioral_continuity, r.replacement_fraction * 100.0f);
      report("07_twinning", r.passed(), buf);
    }

    // 08 - Ablation study
    {
      AblationExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "graceful=%.2f, half_life=%.0f%%",
               r.graceful_score, r.half_life_fraction * 100.0f);
      report("08_ablation", r.passed(), buf);
    }

    // 09 - Compensated ablation (prosthesis demo)
    {
      CompensatedAblationExperiment exp;
      auto r = exp.Run(cfg.seed);
      char buf[128];
      snprintf(buf, sizeof(buf), "benefit=%.2f, ext=%.1fx",
               r.compensation_benefit, r.lifetime_extension);
      report("09_compensated", r.passed(), buf);
    }

    double suite_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_suite_start).count();
    Log(LogLevel::kInfo, "=== %d passed, %d failed (%.2fs) ===",
        n_pass, n_fail, suite_sec);
    return (n_fail == 0) ? 0 : 1;
  }

  if (cfg.conditioning) {
    return RunConditioning(cfg.seed);
  }
  if (cfg.bridge_test) {
    return RunBridgeTest();
  }
  if (cfg.escape_optimize) {
    VisualEscapeOptimizer opt;
    opt.seed = cfg.seed;
    opt.n_iterations = 300;
    auto best = opt.Run();
    Log(LogLevel::kInfo, "Escape-optimize: latency=%.1fms, angle=%.1fdeg",
        best.escape.escape_latency_ms, best.escape.angle_at_escape_deg);
    return best.escape.timing_plausible() ? 0 : 1;
  }
  if (cfg.visual_escape) {
    VisualEscapeExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Visual escape: escaped=%d, latency=%.1fms, angle=%.1fdeg",
        result.escaped(), result.escape_latency_ms, result.angle_at_escape_deg);
    return result.escaped() ? 0 : 1;
  }
  if (cfg.nav_optimize) {
    NavigationOptimizer opt;
    opt.seed = cfg.seed;
    opt.n_iterations = 300;
    auto best = opt.Run();
    Log(LogLevel::kInfo, "Nav-optimize: best err=%.1fdeg, R2=%.3f, bump=%.1f",
        best.nav.heading_error_deg, best.nav.rotation_tracking_r2,
        best.nav.bump_amplitude);
    return best.nav.bump_present() ? 0 : 1;
  }
  if (cfg.navigation) {
    NavigationExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Navigation: bump=%.1f, heading_err=%.1fdeg, dark_err=%.1fdeg",
        result.bump_amplitude, result.heading_error_deg, result.heading_error_dark_deg);
    return result.bump_present() ? 0 : 1;
  }
  if (cfg.whisker) {
    WhiskerExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Whisker: L4=%.1fms, L23=%.1fms, L5=%.1fms, L23/L4=%.2f",
        result.l4_first_spike_ms, result.l23_first_spike_ms,
        result.l5_first_spike_ms, result.l23_l4_ratio);
    return result.responded() ? 0 : 1;
  }
  if (cfg.prey_capture) {
    PreyCaptureExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Prey capture: tectal=%.1fms, motor=%.1fms, PVN=%.1fHz, hb_spikes=%d",
        result.tectal_onset_ms, result.capture_latency_ms,
        result.tectal_peak_rate_hz, result.hindbrain_spikes);
    return result.responded() ? 0 : 1;
  }
  if (cfg.courtship) {
    CourtshipExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Courtship: pulses=%d, IPI=%.1fms, dPR1=%.1fHz, MN=%.1fHz",
        result.n_pulses, result.mean_ipi_ms,
        result.dpr1_rate_hz, result.motor_peak_rate_hz);
    return result.produced_pulses() ? 0 : 1;
  }
  if (cfg.twinning) {
    TwinningExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Twinning: continuity=%.3f, replaced=%.1f%%, CS+ %d->%d",
        result.behavioral_continuity, result.replacement_fraction * 100.0f,
        result.pre_cs_plus_spikes, result.post_cs_plus_spikes);
    return result.passed() ? 0 : 1;
  }
  if (cfg.ablation) {
    AblationExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Ablation: graceful=%.2f, half_life=%.0f%%, cliff=%.0f%%",
        result.graceful_score, result.half_life_fraction * 100,
        result.cliff_fraction * 100);
    return result.passed() ? 0 : 1;
  }
  if (cfg.compensated) {
    CompensatedAblationExperiment exp;
    auto result = exp.Run(cfg.seed);
    Log(LogLevel::kInfo, "Compensated: benefit=%.2f, pure_hl=%.0f%%, comp_hl=%.0f%%, ext=%.1fx",
        result.compensation_benefit, result.pure_half_life * 100,
        result.compensated_half_life * 100, result.lifetime_extension);
    return result.passed() ? 0 : 1;
  }
  if (cfg.multiscale) {
    return RunMultiscale(cfg);
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
