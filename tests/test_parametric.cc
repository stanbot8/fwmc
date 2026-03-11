// Parametric brain tests: cell type manager, parametric generator, brain spec loader,
// parameter sweep, scoring functions
#include "test_harness.h"

#include "core/cell_types.h"
#include "core/connectome_loader.h"
#include "core/connectome_export.h"
#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/parametric_gen.h"
#include "core/brain_spec_loader.h"
#include "core/param_sweep.h"
#include "core/parametric_sync.h"
#include "core/region_metrics.h"
#include "core/structural_plasticity.h"

// ===== CellTypeManager tests =====

TEST(cell_type_params) {
  auto kc = ParamsForCellType(CellType::kKenyonCell);
  assert(std::abs(kc.a - 0.02f) < 0.001f);
  assert(std::abs(kc.d - 8.0f) < 0.01f);

  auto fs = ParamsForCellType(CellType::kLN_local);
  assert(std::abs(fs.a - 0.1f) < 0.001f);
  assert(std::abs(fs.d - 2.0f) < 0.01f);

  auto burst = ParamsForCellType(CellType::kPN_excitatory);
  assert(std::abs(burst.c - (-50.0f)) < 0.01f);
}

TEST(cell_type_manager_assign) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 5; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);
  for (size_t i = 5; i < 10; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kLN_local);

  CellTypeManager types;
  types.AssignFromTypes(neurons);

  assert(types.neuron_params.size() == 10);
  // KCs have a=0.02, LNs have a=0.1
  assert(std::abs(types.Get(0).a - 0.02f) < 0.001f);
  assert(std::abs(types.Get(5).a - 0.1f) < 0.001f);
}

TEST(cell_type_manager_override) {
  NeuronArray neurons;
  neurons.Resize(5);
  for (size_t i = 0; i < 5; ++i)
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);

  CellTypeManager types;
  IzhikevichParams custom;
  custom.a = 0.05f;
  custom.b = 0.25f;
  custom.c = -55.0f;
  custom.d = 6.0f;
  types.SetOverride(CellType::kKenyonCell, custom);
  types.AssignFromTypes(neurons);

  // All neurons should have the custom override
  for (size_t i = 0; i < 5; ++i) {
    assert(std::abs(types.Get(i).a - 0.05f) < 0.001f);
    assert(std::abs(types.Get(i).c - (-55.0f)) < 0.01f);
  }
}

TEST(heterogeneous_step) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kKenyonCell);
    neurons.i_ext[i] = 15.0f;  // strong drive to ensure spiking
  }

  CellTypeManager types;
  types.AssignFromTypes(neurons);

  float sim_time = 0.0f;
  for (int step = 0; step < 2000; ++step) {
    IzhikevichStepHeterogeneous(neurons, 0.1f, sim_time, types);
    sim_time += 0.1f;
  }

  // With strong drive, at least some neurons should have spiked
  int spikes = neurons.CountSpikes();
  // Check that last_spike_time was updated for at least one neuron
  bool any_spike_recorded = false;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.last_spike_time[i] > 0.0f) {
      any_spike_recorded = true;
      break;
    }
  }
  assert(any_spike_recorded);
}

// ===== ParametricGenerator tests =====

TEST(parametric_gen_single_region) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test_region";
  reg.n_neurons = 50;
  reg.internal_density = 0.1f;
  reg.default_nt = kACh;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);

  assert(total == 50);
  assert(neurons.n == 50);
  assert(synapses.Size() > 0);  // ~10% density should create some synapses
  assert(gen.region_ranges.size() == 1);
  assert(gen.region_ranges[0].start == 0);
  assert(gen.region_ranges[0].end == 50);
}

TEST(parametric_gen_two_regions_with_projection) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec r1;
  r1.name = "region_a";
  r1.n_neurons = 30;
  r1.internal_density = 0.05f;
  spec.regions.push_back(r1);

  RegionSpec r2;
  r2.name = "region_b";
  r2.n_neurons = 20;
  r2.internal_density = 0.05f;
  spec.regions.push_back(r2);

  ProjectionSpec proj;
  proj.from_region = "region_a";
  proj.to_region = "region_b";
  proj.density = 0.1f;
  proj.nt_type = kACh;
  proj.weight_mean = 1.5f;
  proj.weight_std = 0.2f;
  spec.projections.push_back(proj);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  uint32_t total = gen.Generate(spec, neurons, synapses, types);

  assert(total == 50);
  assert(gen.region_ranges.size() == 2);
  assert(gen.region_ranges[0].end == 30);
  assert(gen.region_ranges[1].start == 30);
  assert(gen.region_ranges[1].end == 50);

  // Region assignments
  assert(neurons.region[0] == 0);
  assert(neurons.region[30] == 1);
}

TEST(parametric_gen_nt_distribution) {
  BrainSpec spec;
  spec.seed = 123;

  RegionSpec reg;
  reg.name = "mixed_nt";
  reg.n_neurons = 100;
  reg.internal_density = 0.2f;
  reg.nt_distribution.push_back({kACh, 0.5f});
  reg.nt_distribution.push_back({kGABA, 0.5f});
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // With 50/50 ACh/GABA distribution, count actual NT types in synapse table
  assert(synapses.Size() > 0);
  size_t n_ach = 0, n_gaba = 0;
  for (size_t i = 0; i < synapses.Size(); ++i) {
    if (synapses.nt_type[i] == kACh) n_ach++;
    else if (synapses.nt_type[i] == kGABA) n_gaba++;
  }
  assert(n_ach > 0);
  assert(n_gaba > 0);
  // Ratio should be roughly 50/50 (within 30% tolerance)
  float ratio = static_cast<float>(n_ach) / (n_ach + n_gaba);
  assert(ratio > 0.2f && ratio < 0.8f);
}

TEST(parametric_gen_cell_type_assignment) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec reg;
  reg.name = "typed_region";
  reg.n_neurons = 100;
  reg.internal_density = 0.01f;
  reg.cell_types.push_back({CellType::kKenyonCell, 0.6f});
  reg.cell_types.push_back({CellType::kLN_local, 0.4f});
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Count cell types
  int kc_count = 0, ln_count = 0;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.type[i] == static_cast<uint8_t>(CellType::kKenyonCell)) kc_count++;
    if (neurons.type[i] == static_cast<uint8_t>(CellType::kLN_local)) ln_count++;
  }
  assert(kc_count == 60);  // 60% of 100
  assert(ln_count == 40);  // 40% of 100
}

TEST(parametric_gen_run_simulation) {
  BrainSpec spec;
  spec.seed = 42;

  RegionSpec reg;
  reg.name = "sim_test";
  reg.n_neurons = 20;
  reg.internal_density = 0.15f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Inject current and simulate
  for (size_t i = 0; i < neurons.n; ++i) neurons.i_ext[i] = 10.0f;

  float sim_time = 0.0f;
  for (int step = 0; step < 1000; ++step) {
    neurons.ClearSynapticInput();
    synapses.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(neurons, 0.1f, sim_time, types);
    sim_time += 0.1f;
  }

  // Should produce activity
  bool any_spiked = false;
  for (size_t i = 0; i < neurons.n; ++i) {
    if (neurons.last_spike_time[i] > 0.0f) { any_spiked = true; break; }
  }
  assert(any_spiked);
}

// ===== BrainSpecLoader tests =====

TEST(brain_spec_loader_roundtrip) {
  // Write a minimal .brain file
  const char* content =
    "name = test_brain\n"
    "seed = 123\n"
    "weight_mean = 1.5\n"
    "weight_std = 0.4\n"
    "\n"
    "region.0.name = region_a\n"
    "region.0.n_neurons = 100\n"
    "region.0.density = 0.1\n"
    "region.0.nt = ACh\n"
    "region.0.types = KC:0.7 LN:0.3\n"
    "\n"
    "region.1.name = region_b\n"
    "region.1.n_neurons = 50\n"
    "region.1.density = 0.05\n"
    "\n"
    "projection.0.from = region_a\n"
    "projection.0.to = region_b\n"
    "projection.0.density = 0.02\n"
    "projection.0.nt = GABA\n"
    "projection.0.weight_mean = 1.2\n"
    "projection.0.weight_std = 0.3\n";

  std::string path = "test_tmp_brain_spec.brain";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  assert(result.has_value());
  auto& spec = *result;
  assert(spec.name == "test_brain");
  assert(spec.seed == 123);
  assert(std::abs(spec.global_weight_mean - 1.5f) < 0.01f);
  assert(spec.regions.size() == 2);
  assert(spec.regions[0].name == "region_a");
  assert(spec.regions[0].n_neurons == 100);
  assert(spec.regions[0].cell_types.size() == 2);
  assert(spec.regions[1].name == "region_b");
  assert(spec.regions[1].n_neurons == 50);
  assert(spec.projections.size() == 1);
  assert(spec.projections[0].from_region == "region_a");
  assert(spec.projections[0].to_region == "region_b");
  assert(spec.projections[0].nt_type == kGABA);
}

TEST(brain_spec_loader_missing_file) {
  auto result = BrainSpecLoader::Load("nonexistent_file.brain");
  assert(!result.has_value());
}

TEST(brain_spec_loader_empty_regions) {
  std::string path = "test_tmp_empty_brain.brain";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fputs("name = empty\n", f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  assert(!result.has_value());  // No regions = error
}

TEST(brain_spec_loader_nt_distribution) {
  const char* content =
    "region.0.name = mixed\n"
    "region.0.n_neurons = 50\n"
    "region.0.density = 0.1\n"
    "region.0.nt_dist = ACh:0.6 GABA:0.4\n";

  std::string path = "test_tmp_nt_dist.brain";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  assert(result.has_value());
  assert(result->regions[0].nt_distribution.size() == 2);
  assert(result->regions[0].nt_distribution[0].nt == kACh);
  assert(std::abs(result->regions[0].nt_distribution[0].fraction - 0.6f) < 0.01f);
}

// ===== ParamSweep tests =====

TEST(param_sweep_grid) {
  // Small grid sweep with tiny neuron population
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;  // no synapses, just test grid mechanics

  ParamSweep sweep;
  sweep.grid_steps = 2;  // 2^4 = 16 points
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.GridSweep(CellType::kGeneric, neurons, synapses,
                  scoring::TargetFiringRate(10.0f, 0.5f));

  assert(sweep.results.size() == 16);
  // Results should be sorted best-first
  for (size_t i = 1; i < sweep.results.size(); ++i) {
    assert(sweep.results[i-1].score >= sweep.results[i].score);
  }
}

TEST(param_sweep_random) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;

  ParamSweep sweep;
  sweep.random_samples = 20;
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.RandomSweep(CellType::kGeneric, neurons, synapses,
                    scoring::TargetFiringRate(10.0f, 0.5f));

  assert(sweep.results.size() == 20);
  // Sorted best-first
  for (size_t i = 1; i < sweep.results.size(); ++i) {
    assert(sweep.results[i-1].score >= sweep.results[i].score);
  }
}

TEST(param_sweep_refine) {
  NeuronArray neurons;
  neurons.Resize(10);
  for (size_t i = 0; i < 10; ++i) {
    neurons.type[i] = static_cast<uint8_t>(CellType::kGeneric);
    neurons.i_ext[i] = 8.0f;
  }

  SynapseTable synapses;

  ParamSweep sweep;
  sweep.grid_steps = 2;
  sweep.sim_duration_ms = 50.0f;
  sweep.dt_ms = 0.5f;

  sweep.GridSweep(CellType::kGeneric, neurons, synapses,
                  scoring::TargetFiringRate(10.0f, 0.5f));

  float score_before = sweep.BestScore();
  sweep.Refine(CellType::kGeneric, neurons, synapses,
               scoring::TargetFiringRate(10.0f, 0.5f), 10, 0.5f);

  // Refinement should not worsen the best score
  assert(sweep.BestScore() >= score_before);
}

// ===== Scoring function tests =====

TEST(scoring_target_firing_rate) {
  auto fn = scoring::TargetFiringRate(10.0f, 0.1f);

  NeuronArray neurons;
  neurons.Resize(100);
  // No spikes; score should reflect 0 Hz vs 10 Hz target
  float score_zero = fn(neurons, 1000.0f);

  // Simulate some spikes
  for (size_t i = 0; i < 100; ++i) neurons.spiked[i] = 1;
  float score_some = fn(neurons, 1000.0f);

  // Some spiking should score differently than zero
  assert(score_zero != score_some);
  assert(score_zero > 0.0f);  // 1/(1+error), always positive
}

TEST(scoring_activity_in_range) {
  auto fn = scoring::ActivityInRange(0.1f, 0.5f);

  NeuronArray neurons;
  neurons.Resize(100);

  // No activity
  float score_none = fn(neurons, 1000.0f);
  assert(score_none < 1.0f);

  // 30% active (in range)
  for (size_t i = 0; i < 30; ++i) neurons.last_spike_time[i] = 50.0f;
  float score_good = fn(neurons, 1000.0f);
  assert(std::abs(score_good - 1.0f) < 0.01f);

  // 80% active (above range)
  for (size_t i = 0; i < 80; ++i) neurons.last_spike_time[i] = 50.0f;
  float score_high = fn(neurons, 1000.0f);
  assert(score_high < 1.0f);
}

// ===== ParametricSync tests =====

TEST(parametric_sync_init) {
  ParametricSync sync;
  sync.Init(20, 100);
  assert(sync.neuron_state.size() == 20);
  assert(sync.weight_velocity.size() == 100);
  assert(sync.weight_error_accum.size() == 100);
  assert(sync.total_steps == 0);
}

TEST(parametric_sync_step_reduces_error) {
  // Create identical model and reference, then perturb model weights.
  // Sync should reduce the divergence over time.
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test";
  reg.n_neurons = 20;
  reg.internal_density = 0.15f;
  spec.regions.push_back(reg);

  NeuronArray model_neurons, ref_neurons;
  SynapseTable model_synapses, ref_synapses;
  CellTypeManager model_types, ref_types;
  ParametricGenerator gen1, gen2;

  gen1.Generate(spec, model_neurons, model_synapses, model_types);
  gen2.Generate(spec, ref_neurons, ref_synapses, ref_types);

  // Perturb model weights to create initial mismatch
  for (size_t i = 0; i < model_synapses.Size(); ++i) {
    model_synapses.weight[i] *= 1.5f;
  }

  // Inject current
  for (size_t i = 0; i < 20; ++i) {
    model_neurons.i_ext[i] = 8.0f;
    ref_neurons.i_ext[i] = 8.0f;
  }

  ParametricSync sync;
  sync.dt_ms = 0.5f;
  sync.metric_interval = 200;
  sync.weight_update_interval = 50;
  sync.param_update_interval = 500;
  sync.Init(20, model_synapses.Size());

  // Run for some steps
  float ref_time = 0.0f;
  for (int step = 0; step < 1000; ++step) {
    ref_neurons.ClearSynapticInput();
    ref_synapses.PropagateSpikes(ref_neurons.spiked.data(),
                                  ref_neurons.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(ref_neurons, 0.5f, ref_time, ref_types);
    ref_time += 0.5f;

    sync.Step(model_neurons, model_synapses, ref_neurons, model_types);
  }

  // Should have recorded metrics
  assert(sync.history.size() >= 2);

  // Correlation should improve over time (later > earlier)
  float early_corr = sync.history.front().global_correlation;
  float late_corr = sync.history.back().global_correlation;
  assert(late_corr > 0.0f);
  assert(late_corr >= early_corr);  // sync should not make things worse
}

TEST(parametric_sync_identical_brains_converge) {
  // Two identical brains should quickly converge
  BrainSpec spec;
  spec.seed = 99;
  RegionSpec reg;
  reg.name = "identical";
  reg.n_neurons = 10;
  reg.internal_density = 0.1f;
  spec.regions.push_back(reg);

  NeuronArray model, ref;
  SynapseTable model_syn, ref_syn;
  CellTypeManager model_types, ref_types;
  ParametricGenerator g1, g2;

  g1.Generate(spec, model, model_syn, model_types);
  g2.Generate(spec, ref, ref_syn, ref_types);

  for (size_t i = 0; i < 10; ++i) {
    model.i_ext[i] = 8.0f;
    ref.i_ext[i] = 8.0f;
  }

  ParametricSync sync;
  sync.dt_ms = 0.5f;
  sync.metric_interval = 100;
  sync.weight_update_interval = 50;
  sync.converge_threshold = 0.7f;
  sync.Init(10, model_syn.Size());

  float ref_time = 0.0f;
  for (int step = 0; step < 500; ++step) {
    ref.ClearSynapticInput();
    ref_syn.PropagateSpikes(ref.spiked.data(), ref.i_syn.data(), 1.0f);
    IzhikevichStepHeterogeneous(ref, 0.5f, ref_time, ref_types);
    ref_time += 0.5f;

    sync.Step(model, model_syn, ref, model_types);
  }

  // Identical brains with current injection should have high correlation
  assert(!sync.history.empty());
  assert(sync.history.back().global_correlation > 0.5f);
}

TEST(parametric_sync_has_converged) {
  ParametricSync sync;
  sync.target_convergence = 0.9f;
  assert(!sync.HasConverged());  // no history

  // Manually add a snapshot showing convergence
  SyncSnapshot snap;
  snap.fraction_converged = 0.95f;
  sync.history.push_back(snap);
  assert(sync.HasConverged());

  snap.fraction_converged = 0.5f;
  sync.history.push_back(snap);
  assert(!sync.HasConverged());
}

// ===== RegionMetrics tests =====

TEST(region_metrics_record) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec r1, r2;
  r1.name = "r1"; r1.n_neurons = 20; r1.internal_density = 0.05f;
  r2.name = "r2"; r2.n_neurons = 30; r2.internal_density = 0.05f;
  spec.regions.push_back(r1);
  spec.regions.push_back(r2);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  RegionMetrics metrics;
  metrics.Init(gen);
  assert(metrics.regions.size() == 2);

  // Simulate some spikes
  neurons.spiked[0] = 1;
  neurons.spiked[5] = 1;
  neurons.spiked[25] = 1;

  metrics.Record(neurons, 10.0f, 0.1f, 100);
  assert(metrics.history.size() == 1);
  assert(metrics.history[0].size() == 2);

  // Region r1 (0-19): 2 spikes
  assert(metrics.history[0][0].spike_count == 2);
  assert(metrics.history[0][0].name == "r1");
  // Region r2 (20-49): 1 spike
  assert(metrics.history[0][1].spike_count == 1);
  assert(metrics.history[0][1].name == "r2");
}

TEST(region_metrics_summary) {
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test"; reg.n_neurons = 10; reg.internal_density = 0.05f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  RegionMetrics metrics;
  metrics.Init(gen);

  // Two snapshots
  neurons.spiked[0] = 1;
  metrics.Record(neurons, 1.0f, 0.1f, 100);
  neurons.spiked[0] = 0;
  neurons.spiked[1] = 1;
  neurons.spiked[2] = 1;
  metrics.Record(neurons, 2.0f, 0.1f, 100);

  assert(metrics.history.size() == 2);
  // First snapshot: 1 spike, second: 2 spikes
  assert(metrics.history[0][0].spike_count == 1);
  assert(metrics.history[1][0].spike_count == 2);
}

// ===== ApplyStimuli tests =====

TEST(apply_stimuli_active_window) {
  NeuronArray neurons;
  neurons.Resize(20);

  ParametricGenerator::RegionRange reg{"test_region", 0, 20};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "pulse";
  stim.target_region = "test_region";
  stim.start_ms = 10.0f;
  stim.end_ms = 20.0f;
  stim.intensity = 5.0f;
  stim.fraction = 1.0f;
  std::vector<StimulusSpec> stimuli = {stim};

  // Before window: no current injected
  ApplyStimuli(stimuli, regions, neurons, 5.0f, 42);
  for (size_t i = 0; i < 20; ++i) assert(neurons.i_ext[i] == 0.0f);

  // Inside window: current injected
  ApplyStimuli(stimuli, regions, neurons, 15.0f, 42);
  for (size_t i = 0; i < 20; ++i) assert(neurons.i_ext[i] == 5.0f);

  // After window: no additional current (clear first)
  for (size_t i = 0; i < 20; ++i) neurons.i_ext[i] = 0.0f;
  ApplyStimuli(stimuli, regions, neurons, 25.0f, 42);
  for (size_t i = 0; i < 20; ++i) assert(neurons.i_ext[i] == 0.0f);
}

TEST(apply_stimuli_partial_fraction) {
  NeuronArray neurons;
  neurons.Resize(100);

  ParametricGenerator::RegionRange reg{"r", 0, 100};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "partial";
  stim.target_region = "r";
  stim.start_ms = 0.0f;
  stim.end_ms = 100.0f;
  stim.intensity = 3.0f;
  stim.fraction = 0.5f;
  std::vector<StimulusSpec> stimuli = {stim};

  ApplyStimuli(stimuli, regions, neurons, 50.0f, 42);

  // 50% of 100 = 50 neurons should get current
  int stimulated = 0;
  for (size_t i = 0; i < 100; ++i) {
    if (neurons.i_ext[i] > 0.0f) stimulated++;
  }
  assert(stimulated == 50);
}

TEST(apply_stimuli_wrong_region) {
  NeuronArray neurons;
  neurons.Resize(10);

  ParametricGenerator::RegionRange reg{"region_a", 0, 10};
  std::vector<ParametricGenerator::RegionRange> regions = {reg};

  StimulusSpec stim;
  stim.label = "miss";
  stim.target_region = "region_b";  // doesn't exist
  stim.start_ms = 0.0f;
  stim.end_ms = 100.0f;
  stim.intensity = 5.0f;
  stim.fraction = 1.0f;
  std::vector<StimulusSpec> stimuli = {stim};

  ApplyStimuli(stimuli, regions, neurons, 50.0f, 42);
  // No current should be injected
  for (size_t i = 0; i < 10; ++i) assert(neurons.i_ext[i] == 0.0f);
}

// ===== ConnectomeExport tests =====

TEST(connectome_export_neurons_roundtrip) {
  NeuronArray neurons;
  neurons.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    neurons.root_id[i] = 1000 + i;
    neurons.x[i] = static_cast<float>(i) * 1.5f;
    neurons.y[i] = static_cast<float>(i) * 2.0f;
    neurons.z[i] = static_cast<float>(i) * 0.5f;
    neurons.type[i] = static_cast<uint8_t>(i % 3);
  }

  std::string path = "test_tmp_export_neurons.bin";
  auto result = ConnectomeExport::ExportNeurons(path, neurons);
  assert(result.has_value());
  assert(*result == 20);

  // Load back
  NeuronArray loaded;
  auto load_result = ConnectomeLoader::LoadNeurons(path, loaded);
  remove(path.c_str());

  assert(load_result.has_value());
  assert(loaded.n == 20);
  for (size_t i = 0; i < 20; ++i) {
    assert(loaded.root_id[i] == neurons.root_id[i]);
    assert(std::abs(loaded.x[i] - neurons.x[i]) < 0.001f);
    assert(std::abs(loaded.y[i] - neurons.y[i]) < 0.001f);
    assert(std::abs(loaded.z[i] - neurons.z[i]) < 0.001f);
    assert(loaded.type[i] == neurons.type[i]);
  }
}

TEST(connectome_export_synapses_roundtrip) {
  // Build a small synapse table
  NeuronArray neurons;
  neurons.Resize(10);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0, 0, 1, 2, 3};
  std::vector<uint32_t> post = {1, 2, 3, 4, 5};
  std::vector<float> weights = {1.0f, 0.5f, 1.5f, 2.0f, 0.8f};
  std::vector<uint8_t> nt = {kACh, kGABA, kACh, kDA, kACh};
  synapses.BuildFromCOO(10, pre, post, weights, nt);

  std::string path = "test_tmp_export_synapses.bin";
  auto result = ConnectomeExport::ExportSynapses(path, synapses);
  assert(result.has_value());
  assert(*result == 5);

  // Load back
  SynapseTable loaded;
  auto load_result = ConnectomeLoader::LoadSynapses(path, 10, loaded);
  remove(path.c_str());

  assert(load_result.has_value());
  assert(loaded.Size() == 5);
}

TEST(connectome_export_empty_neurons) {
  NeuronArray empty;
  auto result = ConnectomeExport::ExportNeurons("test_tmp_empty.bin", empty);
  assert(!result.has_value());  // Should fail for empty array
}

// ===== StructuralPlasticity tests =====

TEST(structural_plasticity_prune) {
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0, 0, 1, 1};
  std::vector<uint32_t> post = {1, 2, 2, 3};
  std::vector<float> weights = {0.01f, 1.0f, 0.03f, 2.0f};
  std::vector<uint8_t> nt = {kACh, kACh, kACh, kACh};
  synapses.BuildFromCOO(4, pre, post, weights, nt);

  StructuralPlasticity sp;
  sp.config.prune_threshold = 0.05f;
  size_t pruned = sp.PruneWeak(synapses);

  // Two synapses below threshold (0.01 and 0.03)
  assert(pruned == 2);
  assert(synapses.weight[0] == 0.0f || synapses.weight[2] == 0.0f);
}

TEST(structural_plasticity_sprout) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weights = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  synapses.BuildFromCOO(5, pre, post, weights, nt);

  // Make several neurons active
  neurons.spiked[0] = 1;
  neurons.spiked[1] = 1;
  neurons.spiked[2] = 1;

  StructuralPlasticity sp;
  sp.config.sprout_rate = 1.0f;  // guarantee sprouting
  std::mt19937 rng(42);
  size_t sprouted = sp.SproutNew(synapses, neurons, rng);

  // Should have added some new synapses
  assert(sprouted > 0);
  assert(synapses.Size() > 1);
}

TEST(structural_plasticity_update_interval) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weights = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  synapses.BuildFromCOO(5, pre, post, weights, nt);

  StructuralPlasticity sp;
  sp.config.update_interval = 100;
  std::mt19937 rng(42);

  // Step 50: should NOT update
  sp.Update(synapses, neurons, 50, rng);
  assert(synapses.Size() == 1);

  // Step 100: should update (but no weak synapses or active neurons)
  sp.Update(synapses, neurons, 100, rng);
  assert(synapses.Size() == 1);  // no pruning (weight=1.0), no sprouting (no active)
}

// ===== Brain spec loader stimulus parsing =====

TEST(brain_spec_loader_stimuli) {
  const char* content =
    "region.0.name = test_r\n"
    "region.0.n_neurons = 50\n"
    "region.0.density = 0.1\n"
    "\n"
    "stimulus.0.label = pulse_a\n"
    "stimulus.0.region = test_r\n"
    "stimulus.0.start = 100\n"
    "stimulus.0.end = 200\n"
    "stimulus.0.intensity = 5.0\n"
    "stimulus.0.fraction = 0.3\n"
    "\n"
    "stimulus.1.label = pulse_b\n"
    "stimulus.1.region = test_r\n"
    "stimulus.1.start = 300\n"
    "stimulus.1.end = 500\n"
    "stimulus.1.intensity = 8.0\n";

  std::string path = "test_tmp_stimuli.brain";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fputs(content, f);
  fclose(f);

  auto result = BrainSpecLoader::Load(path);
  remove(path.c_str());

  assert(result.has_value());
  assert(result->stimuli.size() == 2);
  assert(result->stimuli[0].label == "pulse_a");
  assert(result->stimuli[0].target_region == "test_r");
  assert(std::abs(result->stimuli[0].start_ms - 100.0f) < 0.01f);
  assert(std::abs(result->stimuli[0].end_ms - 200.0f) < 0.01f);
  assert(std::abs(result->stimuli[0].intensity - 5.0f) < 0.01f);
  assert(std::abs(result->stimuli[0].fraction - 0.3f) < 0.01f);
  assert(result->stimuli[1].label == "pulse_b");
  assert(std::abs(result->stimuli[1].intensity - 8.0f) < 0.01f);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
