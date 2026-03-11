// Core tests: neuron dynamics, synapses, STDP, loader, config, checkpoint, stats, recorder
#include <cmath>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif
#include "test_harness.h"

#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/stdp.h"
#include "core/connectome_loader.h"
#include "core/config_loader.h"
#include "core/checkpoint.h"
#include "core/connectome_stats.h"
#include "core/experiment_config.h"
#include "core/recorder.h"
#include "core/parametric_gen.h"
#include "core/rate_monitor.h"
#include "core/motor_output.h"
#include "core/intrinsic_homeostasis.h"
#include "bridge/shadow_tracker.h"
#include "bridge/neuron_replacer.h"
#include "bridge/bridge_checkpoint.h"
#include "core/gap_junctions.h"
#include "core/short_term_plasticity.h"
#include "core/cpg.h"
#include "core/proprioception.h"

// ===== Core tests =====

TEST(neuron_array_init) {
  NeuronArray arr;
  arr.Resize(100);
  assert(arr.n == 100);
  assert(arr.v[0] == -65.0f);
  assert(arr.u[0] == -13.0f);
  assert(arr.spiked[0] == 0);
  assert(arr.CountSpikes() == 0);
}

TEST(empty_neuron_array) {
  NeuronArray arr;
  arr.Resize(0);
  assert(arr.n == 0);
  assert(arr.CountSpikes() == 0);
  arr.ClearSynapticInput();  // should not crash
}

TEST(izhikevich_spike) {
  NeuronArray arr;
  arr.Resize(1);
  arr.i_ext[0] = 15.0f;

  IzhikevichParams p;
  bool spiked = false;
  for (int i = 0; i < 1000 && !spiked; ++i) {
    IzhikevichStep(arr, 0.1f, i * 0.1f, p);
    if (arr.spiked[0]) spiked = true;
    arr.i_syn[0] = 0.0f;
  }
  assert(spiked && "Neuron should spike with strong input");
}

TEST(izhikevich_no_spike_without_input) {
  NeuronArray arr;
  arr.Resize(1);

  IzhikevichParams p;
  for (int i = 0; i < 1000; ++i) {
    IzhikevichStep(arr, 0.1f, i * 0.1f, p);
  }
  assert(arr.v[0] < p.v_thresh);
}

TEST(izhikevich_updates_last_spike_time) {
  NeuronArray arr;
  arr.Resize(1);
  arr.i_ext[0] = 15.0f;

  IzhikevichParams p;
  float initial_spike_time = arr.last_spike_time[0];
  for (int i = 0; i < 1000; ++i) {
    float t = i * 0.1f;
    IzhikevichStep(arr, 0.1f, t, p);
    if (arr.spiked[0]) {
      assert(arr.last_spike_time[0] == t);
      assert(arr.last_spike_time[0] != initial_spike_time);
      return;
    }
    arr.i_syn[0] = 0.0f;
  }
  assert(false && "Should have spiked");
}

TEST(izhikevich_nan_recovery) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = std::numeric_limits<float>::quiet_NaN();
  arr.u[0] = std::numeric_limits<float>::quiet_NaN();

  IzhikevichParams p;
  IzhikevichStep(arr, 0.1f, 0.0f, p);
  assert(std::isfinite(arr.v[0]));
  assert(std::isfinite(arr.u[0]));
}

TEST(lif_spike) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = -70.0f;
  arr.i_ext[0] = 10.0f;

  LIFParams p;
  bool spiked = false;
  for (int i = 0; i < 1000 && !spiked; ++i) {
    LIFStep(arr, 0.1f, i * 0.1f, p);
    if (arr.spiked[0]) spiked = true;
    arr.i_syn[0] = 0.0f;
  }
  assert(spiked && "LIF neuron should spike with strong input");
}

TEST(synapse_table_csr) {
  std::vector<uint32_t> pre  = {0, 0, 1, 2};
  std::vector<uint32_t> post = {1, 2, 2, 0};
  std::vector<float> weight  = {1.0f, 2.0f, 1.5f, 0.5f};
  std::vector<uint8_t> nt    = {kACh, kACh, kGABA, kACh};

  SynapseTable table;
  table.BuildFromCOO(3, pre, post, weight, nt);

  assert(table.n_neurons == 3);
  assert(table.Size() == 4);
  assert(table.row_ptr[1] - table.row_ptr[0] == 2);
  assert(table.row_ptr[2] - table.row_ptr[1] == 1);
  assert(table.row_ptr[3] - table.row_ptr[2] == 1);
}

TEST(spike_propagation) {
  std::vector<uint32_t> pre  = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weight  = {2.0f};
  std::vector<uint8_t> nt    = {kACh};

  SynapseTable table;
  table.BuildFromCOO(2, pre, post, weight, nt);

  uint8_t spiked[2] = {1, 0};
  float i_syn[2] = {0.0f, 0.0f};

  table.PropagateSpikes(spiked, i_syn, 1.0f);
  assert(std::abs(i_syn[1] - 2.0f) < 1e-6f);
  assert(i_syn[0] == 0.0f);
}

TEST(inhibitory_propagation) {
  std::vector<uint32_t> pre  = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> weight  = {3.0f};
  std::vector<uint8_t> nt    = {kGABA};

  SynapseTable table;
  table.BuildFromCOO(2, pre, post, weight, nt);

  uint8_t spiked[2] = {1, 0};
  float i_syn[2] = {0.0f, 0.0f};

  table.PropagateSpikes(spiked, i_syn, 1.0f);
  assert(std::abs(i_syn[1] - (-3.0f)) < 1e-6f);
}

TEST(synapse_no_synapses) {
  SynapseTable table;
  table.BuildFromCOO(5, {}, {}, {}, {});
  assert(table.Size() == 0);

  uint8_t spiked[5] = {1, 0, 1, 0, 1};
  float i_syn[5] = {};
  table.PropagateSpikes(spiked, i_syn, 1.0f);
  for (int i = 0; i < 5; ++i) assert(i_syn[i] == 0.0f);
}

TEST(synapse_oob_indices) {
  SynapseTable table;
  // Post index 5 is out of bounds for 3 neurons
  table.BuildFromCOO(3, {0, 1}, {1, 5}, {1.0f, 2.0f}, {kACh, kACh});
  assert(table.Size() == 0 && "OOB index should produce empty table");

  // Pre index 10 is out of bounds
  SynapseTable table2;
  table2.BuildFromCOO(3, {10, 1}, {1, 2}, {1.0f, 2.0f}, {kACh, kACh});
  assert(table2.Size() == 0);
}

// ===== STDP tests =====

TEST(stdp_potentiation) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  arr.last_spike_time[0] = 10.0f;
  arr.spiked[0] = 0;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 15.0f;

  float original_weight = syn.weight[0];
  STDPParams p;
  STDPUpdate(syn, arr, 15.0f, p);
  assert(syn.weight[0] > original_weight && "Weight should increase (potentiation)");
}

TEST(stdp_depression) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  arr.last_spike_time[1] = 5.0f;
  arr.spiked[0] = 1;
  arr.spiked[1] = 0;
  arr.last_spike_time[0] = 10.0f;

  float original_weight = syn.weight[0];
  STDPParams p;
  STDPUpdate(syn, arr, 10.0f, p);
  assert(syn.weight[0] < original_weight && "Weight should decrease (depression)");
}

TEST(stdp_weight_bounds) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {9.99f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  for (int i = 0; i < 100; ++i) {
    arr.last_spike_time[0] = i * 10.0f;
    arr.spiked[1] = 1;
    arr.last_spike_time[1] = i * 10.0f + 5.0f;
    arr.spiked[0] = 0;
    STDPUpdate(syn, arr, i * 10.0f + 5.0f, p);
  }
  assert(syn.weight[0] <= p.w_max);

  syn.weight[0] = 0.5f;
  float w_before_depression = syn.weight[0];
  for (int i = 0; i < 100; ++i) {
    arr.last_spike_time[1] = i * 10.0f;
    arr.spiked[0] = 1;
    arr.last_spike_time[0] = i * 10.0f + 5.0f;
    arr.spiked[1] = 0;
    STDPUpdate(syn, arr, i * 10.0f + 5.0f, p);
  }
  assert(syn.weight[0] >= p.w_min);
  assert(syn.weight[0] < w_before_depression && "Depression should reduce weight");
}

TEST(stdp_no_change_without_spikes) {
  NeuronArray arr;
  arr.Resize(2);
  arr.spiked[0] = 0;
  arr.spiked[1] = 0;

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  STDPUpdate(syn, arr, 100.0f, p);
  assert(syn.weight[0] == 5.0f);
}

TEST(stdp_timing_window) {
  NeuronArray arr;
  arr.Resize(2);

  std::vector<uint32_t> pre_v = {0}, post_v = {1};
  std::vector<float> w = {5.0f};
  std::vector<uint8_t> nt = {kACh};
  SynapseTable syn;
  syn.BuildFromCOO(2, pre_v, post_v, w, nt);

  STDPParams p;
  arr.last_spike_time[0] = 0.0f;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 200.0f;

  STDPUpdate(syn, arr, 200.0f, p);
  assert(syn.weight[0] == 5.0f && "No change outside timing window");
}

// ===== ConnectomeLoader tests =====

TEST(loader_neurons_roundtrip) {
  uint32_t count = 3;
#ifdef _MSC_VER
  #pragma pack(push, 1)
  struct NeuronRecord { uint64_t id; float x, y, z; uint8_t type; };
  #pragma pack(pop)
#else
  struct NeuronRecord { uint64_t id; float x, y, z; uint8_t type; } __attribute__((packed));
#endif
  NeuronRecord records[3] = {
    {100, 1.0f, 2.0f, 3.0f, 1},
    {200, 4.0f, 5.0f, 6.0f, 2},
    {300, 7.0f, 8.0f, 9.0f, 0},
  };

  std::vector<uint8_t> buf;
  buf.resize(sizeof(count) + sizeof(records));
  memcpy(buf.data(), &count, sizeof(count));
  memcpy(buf.data() + sizeof(count), records, sizeof(records));

  auto path = WriteTempFile("neurons.bin", buf.data(), buf.size());

  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons(path, neurons);
  assert(result.has_value());
  assert(*result == 3);
  assert(neurons.n == 3);
  assert(neurons.root_id[0] == 100);
  assert(neurons.root_id[2] == 300);
  assert(neurons.x[1] == 4.0f);
  assert(neurons.type[0] == 1);

  remove(path.c_str());
}

TEST(loader_synapses_roundtrip) {
  uint32_t count = 2;
#ifdef _MSC_VER
  #pragma pack(push, 1)
  struct SynRecord { uint32_t pre, post; float w; uint8_t nt; };
  #pragma pack(pop)
#else
  struct SynRecord { uint32_t pre, post; float w; uint8_t nt; } __attribute__((packed));
#endif
  SynRecord records[2] = {
    {0, 1, 2.5f, kACh},
    {1, 0, 1.0f, kGABA},
  };

  std::vector<uint8_t> buf;
  buf.resize(sizeof(count) + sizeof(records));
  memcpy(buf.data(), &count, sizeof(count));
  memcpy(buf.data() + sizeof(count), records, sizeof(records));

  auto path = WriteTempFile("synapses.bin", buf.data(), buf.size());

  SynapseTable table;
  auto result = ConnectomeLoader::LoadSynapses(path, 2, table);
  assert(result.has_value());
  assert(*result == 2);
  assert(table.Size() == 2);
  assert(table.n_neurons == 2);

  remove(path.c_str());
}

TEST(loader_missing_file) {
  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons("nonexistent_file.bin", neurons);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kFileNotFound);
}

TEST(loader_truncated_file) {
  uint32_t count = 100;
  auto path = WriteTempFile("trunc.bin", &count, sizeof(count));

  NeuronArray neurons;
  auto result = ConnectomeLoader::LoadNeurons(path, neurons);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kCorruptedData);

  remove(path.c_str());
}

// ===== Config Loader tests =====

TEST(config_loader_basic) {
  std::string path = "test_tmp_config.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "name = test_experiment\n");
  fprintf(f, "dt_ms = 0.5\n");
  fprintf(f, "duration_ms = 5000\n");
  fprintf(f, "enable_stdp = true\n");
  fprintf(f, "bridge_mode = 1\n");
  fprintf(f, "connectome_dir = data/test\n");
  fprintf(f, "output_dir = results/test\n");
  fprintf(f, "monitor_neurons = 0 1 2 3\n");
  fprintf(f, "stimulus: odor_A 100 200 0.8 0,1,2\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(result.has_value());

  auto& cfg = *result;
  assert(cfg.name == "test_experiment");
  assert(std::abs(cfg.dt_ms - 0.5f) < 0.01f);
  assert(std::abs(cfg.duration_ms - 5000.0f) < 0.01f);
  assert(cfg.enable_stdp == true);
  assert(cfg.bridge_mode == 1);
  assert(cfg.connectome_dir == "data/test");
  assert(cfg.output_dir == "results/test");
  assert(cfg.monitor_neurons.size() == 4);
  assert(cfg.monitor_neurons[2] == 2);
  assert(cfg.stimulus_protocol.size() == 1);
  assert(cfg.stimulus_protocol[0].label == "odor_A");
  assert(cfg.stimulus_protocol[0].target_neurons.size() == 3);

  remove(path.c_str());
}

TEST(config_loader_missing_file) {
  auto result = ConfigLoader::Load("nonexistent_config_12345.cfg");
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kFileNotFound);
}

TEST(config_loader_comments_and_blanks) {
  std::string path = "test_tmp_comments.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "# This is a comment\n");
  fprintf(f, "\n");
  fprintf(f, "  # Indented comment\n");
  fprintf(f, "name = after_comments\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(result.has_value());
  assert(result->name == "after_comments");

  remove(path.c_str());
}

TEST(config_loader_invalid_dt) {
  std::string path = "test_tmp_bad_dt.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "dt_ms = -1.0\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(!result.has_value());
  assert(result.error().code == ErrorCode::kInvalidParam);

  remove(path.c_str());
}

TEST(config_loader_invalid_value) {
  std::string path = "test_tmp_bad_val.cfg";
  FILE* f = fopen(path.c_str(), "w");
  assert(f);
  fprintf(f, "dt_ms = notanumber\n");
  fclose(f);

  auto result = ConfigLoader::Load(path);
  assert(!result.has_value());

  remove(path.c_str());
}

// ===== Experiment Config tests =====

// ===== Recorder tests =====

TEST(recorder_open_close) {
  std::string dir = "test_tmp_recorder_out";
  Recorder rec;
  rec.record_spikes = true;
  rec.record_voltages = true;
  rec.record_shadow_metrics = true;
  rec.record_per_neuron_error = false;

  bool ok = rec.Open(dir, 10);
  assert(ok);
  assert(rec.n_neurons == 10);

  NeuronArray arr;
  arr.Resize(10);
  arr.spiked[3] = 1;
  arr.v[3] = -40.0f;

  rec.RecordStep(1.0f, arr, nullptr, 0, 0.0f, nullptr);
  assert(rec.n_recorded_steps == 1);

  rec.Close();

  FILE* sf = fopen((dir + "/spikes.bin").c_str(), "rb");
  assert(sf);
  fclose(sf);

  FILE* vf = fopen((dir + "/voltages.bin").c_str(), "rb");
  assert(vf);
  fclose(vf);

  FILE* mf = fopen((dir + "/metrics.csv").c_str(), "rb");
  assert(mf);
  fclose(mf);

  remove((dir + "/spikes.bin").c_str());
  remove((dir + "/voltages.bin").c_str());
  remove((dir + "/metrics.csv").c_str());
#ifdef _WIN32
  _rmdir(dir.c_str());
#else
  rmdir(dir.c_str());
#endif
}

// ===== Checkpoint tests =====

TEST(checkpoint_save_load_roundtrip) {
  NeuronArray neurons;
  neurons.Resize(10);
  neurons.v[0] = -50.0f;
  neurons.v[5] = -40.0f;
  neurons.u[3] = -10.0f;
  neurons.dopamine[2] = 0.5f;
  neurons.last_spike_time[7] = 42.0f;
  neurons.type[4] = 3;
  neurons.region[6] = 2;
  neurons.i_ext[1] = 5.0f;

  SynapseTable synapses;
  synapses.BuildFromCOO(10, {0, 1, 2}, {1, 2, 3}, {0.5f, 0.7f, 0.3f},
                         {kACh, kGABA, kGlut});
  synapses.weight[1] = 0.99f;
  synapses.InitReleaseProbability(0.3f);
  STPParams stp_params;
  stp_params.U_se = 0.4f;
  synapses.InitSTP(stp_params);
  synapses.stp_u[1] = 0.6f;

  NeuronReplacer replacer;
  replacer.Init(10);
  replacer.state[0] = NeuronReplacer::State::kMonitored;
  replacer.state[3] = NeuronReplacer::State::kBridged;
  replacer.running_correlation[3] = 0.85f;
  replacer.rollback_count[5] = 2;

  ShadowTracker shadow;
  shadow.last_resync_time = 100.0f;
  ShadowTracker::DriftSnapshot snap{};
  snap.time_ms = 50.0f;
  snap.spike_correlation = 0.92f;
  snap.population_rmse = 0.15f;
  shadow.history.push_back(snap);

  float sim_time = 500.0f;
  int total_steps = 5000;
  int total_resyncs = 3;

  std::string path = "test_tmp_checkpoint.bin";
  auto ext = BridgeCheckpoint::Serialize(replacer, shadow);
  bool saved = Checkpoint::Save(path, sim_time, total_steps, total_resyncs,
                                 neurons, synapses, ext);
  assert(saved);

  NeuronArray neurons2;
  neurons2.Resize(10);
  SynapseTable synapses2;
  synapses2.BuildFromCOO(10, {0, 1, 2}, {1, 2, 3}, {0.5f, 0.7f, 0.3f},
                          {kACh, kGABA, kGlut});
  NeuronReplacer replacer2;
  replacer2.Init(10);
  ShadowTracker shadow2;
  float sim_time2 = 0;
  int steps2 = 0, resyncs2 = 0;

  std::vector<uint8_t> ext2;
  bool loaded = Checkpoint::Load(path, sim_time2, steps2, resyncs2,
                                  neurons2, synapses2, ext2);
  assert(loaded);
  BridgeCheckpoint::Deserialize(ext2, replacer2, shadow2);

  assert(sim_time2 == 500.0f);
  assert(steps2 == 5000);
  assert(resyncs2 == 3);
  assert(neurons2.v[0] == -50.0f);
  assert(neurons2.v[5] == -40.0f);
  assert(neurons2.u[3] == -10.0f);
  assert(neurons2.dopamine[2] == 0.5f);
  assert(neurons2.last_spike_time[7] == 42.0f);
  assert(synapses2.weight[1] == 0.99f);
  assert(neurons2.type[4] == 3);
  assert(neurons2.region[6] == 2);
  assert(neurons2.i_ext[1] == 5.0f);
  assert(synapses2.HasStochasticRelease());
  assert(std::abs(synapses2.p_release[0] - 0.3f) < 1e-6f);
  assert(synapses2.HasSTP());
  assert(std::abs(synapses2.stp_u[1] - 0.6f) < 1e-6f);
  assert(std::abs(synapses2.stp_U_se[0] - 0.4f) < 1e-6f);
  assert(replacer2.state[0] == NeuronReplacer::State::kMonitored);
  assert(replacer2.state[3] == NeuronReplacer::State::kBridged);
  assert(std::abs(replacer2.running_correlation[3] - 0.85f) < 1e-6f);
  assert(replacer2.rollback_count[5] == 2);
  assert(shadow2.last_resync_time == 100.0f);
  assert(shadow2.history.size() == 1);
  assert(std::abs(shadow2.history[0].spike_correlation - 0.92f) < 1e-6f);

  remove(path.c_str());
}

TEST(checkpoint_bad_magic) {
  std::string path = "test_tmp_bad_ckpt.bin";
  FILE* f = fopen(path.c_str(), "wb");
  uint32_t garbage = 0xDEADBEEF;
  fwrite(&garbage, 4, 1, f);
  fclose(f);

  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable synapses;
  synapses.BuildFromCOO(5, {0}, {1}, {1.0f}, {kACh});
  std::vector<uint8_t> ext;
  float t = 0; int s = 0, r = 0;

  bool loaded = Checkpoint::Load(path, t, s, r, neurons, synapses, ext);
  assert(!loaded);

  remove(path.c_str());
}

TEST(checkpoint_size_mismatch) {
  NeuronArray neurons;
  neurons.Resize(10);
  SynapseTable synapses;
  synapses.BuildFromCOO(10, {0}, {1}, {1.0f}, {kACh});

  std::string path = "test_tmp_mismatch.bin";
  Checkpoint::Save(path, 0, 0, 0, neurons, synapses);

  NeuronArray neurons2;
  neurons2.Resize(5);
  SynapseTable synapses2;
  synapses2.BuildFromCOO(5, {0}, {1}, {1.0f}, {kACh});
  std::vector<uint8_t> ext;
  float t = 0; int s = 0, r = 0;

  bool loaded = Checkpoint::Load(path, t, s, r, neurons2, synapses2, ext);
  assert(!loaded);

  remove(path.c_str());
}

// ===== Connectome stats tests =====

TEST(connectome_stats_basic) {
  SynapseTable synapses;
  synapses.BuildFromCOO(5,
      {0, 0, 1, 2, 3},
      {1, 2, 3, 4, 0},
      {1.0f, 0.5f, 0.3f, 0.8f, 0.6f},
      {kACh, kGABA, kGlut, kDA, kACh});

  NeuronArray neurons;
  neurons.Resize(5);

  ConnectomeStats stats;
  bool valid = stats.Compute(synapses, neurons);
  assert(valid);

  assert(stats.n_neurons == 5);
  assert(stats.n_synapses == 5);
  assert(stats.n_ach == 2);
  assert(stats.n_gaba == 1);
  assert(stats.n_glut == 1);
  assert(stats.n_da == 1);
  assert(stats.max_out_degree == 2);
  assert(stats.min_weight == 0.3f);
  assert(stats.max_weight == 1.0f);
  assert(stats.n_self_loops == 0);
  assert(stats.n_out_of_bounds == 0);
  assert(stats.n_nan_weights == 0);
}

TEST(connectome_stats_self_loops) {
  SynapseTable synapses;
  synapses.BuildFromCOO(3,
      {0, 1, 1},
      {1, 1, 2},
      {1.0f, 0.5f, 0.3f},
      {kACh, kACh, kACh});

  NeuronArray neurons;
  neurons.Resize(3);

  ConnectomeStats stats;
  stats.Compute(synapses, neurons);
  assert(stats.n_self_loops == 1);
}

TEST(connectome_stats_isolated_neurons) {
  SynapseTable synapses;
  synapses.BuildFromCOO(4, {0}, {1}, {1.0f}, {kACh});

  NeuronArray neurons;
  neurons.Resize(4);

  ConnectomeStats stats;
  stats.Compute(synapses, neurons);
  assert(stats.n_isolated_neurons == 2);
}

// ===== Stochastic synapse tests =====

TEST(stochastic_release_reduces_transmission) {
  // With p_release=0.3, fewer post-synaptic neurons should receive input
  // compared to deterministic (p=1) propagation.
  NeuronArray neurons;
  neurons.Resize(20);
  SynapseTable syn;

  // Wire neuron 0 -> all others with deterministic weights
  std::vector<uint32_t> pre, post;
  std::vector<float> w;
  std::vector<uint8_t> nt;
  for (uint32_t i = 1; i < 20; ++i) {
    pre.push_back(0); post.push_back(i);
    w.push_back(1.0f); nt.push_back(kACh);
  }
  syn.BuildFromCOO(20, pre, post, w, nt);

  // Deterministic: all 19 targets get input
  neurons.spiked[0] = 1;
  std::fill(neurons.i_syn.begin(), neurons.i_syn.end(), 0.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  int det_hits = 0;
  for (uint32_t i = 1; i < 20; ++i) {
    if (neurons.i_syn[i] > 0.0f) det_hits++;
  }
  assert(det_hits == 19);

  // Stochastic with p=0.3: run many trials, average hits should be ~5.7
  syn.InitReleaseProbability(0.3f);
  std::mt19937 rng(42);
  int total_hits = 0;
  int trials = 200;
  for (int t = 0; t < trials; ++t) {
    std::fill(neurons.i_syn.begin(), neurons.i_syn.end(), 0.0f);
    neurons.spiked[0] = 1;
    syn.PropagateSpikesMonteCarlo(neurons.spiked.data(), neurons.i_syn.data(),
                                   1.0f, rng);
    for (uint32_t i = 1; i < 20; ++i) {
      if (neurons.i_syn[i] > 0.0f) total_hits++;
    }
  }
  float avg = static_cast<float>(total_hits) / trials;
  // Expected ~5.7 (19 * 0.3). Allow wide margin.
  assert(avg > 3.0f && avg < 9.0f);
}

TEST(stochastic_release_zero_blocks_all) {
  NeuronArray neurons;
  neurons.Resize(5);
  SynapseTable syn;
  std::vector<uint32_t> pre = {0, 0, 0};
  std::vector<uint32_t> post = {1, 2, 3};
  std::vector<float> w = {1.0f, 1.0f, 1.0f};
  std::vector<uint8_t> nt = {kACh, kACh, kACh};
  syn.BuildFromCOO(5, pre, post, w, nt);
  syn.InitReleaseProbability(0.0f);

  neurons.spiked[0] = 1;
  std::mt19937 rng(99);
  syn.PropagateSpikesMonteCarlo(neurons.spiked.data(), neurons.i_syn.data(),
                                 1.0f, rng);
  for (uint32_t i = 1; i <= 3; ++i) {
    assert(neurons.i_syn[i] == 0.0f);
  }
}

TEST(stp_depression_reduces_weight) {
  // Repeated spikes should depress the synapse (x decreases).
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 200.0f;
  params.tau_f = 50.0f;
  syn.InitSTP(params);

  // First spike
  float ux1 = syn.UpdateSTP(0);
  // Second spike immediately (no recovery)
  float ux2 = syn.UpdateSTP(0);
  // Third spike
  float ux3 = syn.UpdateSTP(0);

  // Each successive spike should produce less transmission (depression)
  assert(ux1 > ux2);
  assert(ux2 > ux3);
  // First spike: u goes to 0.75, x=1 -> ux=0.75, x becomes 0.25
  assert(ux1 > 0.5f);
}

TEST(stp_recovery_restores_strength) {
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 200.0f;
  params.tau_f = 50.0f;
  syn.InitSTP(params);

  // Depress with 5 rapid spikes
  for (int i = 0; i < 5; ++i) syn.UpdateSTP(0);
  float depressed_x = syn.stp_x[0];

  // Recover for a long time (1000ms in 1ms steps)
  for (int i = 0; i < 1000; ++i) syn.RecoverSTP(1.0f);
  float recovered_x = syn.stp_x[0];

  // x should recover toward 1.0
  assert(recovered_x > depressed_x + 0.3f);
  assert(recovered_x > 0.9f);
}

TEST(stp_facilitation_increases_u) {
  SynapseTable syn;
  std::vector<uint32_t> pre = {0};
  std::vector<uint32_t> post = {1};
  std::vector<float> w = {1.0f};
  std::vector<uint8_t> nt = {kACh};
  syn.BuildFromCOO(2, pre, post, w, nt);

  // Low U_se with long facilitation time constant
  STPParams params;
  params.U_se = 0.1f;
  params.tau_d = 500.0f;  // slow depression
  params.tau_f = 200.0f;  // slow facilitation decay
  syn.InitSTP(params);

  float u_before = syn.stp_u[0];
  syn.UpdateSTP(0);
  float u_after = syn.stp_u[0];

  // u should increase after a spike (facilitation)
  assert(u_after > u_before);
}

TEST(parametric_gen_stochastic_release) {
  // Verify that parametric generator passes release_probability through
  BrainSpec spec;
  spec.seed = 42;
  RegionSpec reg;
  reg.name = "test";
  reg.n_neurons = 10;
  reg.internal_density = 0.5f;
  reg.release_probability = 0.4f;
  spec.regions.push_back(reg);

  NeuronArray neurons;
  SynapseTable synapses;
  CellTypeManager types;
  ParametricGenerator gen;
  gen.Generate(spec, neurons, synapses, types);

  // Should have stochastic release enabled
  assert(synapses.HasStochasticRelease());
  // All release probabilities should be 0.4
  for (size_t i = 0; i < synapses.p_release.size(); ++i) {
    assert(std::abs(synapses.p_release[i] - 0.4f) < 0.001f);
  }
}

// ===== Refractory period test =====

TEST(izhikevich_refractory_period) {
  NeuronArray arr;
  arr.Resize(1);
  arr.v[0] = 35.0f;  // above threshold, should spike
  arr.u[0] = -14.0f;
  arr.i_ext[0] = 20.0f;

  IzhikevichParams p;
  p.refractory_ms = 2.0f;

  // First step: should fire
  IzhikevichStep(arr, 0.1f, 0.0f, p);
  assert(arr.spiked[0] == 1);

  // Drive voltage back up immediately
  arr.v[0] = 35.0f;

  // Step at 0.5ms (within refractory): should NOT fire
  IzhikevichStep(arr, 0.1f, 0.5f, p);
  assert(arr.spiked[0] == 0);

  // Step at 3.0ms (past refractory): should fire
  arr.v[0] = 35.0f;
  IzhikevichStep(arr, 0.1f, 3.0f, p);
  assert(arr.spiked[0] == 1);
}

// ===== Synaptic delay test =====

TEST(synapse_delay_ring_buffer) {
  SynapseTable syn;
  syn.BuildFromCOO(3, {0}, {1}, {1.0f}, {kACh});
  syn.InitDelay(1.0f, 0.5f);  // 1ms delay at 0.5ms timestep = 2 steps

  assert(syn.HasDelays());
  assert(syn.delay_steps[0] == 2);

  // Spike neuron 0
  uint8_t spiked[3] = {1, 0, 0};
  float i_syn[3] = {0, 0, 0};

  // Propagate: should write into delay buffer, not i_syn directly
  syn.DeliverDelayed(i_syn);
  syn.PropagateSpikes(spiked, i_syn, 1.0f);
  syn.AdvanceDelayRing();
  assert(i_syn[1] == 0.0f);  // not yet delivered

  // Step 2: still waiting
  spiked[0] = 0;
  std::fill(i_syn, i_syn + 3, 0.0f);
  syn.DeliverDelayed(i_syn);
  syn.PropagateSpikes(spiked, i_syn, 1.0f);
  syn.AdvanceDelayRing();
  assert(i_syn[1] == 0.0f);

  // Step 3: current arrives after 2-step delay
  std::fill(i_syn, i_syn + 3, 0.0f);
  syn.DeliverDelayed(i_syn);
  assert(i_syn[1] > 0.0f);  // delayed current delivered
}

// ===== Glutamate sign test =====

TEST(glutamate_sign_inhibitory) {
  // In Drosophila, glutamate is inhibitory via GluCl receptors
  assert(SynapseTable::Sign(kGlut) == -1.0f);
  assert(SynapseTable::Sign(kGABA) == -1.0f);
  assert(SynapseTable::Sign(kACh) == 1.0f);
}

// ===== STP clamping test =====

TEST(stp_state_clamped) {
  // Rapid repeated spikes should not drive STP state out of [0,1]
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});
  STPParams params;
  params.U_se = 0.5f;
  params.tau_d = 10.0f;   // short recovery
  params.tau_f = 10.0f;
  syn.InitSTP(params);

  // Hammer with 50 spikes, no recovery between them
  for (int i = 0; i < 50; ++i) {
    syn.UpdateSTP(0);
  }
  // u should stay in [0,1], x should stay >= 0
  assert(syn.stp_u[0] >= 0.0f && syn.stp_u[0] <= 1.0f);
  assert(syn.stp_x[0] >= 0.0f && syn.stp_x[0] <= 1.0f);
}

TEST(all_neurons_spike_propagation) {
  // When all neurons spike simultaneously, propagation should work correctly
  NeuronArray neurons;
  neurons.Resize(4);
  for (size_t i = 0; i < 4; ++i) neurons.spiked[i] = 1;

  SynapseTable syn;
  syn.BuildFromCOO(4,
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {kACh, kACh, kACh, kACh});

  float i_syn[4] = {};
  syn.PropagateSpikes(neurons.spiked.data(), i_syn, 1.0f);
  // Every neuron should receive input from its pre-synaptic partner
  for (int i = 0; i < 4; ++i) {
    assert(i_syn[i] > 0.0f);
  }
}

// ===== Exponential synaptic decay test =====

TEST(synaptic_decay_exponential) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.i_syn[0] = 10.0f;
  neurons.i_syn[1] = 5.0f;

  // Decay with tau=2ms, dt=1ms: factor = exp(-0.5) ~ 0.607
  neurons.DecaySynapticInput(1.0f, 2.0f);
  float expected = 10.0f * std::exp(-0.5f);
  assert(std::abs(neurons.i_syn[0] - expected) < 0.01f);
  assert(neurons.i_syn[0] > 0.0f);  // not zeroed
  assert(neurons.i_syn[1] > 0.0f);

  // Multiple steps should continue decaying
  float before = neurons.i_syn[0];
  neurons.DecaySynapticInput(1.0f, 2.0f);
  assert(neurons.i_syn[0] < before);
  assert(neurons.i_syn[0] > 0.0f);
}

TEST(synaptic_decay_accumulates_with_new_spikes) {
  NeuronArray neurons;
  neurons.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {3.0f}, {kACh});

  // Step 1: neuron 0 spikes, delivers current
  neurons.spiked[0] = 1;
  neurons.DecaySynapticInput(0.1f, 2.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  float after_first = neurons.i_syn[1];
  assert(after_first > 0.0f);

  // Step 2: no spike, current decays but doesn't vanish
  neurons.spiked[0] = 0;
  neurons.DecaySynapticInput(0.1f, 2.0f);
  syn.PropagateSpikes(neurons.spiked.data(), neurons.i_syn.data(), 1.0f);
  assert(neurons.i_syn[1] > 0.0f);
  assert(neurons.i_syn[1] < after_first);  // decayed
}

// ===== Eligibility trace tests =====

TEST(eligibility_trace_accumulates) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.tau_eligibility_ms = 1000.0f;

  // Pre fires, then post fires 5ms later (potentiation pattern)
  arr.last_spike_time[0] = 0.0f;
  arr.spiked[0] = 0;
  arr.spiked[1] = 1;
  arr.last_spike_time[1] = 5.0f;

  float w_before = syn.weight[0];
  STDPUpdate(syn, arr, 5.0f, p);

  // Weight should NOT change (trace mode defers weight changes)
  assert(syn.weight[0] == w_before);
  // But trace should be positive (potentiation)
  assert(syn.eligibility_trace[0] > 0.0f);
}

TEST(eligibility_trace_converts_with_dopamine) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.da_scale = 5.0f;
  p.tau_eligibility_ms = 1000.0f;

  // Set a positive eligibility trace (as if a potentiating spike pair occurred)
  syn.eligibility_trace[0] = 0.01f;

  // No dopamine: trace should not convert to weight change
  float w_before = syn.weight[0];
  EligibilityTraceUpdate(syn, arr, 1.0f, p);
  assert(syn.weight[0] == w_before);

  // Now add dopamine at the postsynaptic neuron
  arr.dopamine[1] = 0.5f;
  EligibilityTraceUpdate(syn, arr, 1.0f, p);
  assert(syn.weight[0] > w_before && "Dopamine should convert trace to weight increase");
}

TEST(eligibility_trace_decays) {
  NeuronArray arr;
  arr.Resize(2);
  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {5.0f}, {kACh});
  syn.InitEligibilityTraces();

  STDPParams p;
  p.dopamine_gated = true;
  p.use_eligibility_traces = true;
  p.tau_eligibility_ms = 100.0f;  // fast decay for testing

  syn.eligibility_trace[0] = 1.0f;
  float trace_before = syn.eligibility_trace[0];

  // No dopamine, just decay
  EligibilityTraceUpdate(syn, arr, 50.0f, p);
  float trace_after = syn.eligibility_trace[0];
  assert(trace_after < trace_before);
  // exp(-50/100) ~ 0.607
  float expected = trace_before * std::exp(-50.0f / 100.0f);
  assert(std::abs(trace_after - expected) < 0.01f);
}

// ===== Synaptic scaling tests =====

TEST(synaptic_scaling_upscales_silent_neurons) {
  NeuronArray neurons;
  neurons.Resize(3);

  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 1}, {1, 2}, {2.0f, 2.0f}, {kACh, kACh});

  SynapticScaling scaling;
  scaling.Init(3);
  scaling.target_rate_hz = 10.0f;

  // Simulate 100ms with no spikes
  for (int i = 0; i < 1000; ++i) {
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[0];
  scaling.Apply(syn, p);
  // Silent neurons should have weights scaled up (toward target rate)
  assert(syn.weight[0] > w_before);
}

TEST(synaptic_scaling_downscales_overactive_neurons) {
  NeuronArray neurons;
  neurons.Resize(3);

  SynapseTable syn;
  syn.BuildFromCOO(3, {0, 1}, {1, 2}, {5.0f, 5.0f}, {kACh, kACh});

  SynapticScaling scaling;
  scaling.Init(3);
  scaling.target_rate_hz = 5.0f;

  // Simulate 100ms with neuron 2 firing every step (way above target)
  for (int i = 0; i < 1000; ++i) {
    neurons.spiked[2] = 1;
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[1];  // synapse 1->2
  scaling.Apply(syn, p);
  // Over-active neuron 2 should have incoming weight scaled down
  assert(syn.weight[1] < w_before && "Overactive neuron should have downscaled weights");
}

TEST(synaptic_scaling_clamps_range) {
  NeuronArray neurons;
  neurons.Resize(2);

  SynapseTable syn;
  syn.BuildFromCOO(2, {0}, {1}, {1.0f}, {kACh});

  SynapticScaling scaling;
  scaling.Init(2);
  scaling.min_scale = 0.8f;
  scaling.max_scale = 1.2f;

  // No spikes for 100ms
  for (int i = 0; i < 1000; ++i) {
    scaling.AccumulateSpikes(neurons, 0.1f);
  }

  STDPParams p;
  float w_before = syn.weight[0];
  scaling.Apply(syn, p);
  // Scale should be clamped to max_scale=1.2
  assert(syn.weight[0] <= w_before * 1.2f + 0.01f);
  assert(syn.weight[0] >= w_before * 0.8f - 0.01f);
}

// ---- Conditioning Experiment Tests ----

#include "conditioning_experiment.h"

TEST(conditioning_circuit_builds) {
  // Verify a small conditioning experiment runs without crashing
  ConditioningExperiment exp;
  exp.n_orn = 20;
  exp.n_pn = 10;
  exp.n_kc = 50;
  exp.n_mbon = 5;
  exp.n_dan = 5;
  exp.dt_ms = 0.1f;
  exp.test_duration_ms = 10.0f;
  exp.trial_duration_ms = 20.0f;
  exp.iti_ms = 5.0f;
  exp.n_training_trials = 1;

  auto result = exp.Run();
  assert(result.total_training_trials == 1);
  assert(result.pre_test_cs_plus_spikes >= 0);
  assert(result.post_test_cs_plus_spikes >= 0);
  assert(result.elapsed_seconds > 0.0);
}

TEST(conditioning_weight_changes) {
  // With dopamine-gated STDP, KC->MBON weights should change after training
  ConditioningExperiment exp;
  exp.n_orn = 40;
  exp.n_pn = 20;
  exp.n_kc = 100;
  exp.n_mbon = 10;
  exp.n_dan = 10;
  exp.dt_ms = 0.1f;
  exp.test_duration_ms = 100.0f;
  exp.trial_duration_ms = 200.0f;
  exp.iti_ms = 50.0f;
  exp.n_training_trials = 3;
  exp.seed = 123;

  auto result = exp.Run();
  assert(result.mean_weight_before > 0.0f);
  // Weights should have changed
  bool changed = (result.weight_change_ratio < 0.99f ||
                  result.weight_change_ratio > 1.01f);
  assert(changed && "KC->MBON weights should change after training");
}

TEST(conditioning_result_metrics) {
  // Verify result struct logic
  ConditioningResult r;
  r.learning_index = 0.5f;
  assert(r.learned());

  ConditioningResult r2;
  r2.learning_index = 0.01f;
  assert(!r2.learned());
}

// ===== Rate Monitor tests =====

TEST(rate_monitor_init) {
  NeuronArray neurons;
  neurons.Resize(100);
  for (size_t i = 0; i < 100; ++i)
    neurons.region[i] = static_cast<uint8_t>(i < 50 ? 0 : 1);

  RateMonitor mon;
  std::vector<std::string> names = {"KC", "MBON"};
  mon.Init(neurons, names, 1.0f);

  assert(mon.regions.size() == 2);
  assert(mon.regions[0].neuron_indices.size() == 50);
  assert(mon.regions[1].neuron_indices.size() == 50);
  assert(mon.regions[0].name == "KC");
}

TEST(rate_monitor_computes_rates) {
  NeuronArray neurons;
  neurons.Resize(100);
  for (size_t i = 0; i < 100; ++i)
    neurons.region[i] = 0;

  RateMonitor mon;
  mon.Init(neurons, 1.0f);

  // Simulate 1000ms with 10% of neurons spiking each step
  for (int step = 0; step < 1000; ++step) {
    for (size_t i = 0; i < 100; ++i)
      neurons.spiked[i] = (i < 10) ? 1 : 0;
    mon.RecordStep(neurons);
  }

  auto rates = mon.ComputeRates();
  assert(!rates.empty());
  // 10 out of 100 neurons spike every 1ms step = 10% * 1000 Hz = 100 Hz
  assert(rates[0].rate_hz > 90.0f && rates[0].rate_hz < 110.0f);
}

TEST(rate_monitor_literature_lookup) {
  NeuronArray neurons;
  neurons.Resize(50);
  for (size_t i = 0; i < 50; ++i)
    neurons.region[i] = 0;

  RateMonitor mon;
  std::vector<std::string> names = {"KC"};
  mon.Init(neurons, names, 1.0f);

  // KC reference: 0.5 to 10 Hz
  assert(mon.regions[0].ref_min > 0.0f);
  assert(mon.regions[0].ref_max <= 10.0f);
}

TEST(rate_monitor_in_range_count) {
  std::vector<RegionRate> rates;
  RegionRate r1;
  r1.rate_hz = 5.0f; r1.ref_min_hz = 1.0f; r1.ref_max_hz = 10.0f;
  rates.push_back(r1);
  RegionRate r2;
  r2.rate_hz = 50.0f; r2.ref_min_hz = 1.0f; r2.ref_max_hz = 10.0f;
  rates.push_back(r2);

  assert(RateMonitor::CountInRange(rates) == 1);
}

// ===== Motor Output tests =====

TEST(motor_output_init) {
  MotorOutput motor;
  motor.Init({0, 1, 2}, {3, 4, 5}, {6, 7}, {8, 9});
  assert(motor.HasMotorNeurons());
  assert(motor.TotalNeurons() == 10);
  assert(motor.descending_left.size() == 3);
  assert(motor.avoid_neurons.size() == 2);
}

TEST(motor_output_forward_velocity) {
  NeuronArray neurons;
  neurons.Resize(20);

  MotorOutput motor;
  // 10 left descending, 10 right descending
  std::vector<uint32_t> left, right;
  for (uint32_t i = 0; i < 10; ++i) left.push_back(i);
  for (uint32_t i = 10; i < 20; ++i) right.push_back(i);
  motor.Init(left, right, {}, {});

  // All neurons spiking: should produce forward velocity
  for (size_t i = 0; i < 20; ++i)
    neurons.spiked[i] = 1;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().forward_velocity > 0.0f);
}

TEST(motor_output_turning) {
  NeuronArray neurons;
  neurons.Resize(20);

  MotorOutput motor;
  std::vector<uint32_t> left, right;
  for (uint32_t i = 0; i < 10; ++i) left.push_back(i);
  for (uint32_t i = 10; i < 20; ++i) right.push_back(i);
  motor.Init(left, right, {}, {});

  // Only left neurons spike: should turn left (positive angular velocity)
  for (size_t i = 0; i < 10; ++i)
    neurons.spiked[i] = 1;
  for (size_t i = 10; i < 20; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().angular_velocity > 0.0f);
}

TEST(motor_output_approach_avoid) {
  NeuronArray neurons;
  neurons.Resize(10);

  MotorOutput motor;
  motor.Init({}, {}, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9});

  // Only approach neurons spike
  for (size_t i = 0; i < 5; ++i)
    neurons.spiked[i] = 1;
  for (size_t i = 5; i < 10; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().approach_drive > 0.0f);

  // Now only avoid neurons spike
  for (size_t i = 0; i < 5; ++i)
    neurons.spiked[i] = 0;
  for (size_t i = 5; i < 10; ++i)
    neurons.spiked[i] = 1;

  for (int s = 0; s < 100; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().approach_drive < 0.0f);
}

TEST(motor_output_freeze_when_silent) {
  NeuronArray neurons;
  neurons.Resize(10);

  MotorOutput motor;
  motor.Init({0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {}, {});

  // No neurons spiking
  for (size_t i = 0; i < 10; ++i)
    neurons.spiked[i] = 0;

  for (int s = 0; s < 200; ++s)
    motor.Update(neurons, 1.0f);

  assert(motor.Command().freeze > 0.5f);
}

TEST(motor_output_from_regions) {
  NeuronArray neurons;
  neurons.Resize(20);
  // SEZ region = 12, MBON region = 3
  for (size_t i = 0; i < 10; ++i) {
    neurons.region[i] = 12;
    neurons.x[i] = (i < 5) ? 100.0f : 400.0f;  // L/R split at 250
  }
  for (size_t i = 10; i < 20; ++i) {
    neurons.region[i] = 3;
    neurons.type[i] = (i < 15) ? 2 : 3;  // cholinergic vs GABAergic
  }

  MotorOutput motor;
  motor.InitFromRegions(neurons, 12, 3, 250.0f);

  assert(motor.descending_left.size() == 5);
  assert(motor.descending_right.size() == 5);
  assert(motor.approach_neurons.size() == 5);
  assert(motor.avoid_neurons.size() == 5);
}

// ===== Intrinsic Homeostasis tests =====

TEST(homeostasis_init) {
  IntrinsicHomeostasis homeo;
  homeo.Init(100, 5.0f, 0.1f);
  assert(homeo.bias_current.size() == 100);
  assert(homeo.MeanBias() == 0.0f);
}

TEST(homeostasis_silent_neurons_get_positive_bias) {
  NeuronArray neurons;
  neurons.Resize(10);
  // All neurons silent

  IntrinsicHomeostasis homeo;
  homeo.Init(10, 5.0f, 1.0f);

  // Record 1000ms of silence
  for (int i = 0; i < 1000; ++i)
    homeo.RecordSpikes(neurons);

  homeo.Apply(neurons);

  // Silent neurons should get positive bias (to encourage firing)
  assert(homeo.MeanBias() > 0.0f);
  assert(homeo.FractionExcited() == 1.0f);
}

TEST(homeostasis_active_neurons_get_negative_bias) {
  NeuronArray neurons;
  neurons.Resize(10);

  IntrinsicHomeostasis homeo;
  homeo.Init(10, 5.0f, 1.0f);
  homeo.target_rate_hz = 5.0f;

  // All neurons fire every step for 1000ms (= 1000 Hz, way above target)
  for (int i = 0; i < 1000; ++i) {
    for (size_t j = 0; j < 10; ++j) neurons.spiked[j] = 1;
    homeo.RecordSpikes(neurons);
  }

  homeo.Apply(neurons);

  // Overactive neurons should get negative bias
  assert(homeo.MeanBias() < 0.0f);
  assert(homeo.FractionExcited() == 0.0f);
}

TEST(homeostasis_bias_clamps) {
  NeuronArray neurons;
  neurons.Resize(2);

  IntrinsicHomeostasis homeo;
  homeo.Init(2, 5.0f, 1.0f);
  homeo.max_bias = 3.0f;

  // Apply many times with silence to push bias high
  for (int round = 0; round < 50; ++round) {
    for (int i = 0; i < 1000; ++i)
      homeo.RecordSpikes(neurons);
    homeo.Apply(neurons);
  }

  // Bias should be clamped
  for (float b : homeo.bias_current) {
    assert(b <= homeo.max_bias + 0.001f);
    assert(b >= -homeo.max_bias - 0.001f);
  }
}

TEST(homeostasis_maybe_apply_respects_interval) {
  NeuronArray neurons;
  neurons.Resize(5);

  IntrinsicHomeostasis homeo;
  homeo.Init(5, 5.0f, 1.0f);
  homeo.update_interval_ms = 100.0f;

  // 50ms of recording: not enough
  for (int i = 0; i < 50; ++i)
    homeo.RecordSpikes(neurons);
  assert(!homeo.MaybeApply(neurons));

  // 50 more ms: now at 100ms, should apply
  for (int i = 0; i < 50; ++i)
    homeo.RecordSpikes(neurons);
  assert(homeo.MaybeApply(neurons));
}

// ===== Multi-Trial tests =====

#include "multi_trial.h"

TEST(multi_trial_stats_computation) {
  // Test stat computation with known results
  std::vector<ConditioningResult> results;

  ConditioningResult r1;
  r1.learning_index = 0.1f;
  r1.discrimination_index = 0.2f;
  r1.behavioral_learning = 0.05f;
  r1.weight_change_ratio = 1.1f;
  r1.regions_in_range = 3;
  r1.elapsed_seconds = 0.5;
  results.push_back(r1);

  ConditioningResult r2;
  r2.learning_index = 0.3f;
  r2.discrimination_index = 0.4f;
  r2.behavioral_learning = 0.15f;
  r2.weight_change_ratio = 1.3f;
  r2.regions_in_range = 4;
  r2.elapsed_seconds = 0.6;
  results.push_back(r2);

  auto stats = MultiTrialRunner::ComputeStats(results);
  assert(stats.n_trials == 2);
  assert(std::abs(stats.learning_mean - 0.2f) < 0.01f);
  assert(stats.learning_min == 0.1f);
  assert(stats.learning_max == 0.3f);
  assert(stats.n_learned == 2);  // both > 0.05
  assert(std::abs(stats.success_rate - 1.0f) < 0.01f);
}

TEST(multi_trial_empty_results) {
  std::vector<ConditioningResult> results;
  auto stats = MultiTrialRunner::ComputeStats(results);
  assert(stats.n_trials == 0);
  assert(stats.learning_mean == 0.0f);
}

TEST(multi_trial_single_result) {
  std::vector<ConditioningResult> results;
  ConditioningResult r;
  r.learning_index = 0.5f;
  r.discrimination_index = 0.3f;
  r.behavioral_learning = 0.1f;
  r.weight_change_ratio = 1.2f;
  r.regions_in_range = 5;
  r.elapsed_seconds = 1.0;
  results.push_back(r);

  auto stats = MultiTrialRunner::ComputeStats(results);
  assert(stats.n_trials == 1);
  assert(stats.learning_mean == 0.5f);
  assert(stats.learning_std == 0.0f);
  assert(stats.n_learned == 1);
}

// ===== Gap junction tests =====

TEST(gap_junction_bidirectional_current) {
  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -60.0f;
  arr.v[1] = -40.0f;  // higher voltage
  arr.i_ext[0] = 0.0f;
  arr.i_ext[1] = 0.0f;

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 0.5f);
  gj.PropagateGapCurrents(arr);

  // Current flows from high V (neuron 1) to low V (neuron 0)
  // I = 0.5 * (-40 - (-60)) = 0.5 * 20 = 10
  assert(std::abs(arr.i_ext[0] - 10.0f) < 0.01f);   // neuron 0 gains current
  assert(std::abs(arr.i_ext[1] - (-10.0f)) < 0.01f); // neuron 1 loses current
}

TEST(gap_junction_equal_voltage_no_current) {
  NeuronArray arr;
  arr.Resize(2);
  arr.v[0] = -50.0f;
  arr.v[1] = -50.0f;
  arr.i_ext[0] = 0.0f;
  arr.i_ext[1] = 0.0f;

  GapJunctionTable gj;
  gj.AddJunction(0, 1, 1.0f);
  gj.PropagateGapCurrents(arr);

  assert(std::abs(arr.i_ext[0]) < 0.001f);
  assert(std::abs(arr.i_ext[1]) < 0.001f);
}

TEST(gap_junction_build_from_region) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 10; ++i) arr.region[i] = 0;
  for (size_t i = 10; i < 20; ++i) arr.region[i] = 1;

  GapJunctionTable gj;
  gj.BuildFromRegion(arr, 0, 1.0f, 0.3f);  // density=1.0, all pairs connected
  // 10 neurons, C(10,2) = 45 pairs
  assert(gj.Size() == 45);
  assert(gj.conductance[0] == 0.3f);
}

TEST(gap_junction_empty) {
  NeuronArray arr;
  arr.Resize(5);
  GapJunctionTable gj;
  gj.PropagateGapCurrents(arr);  // should not crash
  assert(gj.Size() == 0);
}

// ===== Short-term plasticity tests =====

TEST(stp_init_defaults) {
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());
  assert(syn.HasSTP());
  assert(syn.stp_u[0] == 0.5f);
  assert(syn.stp_x[0] == 1.0f);
}

TEST(stp_depression_on_spike) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 1;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  float x_before = syn.stp_x[0];
  UpdateSTP(syn, neurons, 0.1f);
  float x_after = syn.stp_x[0];

  // After spike, x should decrease (depression)
  assert(x_after < x_before);
  // Effective weight = base * u * x
  float w_eff = syn.weight[0] * syn.stp_u[0] * syn.stp_x[0];
  assert(w_eff < 1.0f);
}

TEST(stp_recovery_without_spikes) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 0;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  // Manually depress
  syn.stp_x[0] = 0.3f;
  syn.stp_u[0] = 0.8f;

  // Run many steps without spikes
  for (int i = 0; i < 10000; ++i)
    UpdateSTP(syn, neurons, 1.0f);

  // Should recover toward resting state
  assert(std::abs(syn.stp_x[0] - 1.0f) < 0.01f);
  assert(std::abs(syn.stp_u[0] - 0.5f) < 0.01f);
}

TEST(stp_facilitating_increases_u) {
  NeuronArray neurons;
  neurons.Resize(2);
  neurons.spiked[0] = 1;

  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 1, 1};
  syn.post = {1};
  syn.weight = {1.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPFacilitating());

  float u_before = syn.stp_u[0];
  UpdateSTP(syn, neurons, 0.1f);
  float u_after = syn.stp_u[0];

  // Facilitation: u should increase on spike
  assert(u_after > u_before);
}

TEST(stp_effective_weight_bounded) {
  SynapseTable syn;
  syn.n_neurons = 1;
  syn.row_ptr = {0, 1};
  syn.post = {0};
  syn.weight = {5.0f};
  syn.nt_type = {static_cast<uint8_t>(NTType::kACh)};
  syn.InitSTP(STPDepressing());

  float w_eff = syn.weight[0] * syn.stp_u[0] * syn.stp_x[0];
  assert(w_eff <= 5.0f);
  assert(w_eff >= 0.0f);
}

TEST(stp_reset) {
  SynapseTable syn;
  syn.n_neurons = 2;
  syn.row_ptr = {0, 5, 10};
  syn.post.resize(10);
  syn.weight.resize(10, 1.0f);
  syn.nt_type.resize(10, static_cast<uint8_t>(NTType::kACh));
  syn.InitSTP(STPCombined());

  syn.stp_u[5] = 0.99f;
  syn.stp_x[5] = 0.01f;
  ResetSTP(syn);
  assert(std::abs(syn.stp_u[5] - 0.25f) < 0.001f);
  assert(std::abs(syn.stp_x[5] - 1.0f) < 0.001f);
}

TEST(stp_presets) {
  STPParams fac = STPFacilitating();
  STPParams dep = STPDepressing();
  STPParams com = STPCombined();
  assert(fac.U_se < dep.U_se);  // facilitating has lower baseline release
  assert(fac.tau_f > dep.tau_f);  // facilitating has slower facilitation decay
  assert(com.tau_f > dep.tau_f && com.tau_f < fac.tau_f);
}

// ===== CPG tests =====

TEST(cpg_init_splits_groups) {
  NeuronArray arr;
  arr.Resize(100);
  for (size_t i = 0; i < 100; ++i) {
    arr.region[i] = 5;  // all VNC
    arr.x[i] = static_cast<float>(i * 5);  // spread across x
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.3f);
  assert(cpg.initialized);
  assert(!cpg.group_a.empty());
  assert(!cpg.group_b.empty());
  // Groups should be non-overlapping
  size_t total = cpg.group_a.size() + cpg.group_b.size();
  assert(total == 70);  // 100 - 30% sensory = 70 motor neurons
}

TEST(cpg_zero_drive_no_current) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 25);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.0f);
  cpg.Step(arr, 1.0f, 0.0f);  // zero descending drive

  // With drive=0, CPG should be silent
  float total_current = 0.0f;
  for (size_t i = 0; i < 20; ++i) total_current += arr.i_ext[i];
  assert(std::abs(total_current) < 0.01f);
}

TEST(cpg_full_drive_injects_current) {
  NeuronArray arr;
  arr.Resize(20);
  for (size_t i = 0; i < 20; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 25);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.0f);
  cpg.drive_scale = 1.0f;  // force full drive immediately
  cpg.Step(arr, 1.0f, 1.0f);

  // Some neurons should receive positive current
  float total_current = 0.0f;
  for (size_t i = 0; i < 20; ++i) total_current += arr.i_ext[i];
  assert(total_current > 0.0f);
}

TEST(cpg_antiphase_groups) {
  NeuronArray arr;
  arr.Resize(100);
  for (size_t i = 0; i < 100; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 5);
    arr.i_ext[i] = 0.0f;
  }

  CPGOscillator cpg;
  cpg.Init(arr, 5, 250.0f, 0.3f);
  cpg.drive_scale = 1.0f;
  cpg.phase = 0.5f;  // set phase so groups get different currents
  cpg.Step(arr, 1.0f, 1.0f);

  // Sum current for each group
  float sum_a = 0.0f, sum_b = 0.0f;
  for (uint32_t i : cpg.group_a) sum_a += arr.i_ext[i];
  for (uint32_t i : cpg.group_b) sum_b += arr.i_ext[i];

  // Groups should receive different current levels (anti-phase)
  assert(std::abs(sum_a - sum_b) > 0.1f);
}

// ===== Proprioception tests =====

TEST(proprio_init_assigns_channels) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);
  assert(pm.initialized);
  // Should have assigned neurons to joint angle channels
  int assigned = 0;
  for (int j = 0; j < 42; ++j) assigned += static_cast<int>(pm.joint_angle_neurons[j].size());
  assert(assigned > 0);
}

TEST(proprio_inject_contact) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};
  state.contacts[0] = 1.0f;  // left front leg touching ground

  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Contact neurons for leg 0 should have received current
  if (!pm.contact_neurons[0].empty()) {
    float current = arr.i_ext[pm.contact_neurons[0][0]];
    assert(current > 0.0f);
  }
}

TEST(proprio_haltere_asymmetry) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};
  state.body_velocity[2] = 2.0f;  // positive yaw (turning left)

  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Positive yaw should excite right haltere more than left
  float sum_left = 0.0f, sum_right = 0.0f;
  for (uint32_t i : pm.haltere_left) sum_left += arr.i_ext[i];
  for (uint32_t i : pm.haltere_right) sum_right += arr.i_ext[i];
  assert(sum_right > sum_left);
}

TEST(proprio_zero_state_minimal_current) {
  NeuronArray arr;
  arr.Resize(200);
  for (size_t i = 0; i < 200; ++i) {
    arr.region[i] = 5;
    arr.x[i] = static_cast<float>(i * 2.5f);
    arr.i_ext[i] = 0.0f;
  }

  ProprioMap pm;
  pm.Init(arr, 5, 250.0f);

  ProprioState state{};  // all zeros
  ProprioConfig cfg;
  pm.Inject(arr, state, cfg);

  // Zero joint angles still produce some current due to sigmoid offset,
  // but contacts and haltere should be zero
  float haltere_current = 0.0f;
  for (uint32_t i : pm.haltere_left) haltere_current += arr.i_ext[i];
  for (uint32_t i : pm.haltere_right) haltere_current += arr.i_ext[i];
  assert(std::abs(haltere_current) < 0.01f);
}

// ---- Main ----

int main() {
  return RunAllTests();
}
