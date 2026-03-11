#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/neuron_array.h"
#include "core/synapse_table.h"
#include "core/izhikevich.h"
#include "core/stdp.h"
#include "core/parametric_gen.h"
#include "bridge/twin_bridge.h"
#include "bridge/opsin_model.h"
#include "bridge/validation.h"

using namespace fwmc;
using Clock = std::chrono::steady_clock;

template <typename F>
double Bench(const char* name, int iterations, F&& fn) {
  auto t0 = Clock::now();
  for (int i = 0; i < iterations; ++i) fn();
  auto t1 = Clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double per_iter = ms / iterations;
  printf("  %-40s  %8.3f ms/iter  (%d iters, %.1f ms total)\n",
         name, per_iter, iterations, ms);
  return per_iter;
}

static void BuildRandomConnectome(size_t n_neurons, double conn_prob,
                                  SynapseTable& table) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> weight_dist(0.5f, 3.0f);
  std::bernoulli_distribution conn_dist(conn_prob);

  std::vector<uint32_t> pre, post;
  std::vector<float> weight;
  std::vector<uint8_t> nt;

  for (size_t i = 0; i < n_neurons; ++i) {
    for (size_t j = 0; j < n_neurons; ++j) {
      if (i != j && conn_dist(rng)) {
        pre.push_back(static_cast<uint32_t>(i));
        post.push_back(static_cast<uint32_t>(j));
        weight.push_back(weight_dist(rng));
        nt.push_back((rng() % 4 == 0) ? kGABA : kACh);
      }
    }
  }
  table.BuildFromCOO(n_neurons, pre, post, weight, nt);
}

int main() {
  printf("=== FWMC Benchmarks ===\n\n");

  // Bench: IzhikevichStep at various scales
  printf("--- Izhikevich Step ---\n");
  for (size_t n : {1000u, 10000u, 100000u}) {
    NeuronArray arr;
    arr.Resize(n);
    for (size_t i = 0; i < n; ++i) arr.i_ext[i] = 5.0f;
    IzhikevichParams p;

    char name[64];
    snprintf(name, sizeof(name), "IzhikevichStep N=%zu", n);
    int iters = (n <= 10000) ? 10000 : 1000;
    double ms = Bench(name, iters, [&]() {
      IzhikevichStep(arr, 0.1f, 0.0f, p);
    });

    double neurons_per_sec = n / (ms / 1000.0);
    printf("    -> %.2f M neurons/sec\n", neurons_per_sec / 1e6);
  }

  // Bench: Spike propagation
  printf("\n--- Spike Propagation ---\n");
  for (double prob : {0.01, 0.05, 0.10}) {
    size_t n = 1000;
    SynapseTable table;
    BuildRandomConnectome(n, prob, table);

    NeuronArray arr;
    arr.Resize(n);
    // Make ~10% of neurons spike
    for (size_t i = 0; i < n; i += 10) arr.spiked[i] = 1;

    char name[64];
    snprintf(name, sizeof(name), "PropagateSpikes N=%zu p=%.0f%% (%zu syn)",
             n, prob * 100, table.Size());
    Bench(name, 5000, [&]() {
      std::fill(arr.i_syn.begin(), arr.i_syn.end(), 0.0f);
      table.PropagateSpikes(arr.spiked.data(), arr.i_syn.data(), 1.0f);
    });
  }

  // Bench: STDP
  printf("\n--- STDP Update ---\n");
  {
    size_t n = 1000;
    SynapseTable table;
    BuildRandomConnectome(n, 0.05, table);

    NeuronArray arr;
    arr.Resize(n);
    for (size_t i = 0; i < n; i += 10) {
      arr.spiked[i] = 1;
      arr.last_spike_time[i] = 50.0f;
    }
    STDPParams sp;

    char name[64];
    snprintf(name, sizeof(name), "STDPUpdate N=%zu (%zu syn)", n, table.Size());
    Bench(name, 1000, [&]() {
      STDPUpdate(table, arr, 55.0f, sp);
    });
  }

  // Bench: Large-scale spike propagation (OpenMP test)
  printf("\n--- Large-Scale Propagation (OpenMP) ---\n");
  {
    size_t n = 50000;
    SynapseTable table;
    // Build sparse random connectome: ~0.1% density = ~2.5M synapses
    std::mt19937 rng_large(123);
    std::vector<uint32_t> pre, post_arr;
    std::vector<float> w;
    std::vector<uint8_t> nt;
    std::geometric_distribution<int> gap_dist(0.001);
    std::uniform_int_distribution<uint32_t> post_dist(0, static_cast<uint32_t>(n - 1));
    std::normal_distribution<float> w_dist(1.0f, 0.3f);

    // Geometric skip sampling for speed
    for (size_t i = 0; i < n; ++i) {
      int n_out = std::poisson_distribution<int>(n * 0.001)(rng_large);
      for (int j = 0; j < n_out; ++j) {
        uint32_t p = post_dist(rng_large);
        if (p == static_cast<uint32_t>(i)) continue;
        pre.push_back(static_cast<uint32_t>(i));
        post_arr.push_back(p);
        w.push_back(std::max(0.01f, w_dist(rng_large)));
        nt.push_back(kACh);
      }
    }
    table.BuildFromCOO(n, pre, post_arr, w, nt);

    NeuronArray arr;
    arr.Resize(n);
    for (size_t i = 0; i < n; i += 10) arr.spiked[i] = 1;

    char name[128];
    snprintf(name, sizeof(name), "PropagateSpikes N=%zu (%zu syn, OpenMP)",
             n, table.Size());
    Bench(name, 100, [&]() {
      std::fill(arr.i_syn.begin(), arr.i_syn.end(), 0.0f);
      table.PropagateSpikes(arr.spiked.data(), arr.i_syn.data(), 1.0f);
    });
  }

  // Bench: Full bridge step
  printf("\n--- Full Bridge Step ---\n");
  {
    size_t n = 10000;
    TwinBridge bridge;
    bridge.Init(n);
    bridge.dt_ms = 0.1f;
    bridge.mode = BridgeMode::kOpenLoop;
    BuildRandomConnectome(n, 0.005, bridge.synapses);

    for (size_t i = 0; i < n; i += 5) bridge.digital.i_ext[i] = 8.0f;

    char name[64];
    snprintf(name, sizeof(name), "TwinBridge::Step N=%zu (%zu syn)",
             n, bridge.synapses.Size());
    double ms = Bench(name, 1000, [&]() { bridge.Step(); });

    double realtime_ratio = 0.1 / ms;  // dt=0.1ms per step
    printf("    -> %.2fx real-time\n", realtime_ratio);
  }

  // Bench: Parametric brain generation
  printf("\n--- Parametric Brain Generation ---\n");
  {
    BrainSpec spec;
    spec.seed = 99;
    spec.regions.push_back({"region_A", 10000, 0.005f, kACh, {}, {}});
    spec.regions.push_back({"region_B", 5000, 0.008f, kACh, {}, {}});
    spec.projections.push_back({"region_A", "region_B", 0.002f, kACh, 1.0f, 0.2f});

    NeuronArray neurons;
    SynapseTable synapses;
    CellTypeManager types;
    ParametricGenerator gen;

    char name[64];
    snprintf(name, sizeof(name), "ParametricGen 15K neurons");
    Bench(name, 3, [&]() {
      gen.Generate(spec, neurons, synapses, types);
    });
    printf("    -> %zu synapses generated\n", synapses.Size());
  }

  // Bench: Full-scale connectome (140K neurons)
  printf("\n--- Full-Scale Connectome (140K neurons) ---\n");
  {
    size_t n = 140000;
    NeuronArray arr;
    arr.Resize(n);
    for (size_t i = 0; i < n; ++i) arr.i_ext[i] = 5.0f;
    IzhikevichParams p;

    double ms_izh = Bench("IzhikevichStep N=140K", 100, [&]() {
      IzhikevichStep(arr, 0.1f, 0.0f, p);
    });
    printf("    -> %.2f M neurons/sec\n", n / (ms_izh / 1000.0) / 1e6);

    // Build sparse connectome: mean out-degree 50
    printf("  Building 140K-neuron connectome (mean out-degree 50)...\n");
    SynapseTable table;
    {
      std::mt19937 rng_fs(777);
      std::vector<uint32_t> pre_fs, post_fs;
      std::vector<float> w_fs;
      std::vector<uint8_t> nt_fs;
      std::poisson_distribution<int> degree_dist(50);
      std::uniform_int_distribution<uint32_t> target_dist(0, static_cast<uint32_t>(n - 1));
      std::normal_distribution<float> w_dist2(1.5f, 0.5f);

      pre_fs.reserve(n * 50);
      post_fs.reserve(n * 50);
      w_fs.reserve(n * 50);
      nt_fs.reserve(n * 50);

      for (size_t i = 0; i < n; ++i) {
        int n_out = degree_dist(rng_fs);
        for (int j = 0; j < n_out; ++j) {
          uint32_t tgt = target_dist(rng_fs);
          if (tgt == static_cast<uint32_t>(i)) continue;
          pre_fs.push_back(static_cast<uint32_t>(i));
          post_fs.push_back(tgt);
          w_fs.push_back(std::max(0.01f, w_dist2(rng_fs)));
          nt_fs.push_back((rng_fs() % 5 == 0) ? kGABA : kACh);
        }
      }
      table.BuildFromCOO(n, pre_fs, post_fs, w_fs, nt_fs);
    }
    printf("    Connectome: %zu neurons, %zu synapses\n", n, table.Size());

    // Spike propagation at full scale
    for (size_t i = 0; i < n; i += 20) arr.spiked[i] = 1;
    {
      char name2[128];
      snprintf(name2, sizeof(name2), "PropagateSpikes N=140K (%zuM syn, 5%%)",
               table.Size() / 1000000);
      Bench(name2, 20, [&]() {
        std::fill(arr.i_syn.begin(), arr.i_syn.end(), 0.0f);
        table.PropagateSpikes(arr.spiked.data(), arr.i_syn.data(), 1.0f);
      });
    }

    // STDP at full scale
    for (size_t i = 0; i < n; i += 20) arr.last_spike_time[i] = 50.0f;
    STDPParams sp_full;
    {
      char name2[128];
      snprintf(name2, sizeof(name2), "STDPUpdate N=140K (%zuM syn)", table.Size() / 1000000);
      Bench(name2, 10, [&]() {
        STDPUpdate(table, arr, 55.0f, sp_full);
      });
    }

    // Full simulation step
    {
      auto t0 = Clock::now();
      int n_steps = 10;
      for (int s = 0; s < n_steps; ++s) {
        arr.ClearSynapticInput();
        table.PropagateSpikes(arr.spiked.data(), arr.i_syn.data(), 1.0f);
        IzhikevichStep(arr, 0.1f, s * 0.1f, p);
        STDPUpdate(table, arr, s * 0.1f, sp_full);
      }
      auto t1 = Clock::now();
      double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      double per_step = total_ms / n_steps;
      printf("  %-40s  %8.3f ms/step  (%d steps)\n",
             "Full step (propagate+izh+stdp) N=140K", per_step, n_steps);
      printf("    -> %.2fx real-time (dt=0.1ms)\n", 0.1 / per_step);
    }
  }

  // Bench: Opsin kinetics at scale
  printf("\n--- Opsin Kinetics ---\n");
  {
    OpsinPopulation pop;
    pop.Init(140000, OpsinType::kChRmine);
    NeuronArray arr;
    arr.Resize(140000);
    for (size_t i = 0; i < 140000; i += 10) {
      pop.SetIrradiance(static_cast<uint32_t>(i), 5.0f);
    }
    Bench("OpsinPopulation::Step N=140K", 100, [&]() {
      pop.Step(0.1f, arr.v.data(), arr.i_ext.data(), arr.n);
    });
  }

  // Bench: Validation engine
  printf("\n--- Validation Engine ---\n");
  {
    std::mt19937 rng_val(42);
    std::uniform_real_distribution<float> spike_time(0.0f, 1000.0f);
    std::vector<SpikeTrain> sim_trains(1000), rec_trains(1000);
    for (size_t i = 0; i < 1000; ++i) {
      sim_trains[i].neuron_idx = static_cast<uint32_t>(i);
      rec_trains[i].neuron_idx = static_cast<uint32_t>(i);
      for (int j = 0; j < 50; ++j) {
        float t = spike_time(rng_val);
        sim_trains[i].times_ms.push_back(t);
        rec_trains[i].times_ms.push_back(t + (static_cast<int>(rng_val() % 10) - 5) * 0.5f);
      }
      std::sort(sim_trains[i].times_ms.begin(), sim_trains[i].times_ms.end());
      std::sort(rec_trains[i].times_ms.begin(), rec_trains[i].times_ms.end());
    }
    ValidationEngine engine;
    Bench("ValidatePopulation 1K neurons x 50 spikes", 5, [&]() {
      engine.ValidatePopulation(sim_trains, rec_trains, 1000.0f);
    });
  }

  printf("\n=== Done ===\n");
  return 0;
}
