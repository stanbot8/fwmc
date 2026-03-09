#ifndef FWMC_SYNAPSE_TABLE_H_
#define FWMC_SYNAPSE_TABLE_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace fwmc {

// Neurotransmitter types (from FlyWire classification, Eckstein et al. 2024)
enum NTType : uint8_t {
  kACh = 0,    // acetylcholine (excitatory)
  kGABA = 1,   // GABA (inhibitory)
  kGlut = 2,   // glutamate (variable, treated as excitatory in Drosophila)
  kDA = 3,     // dopamine (modulatory)
  k5HT = 4,    // serotonin (modulatory)
  kOA = 5,     // octopamine (modulatory)
  kUnknown = 255
};

// Tsodyks-Markram short-term plasticity parameters (per synapse class).
// u: utilization variable (fraction of available resources used per spike)
// x: available resources (fraction of vesicle pool that is ready)
// U_se: baseline utilization (release probability at rest)
// tau_d: depression recovery time constant (ms)
// tau_f: facilitation decay time constant (ms)
struct STPParams {
  float U_se = 0.5f;    // baseline release probability
  float tau_d = 200.0f; // depression recovery (ms), vesicle replenishment
  float tau_f = 50.0f;  // facilitation decay (ms), residual calcium clearance
};

// CSR (Compressed Sparse Row) synapse storage.
// Sorted by pre-synaptic neuron for cache-friendly spike propagation.
// 50M synapses at 9 bytes each = ~450MB (plus optional STP/release state).
struct SynapseTable {
  // CSR index: neuron i's outgoing synapses are in [row_ptr[i], row_ptr[i+1])
  std::vector<uint32_t> row_ptr;

  // Synapse data (sorted by pre-synaptic neuron)
  std::vector<uint32_t> post;     // post-synaptic neuron index
  std::vector<float> weight;      // synaptic weight
  std::vector<uint8_t> nt_type;   // neurotransmitter type

  // Stochastic release: per-synapse release probability [0,1].
  // If empty, all synapses transmit deterministically (p=1).
  // Biological basis: vesicle fusion probability depends on presynaptic
  // calcium concentration and number of docked vesicles per active zone.
  // KC-PN synapses: p_rel ~ 0.3 (sparse coding mechanism).
  std::vector<float> p_release;

  // Short-term plasticity state (Tsodyks-Markram model).
  // If empty, STP is disabled. When enabled, effective weight is
  // weight * u * x, where u tracks facilitation (residual calcium)
  // and x tracks depression (vesicle pool depletion).
  std::vector<float> stp_u;      // utilization variable (facilitation)
  std::vector<float> stp_x;      // available resources (depression)
  std::vector<float> stp_U_se;   // baseline utilization per synapse
  std::vector<float> stp_tau_d;  // depression recovery time constant
  std::vector<float> stp_tau_f;  // facilitation decay time constant

  // Synaptic delay: per-synapse delay in timesteps.
  // If empty, all synapses transmit instantaneously (delay=0).
  // Drosophila default: ~1.8ms uniform delay.
  std::vector<uint8_t> delay_steps;

  // Delay ring buffer: per-neuron circular buffer for incoming current.
  // delay_buffer[neuron * ring_size + slot] holds pending current.
  std::vector<float> delay_buffer;
  size_t ring_size = 0;       // slots in ring buffer (max_delay_steps + 1)
  size_t ring_head = 0;       // current read position

  // Eligibility traces for three-factor learning (Izhikevich 2007).
  // Each synapse accumulates a trace from STDP spike pairs. The trace
  // decays exponentially and is converted to weight change when
  // dopamine is present at the postsynaptic neuron.
  std::vector<float> eligibility_trace;

  size_t n_neurons = 0;

  size_t Size() const { return post.size(); }

  bool HasStochasticRelease() const { return !p_release.empty(); }
  bool HasSTP() const { return !stp_u.empty(); }
  bool HasDelays() const { return !delay_steps.empty(); }
  bool HasEligibilityTraces() const { return !eligibility_trace.empty(); }

  void InitEligibilityTraces() {
    eligibility_trace.assign(post.size(), 0.0f);
  }

  // Sign of synapse based on neurotransmitter.
  // In Drosophila, glutamate acts on GluCl receptors (inhibitory chloride
  // channels) on many postsynaptic targets, making it functionally inhibitory.
  static float Sign(uint8_t nt) {
    return (nt == kGABA || nt == kGlut) ? -1.0f : 1.0f;
  }

  // Initialize stochastic release with a uniform probability for all synapses.
  void InitReleaseProbability(float p) {
    p_release.assign(post.size(), p);
  }

  // Initialize short-term plasticity state for all synapses.
  void InitSTP(const STPParams& params) {
    size_t n = post.size();
    stp_u.assign(n, params.U_se);
    stp_x.assign(n, 1.0f);           // fully recovered at start
    stp_U_se.assign(n, params.U_se);
    stp_tau_d.assign(n, params.tau_d);
    stp_tau_f.assign(n, params.tau_f);
  }

  // Initialize uniform synaptic delay for all synapses.
  // delay_ms: desired delay in milliseconds
  // dt_ms: simulation timestep in milliseconds
  // Drosophila default: 1.8ms (comparable to membrane time constants).
  void InitDelay(float delay_ms, float dt_ms) {
    uint8_t steps = static_cast<uint8_t>(std::max(1.0f, delay_ms / dt_ms));
    delay_steps.assign(post.size(), steps);
    ring_size = static_cast<size_t>(steps) + 1;
    delay_buffer.assign(n_neurons * ring_size, 0.0f);
    ring_head = 0;
  }

  // Deliver delayed current: read the current slot, add to i_syn, then zero it.
  // Call this once per timestep BEFORE PropagateSpikes.
  void DeliverDelayed(float* i_syn) {
    if (!HasDelays()) return;
    for (size_t i = 0; i < n_neurons; ++i) {
      size_t slot = i * ring_size + ring_head;
      i_syn[i] += delay_buffer[slot];
      delay_buffer[slot] = 0.0f;
    }
  }

  // Advance the ring buffer head. Call once per timestep AFTER PropagateSpikes.
  void AdvanceDelayRing() {
    if (!HasDelays()) return;
    ring_head = (ring_head + 1) % ring_size;
  }

  // Update STP state for a single synapse when a presynaptic spike arrives.
  // Returns the effective release fraction (u * x) for this spike.
  // Called before delivery; updates u (facilitation) then depletes x (depression).
  float UpdateSTP(size_t s) {
    // Facilitation: residual calcium adds to utilization
    stp_u[s] += stp_U_se[s] * (1.0f - stp_u[s]);
    stp_u[s] = std::clamp(stp_u[s], 0.0f, 1.0f);
    // Effective transmission = u * x (fraction utilized * fraction available)
    float ux = stp_u[s] * stp_x[s];
    // Depression: deplete the vesicle pool
    stp_x[s] = std::max(0.0f, stp_x[s] - ux);
    return ux;
  }

  // Recover STP state between spikes (called once per timestep for all synapses).
  // Uses exact exponential recovery: alpha = 1 - exp(-dt/tau), always in [0,1].
  void RecoverSTP(float dt_ms) {
    const size_t n = stp_u.size();
    for (size_t s = 0; s < n; ++s) {
      float alpha_f = 1.0f - std::exp(-dt_ms / stp_tau_f[s]);
      float alpha_d = 1.0f - std::exp(-dt_ms / stp_tau_d[s]);
      // u decays back toward U_se (facilitation fades as calcium clears)
      stp_u[s] += (stp_U_se[s] - stp_u[s]) * alpha_f;
      // x recovers toward 1.0 (vesicle pool refills)
      stp_x[s] += (1.0f - stp_x[s]) * alpha_d;
    }
  }

  // Build CSR from unsorted COO (coordinate) input.
  // After this call, synapses are sorted by pre-neuron for fast traversal.
  void BuildFromCOO(size_t num_neurons,
                    const std::vector<uint32_t>& pre_in,
                    const std::vector<uint32_t>& post_in,
                    const std::vector<float>& weight_in,
                    const std::vector<uint8_t>& nt_in) {
    BuildFromCOOImpl(num_neurons, pre_in, post_in, weight_in, nt_in, {});
  }

  // Extended BuildFromCOO that also reorders per-synapse release probabilities.
  void BuildFromCOO(size_t num_neurons,
                    const std::vector<uint32_t>& pre_in,
                    const std::vector<uint32_t>& post_in,
                    const std::vector<float>& weight_in,
                    const std::vector<uint8_t>& nt_in,
                    const std::vector<float>& p_release_in) {
    BuildFromCOOImpl(num_neurons, pre_in, post_in, weight_in, nt_in, p_release_in);
  }

 private:
  // Shared implementation for both BuildFromCOO overloads.
  // Computes the sort order once and reuses it for all column arrays.
  void BuildFromCOOImpl(size_t num_neurons,
                        const std::vector<uint32_t>& pre_in,
                        const std::vector<uint32_t>& post_in,
                        const std::vector<float>& weight_in,
                        const std::vector<uint8_t>& nt_in,
                        const std::vector<float>& p_release_in) {
    n_neurons = num_neurons;
    size_t nnz = pre_in.size();

    // Validate indices before allocating anything
    for (size_t i = 0; i < nnz; ++i) {
      if (pre_in[i] >= num_neurons || post_in[i] >= num_neurons) {
        post.clear(); weight.clear(); nt_type.clear(); p_release.clear();
        row_ptr.assign(num_neurons + 1, 0);
        return;
      }
    }

    // Sort by pre-synaptic index (single O(n log n) pass)
    std::vector<size_t> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](size_t a, size_t b) { return pre_in[a] < pre_in[b]; });

    post.resize(nnz);
    weight.resize(nnz);
    nt_type.resize(nnz);
    for (size_t i = 0; i < nnz; ++i) {
      post[i]   = post_in[order[i]];
      weight[i] = weight_in[order[i]];
      nt_type[i] = nt_in[order[i]];
    }

    if (!p_release_in.empty()) {
      p_release.resize(nnz);
      for (size_t i = 0; i < nnz; ++i) {
        p_release[i] = p_release_in[order[i]];
      }
    }

    // Build row pointers
    row_ptr.assign(num_neurons + 1, 0);
    for (size_t i = 0; i < nnz; ++i) {
      row_ptr[pre_in[order[i]] + 1]++;
    }
    for (size_t i = 1; i <= num_neurons; ++i) {
      row_ptr[i] += row_ptr[i - 1];
    }
  }

 public:
  // Propagate spikes: for each neuron that spiked, deliver weighted
  // current to all post-synaptic targets.
  // This is the hot loop. CSR layout means sequential memory access
  // for each pre-neuron's outgoing synapses.
  //
  // OpenMP parallelization: each pre-neuron's outgoing synapses are
  // independent reads, but multiple pre-neurons can target the same
  // post-neuron (write conflict on i_syn). We use atomic adds for
  // correctness. On x86, atomic float add compiles to lock cmpxchg
  // loop which is ~2-3x slower per write than plain add, but the
  // parallelism across pre-neurons more than compensates at scale.
  void PropagateSpikes(const uint8_t* spiked, float* i_syn,
                       float weight_scale) {
    const bool has_delay = HasDelays();
    const int n = static_cast<int>(n_neurons);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) if(n > 10000)
    #endif
    for (int pre = 0; pre < n; ++pre) {
      if (!spiked[pre]) continue;
      uint32_t start = row_ptr[pre];
      uint32_t end = row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        float val = Sign(nt_type[s]) * weight[s] * weight_scale;
        if (has_delay) {
          // Write into delay ring buffer at the future slot
          size_t future = (ring_head + delay_steps[s]) % ring_size;
          size_t slot = post[s] * ring_size + future;
          #ifdef _OPENMP
          #pragma omp atomic
          #endif
          delay_buffer[slot] += val;
        } else {
          #ifdef _OPENMP
          #pragma omp atomic
          #endif
          i_syn[post[s]] += val;
        }
      }
    }
  }

  // Stochastic spike propagation: each synapse transmits with probability
  // p_release[s]. If STP is enabled, the effective weight is further
  // modulated by the utilization-resource product (u * x).
  // STP state is updated for ALL synapses of a spiking neuron (the
  // presynaptic spike triggers facilitation/depression regardless of
  // whether the vesicle was released). The stochastic gate only controls
  // whether current is delivered to the postsynaptic neuron.
  // rng must be thread-local or externally synchronized.
  void PropagateSpikesMonteCarlo(const uint8_t* spiked, float* i_syn,
                                  float weight_scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    const bool has_stp = HasSTP();
    const int n = static_cast<int>(n_neurons);

    for (int pre = 0; pre < n; ++pre) {
      if (!spiked[pre]) continue;
      uint32_t start = row_ptr[pre];
      uint32_t end = row_ptr[pre + 1];
      for (uint32_t s = start; s < end; ++s) {
        // STP state update happens for every synapse of a spiking neuron
        float stp_factor = 1.0f;
        if (has_stp) {
          stp_factor = UpdateSTP(s);
        }

        // Stochastic release gate (only gates current delivery)
        if (coin(rng) >= p_release[s]) continue;

        float val = Sign(nt_type[s]) * weight[s] * weight_scale * stp_factor;
        i_syn[post[s]] += val;
      }
    }
  }
};

}  // namespace fwmc

#endif  // FWMC_SYNAPSE_TABLE_H_
